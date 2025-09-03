#if os(visionOS)

import CompositorServices
import Metal
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import Spatial
import SwiftUI
import ARKit

// MARK: - Utility Extensions

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

// MARK: - Scene Renderer

class VisionSceneRenderer {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "VisionSceneRenderer")

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var modelTranslation: SIMD3<Float> = SIMD3<Float>(0, 0, Constants.modelCenterZ)

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    let handTracking: HandTrackingProvider

    var latestLeftHandAnchor: HandAnchor?
    var latestRightHandAnchor: HandAnchor?
    
    // MARK: - Pinch-driven grab + drag

    private var isGrabbing = false
    private var grabOffset: SIMD3<Float> = .zero

    // MARK: - Init

    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        handTracking = HandTrackingProvider()
        arSession = ARKitSession()
    }

    // MARK: - Pinch detection
    
    /// Get world-space position of a joint
    func jointWorldPosition(_ anchor: HandAnchor, joint: HandSkeleton.JointName) -> SIMD3<Float>? {
        guard let jointXf = anchor.handSkeleton?.joint(joint).anchorFromJointTransform else {
            return nil
        }
        // Convert anchor-local to world
        let worldXf = anchor.originFromAnchorTransform * jointXf
        return SIMD3<Float>(worldXf.columns.3.x,
                            worldXf.columns.3.y,
                            worldXf.columns.3.z)
    }
    
    /// Detect if hand is pinching
    func isPinching(_ anchor: HandAnchor) -> Bool {
        guard let thumbPos = jointWorldPosition(anchor, joint: .thumbTip),
              let indexPos = jointWorldPosition(anchor, joint: .indexFingerTip) else {
            return false
        }
        let distance = simd_length(thumbPos - indexPos)
        return distance < 0.025 // tweak threshold if needed
    }
    
    /// Update the model translation based on pinch state
    func updateGrabState(with anchor: HandAnchor, pinching: Bool) {
        guard let thumbPos = jointWorldPosition(anchor, joint: .thumbTip),
              let indexPos = jointWorldPosition(anchor, joint: .indexFingerTip) else { return }
        
        let pinchCenterWorld = (thumbPos + indexPos) / 2.0

        if pinching && !isGrabbing {
            // Just started pinching
            isGrabbing = true
            grabOffset = modelTranslation - pinchCenterWorld
        } else if pinching && isGrabbing {
            // While pinching, move the model
            let target = pinchCenterWorld + grabOffset
            modelTranslation = simd_mix(modelTranslation, target, SIMD3<Float>(repeating: 0.2))
        } else if !pinching && isGrabbing {
            // Released pinch
            isGrabbing = false
            grabOffset = .zero
        }
    }

    // MARK: - Model Loading

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try SplatRenderer(device: device,
                                          colorFormat: layerRenderer.configuration.colorFormat,
                                          depthFormat: layerRenderer.configuration.depthFormat,
                                          sampleCount: 1,
                                          maxViewCount: layerRenderer.properties.viewCount,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            try await splat.read(from: url)
            modelRenderer = splat
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: layerRenderer.configuration.colorFormat,
                                                   depthFormat: layerRenderer.configuration.depthFormat,
                                                   sampleCount: 1,
                                                   maxViewCount: layerRenderer.properties.viewCount,
                                                   maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    // MARK: - Render Loop

    func startRenderLoop() {
        Task {
            do {
                try await arSession.run([worldTracking, handTracking])
            } catch {
                fatalError("Failed to initialize ARSession")
            }

            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) -> [ModelRendererViewportDescriptor] {
        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        let translationMatrix = matrix4x4_translation(modelTranslation.x,
                                                      modelTranslation.y,
                                                      modelTranslation.z)
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        return drawable.views.map { view in
            let userViewpointMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = ProjectiveTransform3D(leftTangent: Double(view.tangents[0]),
                                                         rightTangent: Double(view.tangents[1]),
                                                         topTangent: Double(view.tangents[2]),
                                                         bottomTangent: Double(view.tangents[3]),
                                                         nearZ: Double(drawable.depthRange.y),
                                                         farZ: Double(drawable.depthRange.x),
                                                         reverseZ: true)
            let screenSize = SIMD2(x: Int(view.textureMap.viewport.width),
                                   y: Int(view.textureMap.viewport.height))
            return ModelRendererViewportDescriptor(viewport: view.textureMap.viewport,
                                                   projectionMatrix: .init(projectionMatrix),
                                                   viewMatrix: userViewpointMatrix * translationMatrix * rotationMatrix * commonUpCalibration,
                                                   screenSize: screenSize)
        }
    }

    private func updateRotation() {
        let now = Date()
        defer { lastRotationUpdateTimestamp = now }
        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }

        updateRotation()

        let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)

        do {
            try modelRenderer?.render(viewports: viewports,
                                      colorTexture: drawable.colorTextures[0],
                                      colorStoreAction: .store,
                                      depthTexture: drawable.depthTextures[0],
                                      rasterizationRateMap: drawable.rasterizationRateMaps.first,
                                      renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                                      to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        drawable.encodePresent(commandBuffer: commandBuffer)
        commandBuffer.commit()
        frame.endSubmission()
    }

    func renderLoop() {
        Task {
            do {
                for await update in handTracking.anchorUpdates {
                    switch update.event {
                    case .added, .updated:
                        if update.anchor.chirality == .right {
                            latestRightHandAnchor = update.anchor
                            let pinching = isPinching(update.anchor)
                            if pinching || isGrabbing {
                                updateGrabState(with: update.anchor, pinching: pinching)
                            }
                        } else {
                            latestLeftHandAnchor = update.anchor
                        }
                    case .removed:
                        if update.anchor.chirality == .right {
                            latestRightHandAnchor = nil
                            isGrabbing = false
                        } else {
                            latestLeftHandAnchor = nil
                        }
                    }
                }
            } catch {
                Self.log.error("Failed to receive hand anchor updates: \(error.localizedDescription)")
            }
        }

        while true {
            if layerRenderer.state == .invalidated {
                Self.log.warning("Layer is invalidated")
                return
            } else if layerRenderer.state == .paused {
                layerRenderer.waitUntilRunning()
                continue
            } else {
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
}

#endif // os(visionOS)
