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

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

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

    /// Update the model translation based on pinch state for the given hand
    func updateGrabState(with anchor: HandAnchor) {
        guard let thumbTip = anchor.handSkeleton?.joint(.thumbTip).parentFromJointTransform,
              let indexTip = anchor.handSkeleton?.joint(.indexFingerTip).parentFromJointTransform else {
            return
        }

        // Positions in hand-local space
        let thumbPos = SIMD3<Float>(thumbTip.columns.3.x,
                                    thumbTip.columns.3.y,
                                    thumbTip.columns.3.z)
        let indexPos = SIMD3<Float>(indexTip.columns.3.x,
                                    indexTip.columns.3.y,
                                    indexTip.columns.3.z)
        let pinchCenterLocal = (thumbPos + indexPos) / 2.0

        // Convert into world space (so model doesn’t "stick" to hand anchor space)
        let anchorTransform = anchor.originFromAnchorTransform
        let pinchCenterWorld4 = anchorTransform * SIMD4<Float>(pinchCenterLocal, 1)
        let pinchCenterWorld = SIMD3<Float>(pinchCenterWorld4.x,
                                            pinchCenterWorld4.y,
                                            pinchCenterWorld4.z)

        // Detect pinch
        let distance = simd_length(thumbPos - indexPos)
        let pinching = distance < 0.02 // ~2cm

        if pinching && !isGrabbing {
            // Just started pinching → "grab" the model
            isGrabbing = true
            grabOffset = modelTranslation - pinchCenterWorld
        } else if pinching && isGrabbing {
            // Continue dragging → update model position
            let target = pinchCenterWorld + grabOffset
            modelTranslation = simd_mix(modelTranslation, target, SIMD3<Float>(repeating: 0.2))
        } else if !pinching && isGrabbing {
            // Pinch released → drop the model
            isGrabbing = false
        }
    }


    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        handTracking = HandTrackingProvider()
        arSession = ARKitSession()
    }
    
    func isPinching(_ anchor: HandAnchor) -> Bool {
        guard let thumbTip = anchor.handSkeleton?.joint(.thumbTip).parentFromJointTransform,
              let indexTip = anchor.handSkeleton?.joint(.indexFingerTip).parentFromJointTransform else {
            return false
        }
        let thumbPos = SIMD3<Float>(thumbTip.columns.3.x,
                                    thumbTip.columns.3.y,
                                    thumbTip.columns.3.z)
        let indexPos = SIMD3<Float>(indexTip.columns.3.x,
                                    indexTip.columns.3.y,
                                    indexTip.columns.3.z)

        let distance = simd_length(thumbPos - indexPos)
        return distance < 0.02  // ~2 cm threshold
    }

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
        defer {
            lastRotationUpdateTimestamp = now
        }

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
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

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
                            updateGrabState(with: update.anchor)
                        } else {
                            latestLeftHandAnchor = update.anchor
                        }
                    case .removed:
                        if update.anchor.chirality == .right {
                            latestRightHandAnchor = nil
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





   

   
