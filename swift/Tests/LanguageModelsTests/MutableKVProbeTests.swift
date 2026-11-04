import CoreAI
import Foundation
import Testing
@testable import CoreAILanguageModels

struct MutableKVProbeTests {
    @Test func explicitStatePersists() async throws {
        guard let path = ProcessInfo.processInfo.environment["COREAI_STATE_PROBE"] else { return }
        let model = try await AIModel(contentsOf: URL(fileURLWithPath: path),
            options: SpecializationOptions(preferredComputeUnitKind: ProcessInfo.processInfo.environment["COREAI_STATE_DEVICE"] == "gpu" ? .gpu : .neuralEngine))
        let function = try #require(try model.loadFunction(named: "main"))
        guard case .ndArray(let descriptor) = function.descriptor.stateDescriptor(of: "cache") else {
            Issue.record("missing state"); return
        }
        var cache = NDArray(descriptor: descriptor)
        _ = cache.mutableRawView().withUnsafeMutableBytes { ptr,_,_ in memset(ptr,0,descriptor.minimumByteCount) }
        guard case .ndArray(let updateDescriptor) = function.descriptor.inputDescriptor(of: "updates") else {
            Issue.record("missing updates"); return
        }
        guard updateDescriptor.shape[0] == 1 else {
            Issue.record("This descriptor-backed probe currently supports one row"); return
        }
        let channels = updateDescriptor.shape[1], query = updateDescriptor.shape[3]
        let capacity = descriptor.shape.last!
        var reference = [Float16](repeating:0,count:descriptor.shape.reduce(1,*))
        let positions = ProcessInfo.processInfo.environment["COREAI_STATE_POSITIONS"]?.split(separator:",").compactMap { Int($0) } ?? [0,1,2,63]
        for (step,position) in positions.enumerated() {
            guard case .ndArray(let updateDesc) = function.descriptor.inputDescriptor(of: "updates"),
                  case .ndArray(let positionDesc) = function.descriptor.inputDescriptor(of: "positions") else {
                Issue.record("missing inputs"); return
            }
            guard position >= 0, position + query <= capacity else { Issue.record("invalid probe position"); return }
            let updateValues = (0..<(channels*query)).map { Float16((step+1)*128 + $0 % 113) }
            let inputs = ["updates":try FunctionalTensorIO.make(updateValues,descriptor:updateDesc),
                          "positions":try FunctionalTensorIO.make([Int32(position)],descriptor:positionDesc)]
            var views = InferenceFunction.MutableViews()
            views.insert(&cache, for:"cache")
            var outputs = try await function.run(inputs:inputs,states:_unsafeEscapeMutableViews(consume views))
            guard let observed = outputs.remove("observed")?.ndArray else {
                Issue.record("missing observed output"); return
            }
            for c in 0..<channels { for j in 0..<query { reference[c*capacity+position+j] = updateValues[c*query+j] } }
            let outputExact = FunctionalTensorIO.read(observed) == reference
            let stateExact = FunctionalTensorIO.read(cache) == reference
            print("mutable-swift step=\(step) output=\(outputExact) state=\(stateExact)")
            #expect(outputExact); #expect(stateExact)
        }
    }
}
