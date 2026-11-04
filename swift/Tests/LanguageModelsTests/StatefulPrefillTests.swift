import Foundation
import Testing
@testable import CoreAILanguageModels

@Suite(.serialized)
struct StatefulPrefillTests {
    @Test(.timeLimit(.minutes(15))) func fullModelStateParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let base = env["COREAI_FUNCTIONAL_ASSET"], let stateAsset = env["COREAI_STATEFUL_ASSET"],
              let fixture = env["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: fixture))) as! [String: Any]
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        let tokens = Array(repeating: seed, count: 600).flatMap { $0 }
        var reference: [[Float16]] = []
        for stateful in [false, true] {
            let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath: base),
                prefixCacheBytes: 128 * 1024 * 1024,
                statefulPrefillURL: stateful ? URL(fileURLWithPath: stateAsset) : nil)
            var values: [[Float16]] = []
            for length in [1,32,63,64,65,96,127,128,129,575,576,577,600,1089,4097,8193,16384] {
                backend.clearPrefixCache()
                let session = try backend.makeBatchSessionState(batchSize: 4)
                values.append(try await backend.prefillRow(row: 0,
                    promptTokens: Array(tokens.prefix(length)), session: session))
                print("stateful-prefill mode=\(stateful) cold=\(length)")
            }
            for prefix in [65,96,1002,1152] {
                backend.clearPrefixCache()
                let session = try backend.makeBatchSessionState(batchSize: 4)
                _ = try await backend.prefillRow(row: 0,
                    promptTokens: Array(tokens.prefix(prefix)), session: session)
                let before = backend.reusedPrefixTokens
                values.append(try await backend.prefillRow(row: 1,
                    promptTokens: Array(tokens.prefix(prefix + 130)), session: session))
                #expect(backend.reusedPrefixTokens > before)
                print("stateful-prefill mode=\(stateful) restore=\(prefix-1)")
            }
            backend.clearPrefixCache()
            let batch = try backend.makeBatchSessionState(batchSize: 4)
            let lengths = [65,600,1089,2049]
            for row in 0..<4 {
                values.append(try await backend.prefillRow(row: row,
                    promptTokens: Array(tokens.prefix(lengths[row])), session: batch))
            }
            values += try await backend.decodeBatch(tokensPerRow: [[1],[2],[3],[4]],
                startPositions: lengths, session: batch)
            // Abandon a live state bank, then reuse that row for unrelated tokens.
            backend.clearPrefixCache()
            try backend.beginPrefill(row: 0, promptTokens: Array(tokens.prefix(1024)), session: batch)
            _ = try await backend.prefillChunk(row: 0, promptTokens: Array(tokens.prefix(1024)), session: batch)
            backend.endPrefill(row: 0, session: batch)
            let concrete = batch as! CoreAIFunctionalBatchBackend.Session
            #expect(concrete.prefillBanks.isEmpty)
            values.append(try await backend.prefillRow(row: 0,
                promptTokens: Array(tokens.dropFirst(7).prefix(129)), session: batch))
            if stateful {
                #expect(values.count == reference.count)
                for (index, actual) in values.enumerated() {
                    let expected = reference[index]
                    let signal = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
                    let noise = zip(actual,expected).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
                    let ratio = noise / max(signal, 1e-30)
                    let top = actual.indices.max { actual[$0] < actual[$1] }
                    let refTop = expected.indices.max { expected[$0] < expected[$1] }
                    let finite = actual.allSatisfy { $0.isFinite }
                    let exact = actual == expected
                    print("stateful-parity case=\(index) snr=\(-10 * log10(max(ratio,1e-30))) top1=\(top == refTop)")
                    #expect(finite); #expect(ratio < 0.0001); #expect(top == refTop); #expect(exact)
                }
                print("stateful-engagement \(backend.statefulPrefillMetrics)")
                #expect(backend.statefulPrefillMetrics.chunks > 0)
                #expect(backend.statefulPrefillMetrics.tokens == backend.statefulPrefillMetrics.chunks * 64)
            } else { reference = values }
        }
    }
}
