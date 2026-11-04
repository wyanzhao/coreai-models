import Foundation
import Testing
@testable import CoreAILanguageModels

@Suite(.serialized)
struct FunctionalBatchIntegrationTests {
    @Test func realModelRaggedRowsAndPrefixRestore() async throws {
        guard let path = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_ASSET"],
              let casePath = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_CASE"] else { return }
        let asset = URL(fileURLWithPath: path)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casePath))) as! [String: Any]
        let tokens = (json["ids"] as! [Int]).map(Int32.init)
        let backend = try await CoreAIFunctionalBatchBackend(assetURL: asset)
        let session = try backend.makeBatchSessionState(batchSize: 2)
        let first = try await backend.prefillRow(row: 0, promptTokens: tokens, session: session)
        let short = Array(tokens.prefix(16))
        _ = try await backend.prefillRow(row: 1, promptTokens: short, session: session)
        let two = try await backend.decodeBatch(tokensPerRow: [[tokens[0]],[tokens[16]]],
            startPositions: [32,16], session: session)
        let single = try backend.makeBatchSessionState(batchSize: 1)
        let before = backend.reusedPrefixTokens
        let restored = try await backend.prefillRow(row: 0, promptTokens: tokens, session: single)
        #expect(backend.reusedPrefixTokens - before == 31)
        #expect(first == restored)
        let one = try await backend.decodeBatch(tokensPerRow: [[tokens[0]]], startPositions: [32], session: single)
        #expect((two[0] == one[0]) == true)
        let singleB = try backend.makeBatchSessionState(batchSize: 1)
        _ = try await backend.prefillRow(row: 0, promptTokens: short, session: singleB)
        let other = try await backend.decodeBatch(tokensPerRow: [[tokens[16]]], startPositions: [16], session: singleB)
        #expect((two[1] == other[0]) == true)
        #expect(two.allSatisfy { $0.allSatisfy { $0.isFinite } })
        var previous = two
        for step in 0..<4 {
            let ids = previous.map { row in Int32(row.indices.max(by: { row[$0] < row[$1] })!) }
            let batched = try await backend.decodeBatch(tokensPerRow: ids.map { [$0] },
                startPositions: [33 + step,17 + step], session: session)
            let a = try await backend.decodeBatch(tokensPerRow: [[ids[0]]], startPositions: [33 + step], session: single)
            let b = try await backend.decodeBatch(tokensPerRow: [[ids[1]]], startPositions: [17 + step], session: singleB)
            #expect(batched[0] == a[0]); #expect(batched[1] == b[0])
            previous = batched
        }
        // Independent full-width prefill reference, not the same incremental path.
        let referenceURL = URL(fileURLWithPath: casePath).deletingLastPathComponent().appendingPathComponent("logits.out.bin")
        let reference = try Data(contentsOf: referenceURL)
        let vocabulary = first.count
        let expected = reference.withUnsafeBytes { bytes in
            (0..<vocabulary).map { i in
                let raw = bytes.loadUnaligned(fromByteOffset: ((tokens.count - 1) * vocabulary + i) * 2, as: Float16.self)
                return Float16(30 * tanh(Float(raw) / 30))
            }
        }
        let noise = zip(first,expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1),2) }
        let signal = expected.reduce(0.0) { $0 + pow(Double($1),2) }
        #expect(noise / signal < 0.0001)

    }
}

extension FunctionalBatchIntegrationTests {
    @Test(.timeLimit(.minutes(10))) func ringWrapAndCrossRequestPrefixRestore() async throws {
        guard ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_LONG"] == "1",
              let path = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_ASSET"],
              let casePath = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casePath))) as! [String: Any]
        let seed = (json["ids"] as! [Int]).map(Int32.init)
        let long = Array(repeating: seed, count: 40).flatMap { $0 }
        let short = Array(long.prefix(1152))
        let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath: path), prefixCacheBytes: 128 * 1024 * 1024)
        let batch = try backend.makeBatchSessionState(batchSize: 2)
        _ = try await backend.prefillRow(row: 0, promptTokens: long, session: batch)
        backend.clearPrefixCache() // Force the later reuse to come from the shorter request.
        _ = try await backend.prefillRow(row: 1, promptTokens: short, session: batch)
        let referenceA = try backend.makeBatchSessionState(batchSize: 1)
        let before = backend.reusedPrefixTokens
        _ = try await backend.prefillRow(row: 0, promptTokens: long, session: referenceA)
        #expect(backend.reusedPrefixTokens > before)
        let referenceB = try backend.makeBatchSessionState(batchSize: 1)
        _ = try await backend.prefillRow(row: 0, promptTokens: short, session: referenceB)
        backend.clearPrefixCache()
        let coldReference = try backend.makeBatchSessionState(batchSize: 1)
        _ = try await backend.prefillRow(row: 0, promptTokens: long, session: coldReference)
        for step in 0..<3 {
            let ids: [[Int32]] = [[seed[step]],[seed[step + 4]]]
            let two = try await backend.decodeBatch(tokensPerRow: ids,
                startPositions: [long.count + step,short.count + step], session: batch)
            let a = try await backend.decodeBatch(tokensPerRow: [ids[0]], startPositions: [long.count + step], session: referenceA)
            let b = try await backend.decodeBatch(tokensPerRow: [ids[1]], startPositions: [short.count + step], session: referenceB)
            let cold = try await backend.decodeBatch(tokensPerRow: [ids[0]], startPositions: [long.count + step], session: coldReference)
            let coldIdentical = two[0] == cold[0]
            let secondIdentical = two[1] == b[0]
            #expect(coldIdentical); #expect(secondIdentical)
            // Restoring at 1151 changes the subsequent q32 chunk boundaries.
            // Check the pre-established FP16 SNR gate plus top1, and independently
            // require bitwise parity above when chunk boundaries are unchanged.
            let noise = zip(two[0],a[0]).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1),2) }
            let signal = two[0].reduce(0.0) { $0 + pow(Double($1),2) }
            let topA = two[0].indices.max(by: { two[0][$0] < two[0][$1] })
            let topB = a[0].indices.max(by: { a[0][$0] < a[0][$1] })
            print("ring-prefix step=\(step) snr=\(-10 * log10(max(noise / signal, 1e-30))) top1=\(topA == topB) cold_exact=\(coldIdentical)")
            #expect(noise / signal < 0.0001); #expect(topA == topB)
            #expect(two.allSatisfy { $0.allSatisfy { $0.isFinite } })
        }
    }
}
