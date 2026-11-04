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
        let tokens = (try #require(json["ids"] as? [Int])).map(Int32.init)
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
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        try #require(seed.count >= 32)
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

extension FunctionalBatchIntegrationTests {
    @Test(.timeLimit(.minutes(10))) func fourRaggedRowsMatchIndependentDecode() async throws {
        guard ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_BATCH4"] == "1",
              let path = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_ASSET"],
              let casePath = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casePath))) as! [String: Any]
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        try #require(seed.count >= 32)
        let repeated = Array(repeating: seed, count: 24).flatMap { $0 }
        let lengths = [32, 65, 577, 640]
        let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath: path), prefixCacheBytes: 0)
        #expect(backend.maxBatchCapacity >= 4)
        let batch = try backend.makeBatchSessionState(batchSize: 4)
        var references: [any BatchSessionState] = []
        for row in 0..<4 {
            let prompt = Array(repeated.prefix(lengths[row]))
            _ = try await backend.prefillRow(row: row, promptTokens: prompt, session: batch)
            let reference = try backend.makeBatchSessionState(batchSize: 1)
            _ = try await backend.prefillRow(row: 0, promptTokens: prompt, session: reference)
            references.append(reference)
        }
        var cursors = lengths
        let phases = [[0,1,2,3], [0,2], [3], [0,1,2,3], [1,3], [0,1,2,3]]
        for (step,live) in phases.enumerated() {
            let ids = (0..<4).map { [seed[step + $0]] }
            let positions = (0..<4).map { live.contains($0) ? cursors[$0] : 0 }
            let output = try await backend.decodeBatch(tokensPerRow: ids,
                startPositions: positions, session: batch)
            for row in 0..<4 {
                if !live.contains(row) { #expect(output[row].isEmpty); continue }
                let reference = try await backend.decodeBatch(tokensPerRow: [ids[row]],
                    startPositions: [cursors[row]], session: references[row])[0]
                cursors[row] += 1
                let values = output[row]
                let signal = reference.reduce(0.0) { $0 + pow(Double($1),2) }
                let noise = zip(values,reference).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
                let top1 = values.indices.max(by: { values[$0] < values[$1] })
                let expectedTop1 = reference.indices.max(by: { reference[$0] < reference[$1] })
                print("b4-compaction row=\(row) step=\(step) snr=\(-10 * log10(max(noise / signal, 1e-30))) top1=\(top1 == expectedTop1)")
                let finite = values.allSatisfy { $0.isFinite }
                #expect(finite)
                #expect(noise / signal < 0.0001)
                #expect(top1 == expectedTop1)
            }
        }
    }
}

extension FunctionalBatchIntegrationTests {
    @Test(.timeLimit(.minutes(10))) func widerPrefillMatchesOriginalChunking() async throws {
        guard ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_BATCH4"] == "1",
              let path = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_ASSET"],
              let casePath = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casePath))) as! [String: Any]
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        try #require(seed.count >= 32)
        let tokens = Array(repeating: seed, count: 20).flatMap { $0 }
        var outputs: [[[Float16]]] = []
        for width in [32,64] {
            let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath: path),
                prefixCacheBytes: 0, prefillQueryLimit: width)
            let session = try backend.makeBatchSessionState(batchSize: 1)
            var values = [try await backend.prefillRow(row: 0, promptTokens: tokens, session: session)]
            for step in 0..<3 {
                values.append(try await backend.decodeBatch(tokensPerRow: [[seed[step]]],
                    startPositions: [tokens.count + step], session: session)[0])
            }
            outputs.append(values)
        }
        for step in 0..<4 {
            let a = outputs[0][step], b = outputs[1][step]
            let signal = a.reduce(0.0) { $0 + pow(Double($1),2) }
            let noise = zip(a,b).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
            let sameTop = a.indices.max(by: { a[$0] < a[$1] }) == b.indices.max(by: { b[$0] < b[$1] })
            print("q64-parity step=\(step) snr=\(-10 * log10(max(noise / signal, 1e-30))) top1=\(sameTop)")
            #expect(noise / signal < 0.0001)
            #expect(sameTop)
        }
    }
}

extension FunctionalBatchIntegrationTests {
    @Test(.timeLimit(.minutes(10))) func eightRaggedRowsMatchIndependentDecode() async throws {
        guard ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_BATCH8"] == "1",
              let path = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_ASSET"],
              let casePath = ProcessInfo.processInfo.environment["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: casePath))) as! [String: Any]
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        try #require(seed.count >= 32)
        let repeated = Array(repeating: seed, count: 24).flatMap { $0 }
        let lengths = [32, 65, 577, 640, 33, 129, 513, 600]
        let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath: path), prefixCacheBytes: 0)
        #expect(backend.maxBatchCapacity >= 8)
        let batch = try backend.makeBatchSessionState(batchSize: 8)
        let phases = [[0,1,2,3,4,5,6,7], [1,3,5,7], [2,6], [7], [0,1,2,3,4,5,6,7]]
        var expected: [String: [Float16]] = [:]
        for row in 0..<8 {
            let prompt = Array(repeated.prefix(lengths[row]))
            _ = try await backend.prefillRow(row: row, promptTokens: prompt, session: batch)
            let reference = try backend.makeBatchSessionState(batchSize: 1)
            _ = try await backend.prefillRow(row: 0, promptTokens: prompt, session: reference)
            var cursor = lengths[row]
            for (step, live) in phases.enumerated() where live.contains(row) {
                expected["\(row):\(step)"] = try await backend.decodeBatch(
                    tokensPerRow: [[seed[step + row]]], startPositions: [cursor], session: reference)[0]
                cursor += 1
            }
        }
        var cursors = lengths
        for (step,live) in phases.enumerated() {
            let ids = (0..<8).map { [seed[step + $0]] }
            let positions = (0..<8).map { live.contains($0) ? cursors[$0] : 0 }
            let output = try await backend.decodeBatch(tokensPerRow: ids,
                startPositions: positions, session: batch)
            for row in 0..<8 {
                if !live.contains(row) { #expect(output[row].isEmpty); continue }
                let reference = expected["\(row):\(step)"]!
                cursors[row] += 1
                let values = output[row]
                let signal = reference.reduce(0.0) { $0 + pow(Double($1),2) }
                let noise = zip(values,reference).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
                let top1 = values.indices.max(by: { values[$0] < values[$1] })
                let expectedTop1 = reference.indices.max(by: { reference[$0] < reference[$1] })
                print("b8-compaction row=\(row) step=\(step) snr=\(-10 * log10(max(noise / signal, 1e-30))) top1=\(top1 == expectedTop1)")
                let finite = values.allSatisfy { $0.isFinite }
                #expect(finite)
                #expect(noise / signal < 0.0001)
                #expect(top1 == expectedTop1)
            }
        }
    }
}

extension FunctionalBatchIntegrationTests {
    @Test(.timeLimit(.minutes(15))) func prefillOptimizationAblations() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let baseline = env["COREAI_PREFILL_BASELINE"],
              let candidate = env["COREAI_FUNCTIONAL_ASSET"],
              let fixture = env["COREAI_FUNCTIONAL_CASE"] else { return }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: fixture))) as! [String: Any]
        let seed = (try #require(json["ids"] as? [Int])).map(Int32.init)
        let all = Array(repeating: seed, count: 40).flatMap { $0 }
        var reference: [[Float16]] = []
        var headless: [[Float16]] = []
        for (variant,path,persistent) in [("baseline",baseline,false),("headless",candidate,false),("persistent",candidate,true)] {
            let backend = try await CoreAIFunctionalBatchBackend(assetURL: URL(fileURLWithPath:path),
                prefixCacheBytes: 128 * 1024 * 1024, prefillQueryLimit:64, persistentPrefill:persistent)
            var outputs: [[Float16]] = []
            let session = try backend.makeBatchSessionState(batchSize:2)
            for length in [32,65,577,600,1089] {
                backend.clearPrefixCache()
                let tokens = Array(all.prefix(length))
                outputs.append(try await backend.prefillRow(row:0,promptTokens:tokens,session:session))
                for step in 0..<3 {
                    outputs.append(try await backend.decodeBatch(tokensPerRow:[[seed[step]],[0]],
                        startPositions:[length+step,0],session:session)[0])
                }
                // Reuse a row and restore an independently owned cached prefix.
                outputs.append(try await backend.prefillRow(row:1,promptTokens:tokens,session:session))
            }
            if variant == "baseline" { reference = outputs }
            else {
                for (i,values) in outputs.enumerated() {
                    let expected = reference[i]
                    let signal = expected.reduce(0.0) { $0 + pow(Double($1),2) }
                    let noise = zip(values,expected).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
                    let top = values.indices.max(by:{values[$0] < values[$1]})
                    let refTop = expected.indices.max(by:{expected[$0] < expected[$1]})
                    print("prefill-ablation \(variant) case=\(i) snr=\(-10 * log10(max(noise/signal,1e-30))) top1=\(top == refTop)")
                    #expect(values.allSatisfy { $0.isFinite })
                    #expect(noise/signal < 0.0001)
                    #expect(top == refTop)
                    if persistent { let identical = values == headless[i]; #expect(identical) }
                }
                if !persistent { headless = outputs }
            }
            print("prefill-io \(variant) \(backend.prefillIOMetrics)")
            if persistent { #expect(backend.prefillIOMetrics.legacyKVBytes == 0) }
        }
    }
}
