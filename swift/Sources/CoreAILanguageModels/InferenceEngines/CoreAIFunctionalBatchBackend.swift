// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause
import CoreAI
import CoreAIShared
import Foundation
import Synchronization

/// Experimental functional-KV backend. Calls must be serialized by BatchScheduler.
/// This asset ABI is separate from the shipping static-shape engine.
final class CoreAIFunctionalBatchBackend: IncrementalPrefillStep, @unchecked Sendable {
    final class Session: BatchSessionState {
        var rows: [ReadonlyBatchKV]
        var deviceCaches: [String: NDArray] = [:]
        var boundCursors: [Int?]
        let deviceCacheBytes = Atomic<Int>(0)
        let hostStateBytes: Int
        init(rows: [ReadonlyBatchKV]) {
            self.rows = rows; boundCursors = Array(repeating: nil, count: rows.count)
            hostStateBytes = rows.reduce(0) { $0 + $1.byteCount }
        }
        var estimatedStateBytes: Int { hostStateBytes + deviceCacheBytes.load(ordering: .relaxed) }
        var estimatedLogitsScratchBytes: Int { 0 }
    }
    let maxBatchCapacity = 2
    let geometry: ReadonlyBatchKV.Geometry
    private let emptyState: ReadonlyBatchKV
    private let model: AIModel
    private let functions: [String: InferenceFunction]
    private let embeddingInputs: [String: NDArray]
    private let embedding: SafeTensorsReader
    private let embeddingScale: Float
    private let ple: PerLayerEmbeddings
    private let theta: [Double]
    private let hidden: Int
    private let vocab: Int
    private let softcap: Float
    private let prefixCache: ReadonlyPrefixCache
    private let counters = Mutex((hits: 0, misses: 0, tokens: 0, bytes: 0))
    var prefixMetrics: (hits: Int, misses: Int, tokens: Int, bytes: Int) { counters.withLock { $0 } }
    var reusedPrefixTokens: Int { prefixMetrics.tokens }

    init(assetURL: URL, prefixCacheBytes: Int = 256 * 1024 * 1024) async throws {
        guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf:
            assetURL.appendingPathComponent("batch-manifest.json"))) as? [String: Any],
            manifest["abi"] as? String == "gemma4-readonly-batch-v1",
            let cfg = manifest["text_config"] as? [String: Any],
            let context = manifest["context_capacity"] as? Int,
            let ring = manifest["sliding_ring"] as? Int,
            let window = manifest["sliding_window"] as? Int,
            let ng = manifest["global_storing_layers"] as? Int,
            let ns = manifest["sliding_storing_layers"] as? Int,
            let h = cfg["hidden_size"] as? Int,
            let v = cfg["vocab_size"] as? Int,
            let sh = cfg["head_dim"] as? Int,
            let gh = cfg["global_head_dim"] as? Int,
            let kv = cfg["num_key_value_heads"] as? Int,
            let parameters = cfg["rope_parameters"] as? [String: [String: Any]],
            let global = parameters["full_attention"], let sliding = parameters["sliding_attention"],
            let globalTheta = global["rope_theta"] as? Double,
            let slidingTheta = sliding["rope_theta"] as? Double,
            let rotary = global["partial_rotary_factor"] as? Double,
            let embeddingFile = manifest["embedding_file"] as? String,
            let pleFile = manifest["ple_file"] as? String else {
            throw InferenceRuntimeError.invalidState("Invalid functional Gemma asset manifest")
        }
        CLILogger.log("Functional manifest loaded: \(assetURL.path)")
        geometry = .init(context: context, ring: ring, window: window, globalLayers: ng,
            slidingLayers: ns, globalChannels: kv * gh, slidingChannels: kv * sh)
        emptyState = try ReadonlyBatchKV(geometry: geometry)
        hidden = h; vocab = v
        softcap = (cfg["final_logit_softcapping"] as? NSNumber)?.floatValue ?? 0
        theta = DualRoPEInputHandler.buildTheta(.init(slidingHeadDim: sh, globalHeadDim: gh,
            slidingRopeTheta: slidingTheta, globalRopeTheta: globalTheta, partialRotaryFactor: rotary))
        prefixCache = ReadonlyPrefixCache(byteLimit: prefixCacheBytes)
        embedding = try SafeTensorsReader(url: assetURL.appendingPathComponent(embeddingFile), singleTensor: "embedding_table")
        let scale = try SafeTensorsReader(url: assetURL.appendingPathComponent(embeddingFile), singleTensor: "scale")
        guard embedding.dtype == "I8", embedding.shape == [v,1,h], embedding.byteCount == v * h,
              scale.dtype == "F16", scale.byteCount == 2 else {
            throw InferenceRuntimeError.invalidState("Unsupported functional embedding encoding")
        }
        embeddingScale = scale.data.withUnsafeBytes { Float($0.loadUnaligned(fromByteOffset: scale.dataStart, as: Float16.self)) }
        ple = try PerLayerEmbeddings(contentsOf: assetURL.appendingPathComponent(pleFile))
        CLILogger.log("Functional sidecars loaded; loading AIModel")
        let runtime = try await Self.loadRuntime(at: assetURL.appendingPathComponent("model.aimodel"), context: context)
        model = runtime.model
        CLILogger.log("Functional AIModel loaded")
        let embeddingReader = embedding
        let packedURL = assetURL.appendingPathComponent("embedding_interleaved8.bin")
        let packed = FileManager.default.fileExists(atPath: packedURL.path) ? try Data(contentsOf: packedURL, options: .alwaysMapped) : nil
        var loaded: [String: InferenceFunction] = [:]
        var tables: [String: NDArray] = [:]
        var sharedDescriptor: NDArrayDescriptor?
        var sharedTable: NDArray?
        for (b,q) in [(1,1),(1,32),(2,1)] {
            let key = "\(b):\(q)"
            CLILogger.log("Loading functional entrypoint \(key)")
            guard let fn = runtime.functions[key],
                  case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: "embedding_table") else {
                throw InferenceRuntimeError.invalidState("Missing functional batch entrypoint \(key)")
            }
            loaded[key] = fn
            if sharedDescriptor == desc, let sharedTable {
                tables[key] = sharedTable
            } else {
                var table: NDArray
                if let packed, packed.count == embeddingReader.byteCount,
                   desc.minimumByteCount == packed.count,
                   desc.interleaveLayout?.dimension == 0, desc.interleaveLayout?.factor == 8 {
                    table = NDArray(descriptor: desc)
                    table.mutableRawView().withUnsafeMutableBytes { destination, _, _ in
                        packed.withUnsafeBytes { source in
                            _ = memcpy(destination, source.baseAddress!, packed.count)
                        }
                    }
                } else {
                    table = try embeddingReader.data.withUnsafeBytes { raw in
                        try FunctionalTensorIO.make(UnsafeRawBufferPointer(rebasing: raw[embeddingReader.dataStart..<embeddingReader.dataStart + embeddingReader.byteCount]),
                            as: Int8.self, descriptor: desc)
                    }
                }
                tables[key] = table
                sharedTable = table; sharedDescriptor = desc
            }
        }
        functions = loaded; embeddingInputs = tables
        CLILogger.log("Functional entrypoints ready")
    }

    /// A cached specialization can load its model yet fail to materialize a declared
    /// function with CoreAI's nilError. Rebuild this experimental asset's cache once.
    private static func loadRuntime(at url: URL, context: Int) async throws
        -> (model: AIModel, functions: [String: InferenceFunction]) {
        for attempt in 0..<2 {
            let model = try await AIModel(contentsOf: url,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            let names = [(1,1),(1,32),(2,1)].map { "decode_b\($0.0)_q\($0.1)_c\(context)" }
            guard names.allSatisfy({ model.functionNames.contains($0) }) else {
                throw InferenceRuntimeError.invalidState("Functional asset is missing a required entrypoint")
            }
            do {
                var functions: [String: InferenceFunction] = [:]
                for (b,q) in [(1,1),(1,32),(2,1)] {
                    guard let fn = try model.loadFunction(named: "decode_b\(b)_q\(q)_c\(context)") else {
                        throw InferenceRuntimeError.invalidState("Declared functional entrypoint unavailable")
                    }
                    functions["\(b):\(q)"] = fn
                }
                return (model, functions)
            } catch {
                guard attempt == 0, String(describing: error) == "nilError" else { throw error }
                CLILogger.log("Functional cached entrypoint failed (\(error)); rebuilding this asset cache once")
                try AIModelCache.default.deleteEntries(for: url)
            }
        }
        throw InferenceRuntimeError.invalidState("Functional runtime retry exhausted")
    }

    func clearPrefixCache() { prefixCache.removeAll(); counters.withLock { $0.bytes = 0 } }

    func makeBatchSessionState(batchSize: Int) throws -> any BatchSessionState {
        guard (1...2).contains(batchSize) else { throw InferenceRuntimeError.invalidState("Batch capacity is 2") }
        return Session(rows: (0..<batchSize).map { _ in emptyState })
    }

    private func cast(_ state: any BatchSessionState) throws -> Session {
        guard let state = state as? Session else { throw InferenceRuntimeError.invalidState("Foreign functional batch session") }
        return state
    }

    func beginPrefill(row: Int, promptTokens: [Int32], session: any BatchSessionState) throws {
        let session = try cast(session)
        guard session.rows.indices.contains(row), !promptTokens.isEmpty, promptTokens.count <= geometry.context else {
            throw InferenceRuntimeError.invalidState("Invalid functional prefill extent")
        }
        let state = prefixCache.lookup(prompt: promptTokens, geometry: geometry) ?? emptyState
        counters.withLock {
            $0.tokens += state.cursor
            if state.cursor > 0 { $0.hits += 1 } else { $0.misses += 1 }
        }
        session.rows[row] = state
        session.boundCursors[row] = nil
    }

    func prefillChunk(row: Int, promptTokens: [Int32], session: any BatchSessionState) async throws -> [LogitsScalarType]? {
        let session = try cast(session)
        var state = session.rows[row]
        let remaining = promptTokens.count - state.cursor
        guard remaining > 0 else { throw InferenceRuntimeError.invalidState("Prefill already complete") }
        let valid = remaining == 1 ? 1 : min(32, remaining - 1)
        let q = valid == 1 ? 1 : 32
        var tokens = Array(promptTokens[state.cursor..<state.cursor + valid])
        tokens += Array(repeating: 0, count: q - valid)
        // The 16K B1/q1 specialization failed numerical parity on this runtime.
        // Use the validated B2/q1 graph with an inert second row for tail tokens.
        let physicalBatch = geometry.context >= 16384 && q == 1 ? 2 : 1
        let result = try await forward(tokens: physicalBatch == 2 ? [tokens,[0]] : [tokens],
            states: physicalBatch == 2 ? [state,emptyState] : [state])
        try commit(result, row: 0, batch: physicalBatch, query: q, valid: valid, state: &state)
        session.rows[row] = state
        session.boundCursors[row] = nil
        if state.cursor < promptTokens.count {
            prefixCache.insert(tokens: Array(promptTokens.prefix(state.cursor)), state: state)
            counters.withLock { $0.bytes = prefixCache.residentBytes }
            return nil
        }
        return capped(Array(result["logits"]![(valid - 1) * vocab..<valid * vocab]))
    }

    func prefillRow(row: Int, promptTokens: [Int32], session: any BatchSessionState) async throws -> [LogitsScalarType] {
        try beginPrefill(row: row, promptTokens: promptTokens, session: session)
        while true {
            if let logits = try await prefillChunk(row: row, promptTokens: promptTokens, session: session) { return logits }
        }
    }

    func decodeBatch(tokensPerRow: [[Int32]], startPositions: [Int], session: any BatchSessionState) async throws -> [[LogitsScalarType]] {
        let session = try cast(session)
        guard tokensPerRow.count == session.rows.count, startPositions.count == session.rows.count,
              tokensPerRow.allSatisfy({ $0.count == 1 }) else {
            throw InferenceRuntimeError.invalidState("Invalid functional decode batch")
        }
        var states = session.rows
        for i in states.indices {
            if startPositions[i] == 0 { states[i] = emptyState }
            else if startPositions[i] != states[i].cursor || states[i].cursor >= geometry.context {
                throw InferenceRuntimeError.invalidState("Functional row cursor mismatch or context exhausted")
            }
        }
        let logicalBatch = states.count
        var inputTokens = tokensPerRow
        var active = startPositions.map { $0 > 0 }
        if geometry.context >= 16384 && logicalBatch == 1 {
            states.append(emptyState); inputTokens.append([0]); active.append(false)
        }
        try bindDeviceRows(session, states: states, active: active)
        let batch = states.count
        let result = try await forward(tokens: inputTokens, states: states, deviceCaches: session.deviceCaches)
        // Release the read-only copies before mutating canonical rows, avoiding a
        // full host KV copy on every decode (snapshots still copy on first write).
        states.removeAll()
        for i in 0..<logicalBatch where startPositions[i] > 0 {
            try commit(result, row: i, batch: batch, query: 1, valid: 1, state: &session.rows[i])
            writeDeviceUpdates(session, outputs: result, row: i, position: startPositions[i], physicalBatch: batch)
            session.boundCursors[i] = session.rows[i].cursor
        }
        return (0..<logicalBatch).map { capped(Array(result["logits"]![$0 * vocab..<($0 + 1) * vocab])) }
    }

    func allPositionLogits(tokensPerRow: [[Int32]], startPosition: Int, session: any BatchSessionState) async throws -> [LogitsScalarType] {
        throw InferenceRuntimeError.invalidState("Functional batch scoring is not enabled")
    }

    private func capped(_ logits: [Float16]) -> [LogitsScalarType] {
        guard softcap > 0 else { return logits }
        return logits.map { Float16(softcap * tanh(Float($0) / softcap)) }
    }

    private func commit(_ outputs: [String: [Float16]], row: Int, batch: Int, query: Int, valid: Int, state: inout ReadonlyBatchKV) throws {
        func extract(_ name: String, layers: Int, channels: Int) -> [Float16] {
            let all = outputs[name]!
            return (0..<layers).flatMap { layer in
                let start = (layer * batch + row) * channels * query
                return Array(all[start..<start + channels * query])
            }
        }
        try state.commit(globalK: extract("new_global_k", layers: geometry.globalLayers, channels: geometry.globalChannels),
            globalV: extract("new_global_v", layers: geometry.globalLayers, channels: geometry.globalChannels),
            slidingK: extract("new_sliding_k", layers: geometry.slidingLayers, channels: geometry.slidingChannels),
            slidingV: extract("new_sliding_v", layers: geometry.slidingLayers, channels: geometry.slidingChannels),
            queryWidth: query, validTokens: valid)
    }

    private var cacheLayouts: [(String, KeyPath<ReadonlyBatchKV, [Float16]>, Int, Int, Int)] {
        [("key_cache", \ReadonlyBatchKV.globalK, geometry.globalLayers, geometry.globalChannels, geometry.context),
         ("value_cache", \ReadonlyBatchKV.globalV, geometry.globalLayers, geometry.globalChannels, geometry.context),
         ("sliding_key_cache", \ReadonlyBatchKV.slidingK, geometry.slidingLayers, geometry.slidingChannels, geometry.ring),
         ("sliding_value_cache", \ReadonlyBatchKV.slidingV, geometry.slidingLayers, geometry.slidingChannels, geometry.ring)]
    }

    /// Rebind complete rows only at admission/restoration. Ordinary decode updates
    /// just the new K/V position in these persistent runtime-layout buffers.
    private func bindDeviceRows(_ session: Session, states: [ReadonlyBatchKV], active: [Bool]) throws {
        let batch = states.count
        let fn = functions["\(batch):1"]!
        for (name,path,layers,channels,capacity) in cacheLayouts {
            if session.deviceCaches[name] == nil {
                guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) else {
                    throw InferenceRuntimeError.invalidState("Missing cache descriptor")
                }
                var array = NDArray(descriptor: desc)
                _ = array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in memset(ptr, 0, desc.minimumByteCount) }
                session.deviceCaches[name] = array
                _ = session.deviceCacheBytes.wrappingAdd(desc.minimumByteCount, ordering: .relaxed)
            }
            for row in states.indices where active[row] && session.boundCursors[row] != states[row].cursor {
                let source = states[row][keyPath: path]
                let factor = session.deviceCaches[name]!.interleaveLayout?.factor ?? 1
                let dim = session.deviceCaches[name]!.interleaveLayout?.dimension ?? 0
                session.deviceCaches[name]!.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, strides in
                    for layer in 0..<layers { for channel in 0..<channels {
                        let outer = layer * batch + row
                        let base = (dim == 0 ? outer / factor : outer) * strides[0]
                            + (dim == 1 ? channel / factor : channel) * strides[1]
                            + (dim == 0 ? outer % factor : (dim == 1 ? channel % factor : 0))
                        for position in 0..<capacity {
                            let offset = base + (dim == 3 ? position / factor : position) * strides[3]
                                + (dim == 3 ? position % factor : 0)
                            ptr[offset] = source[(layer * channels + channel) * capacity + position]
                        }
                    } }
                }
            }
        }
        for row in states.indices where active[row] { session.boundCursors[row] = states[row].cursor }
    }

    private func writeDeviceUpdates(_ session: Session, outputs: [String: [Float16]], row: Int, position: Int, physicalBatch batch: Int) {
        let outputNames = ["new_global_k","new_global_v","new_sliding_k","new_sliding_v"]
        for (index,layout) in cacheLayouts.enumerated() {
            let (name,_,layers,channels,capacity) = layout
            let slot = index < 2 ? position : position % capacity
            let values = outputs[outputNames[index]]!
            let factor = session.deviceCaches[name]!.interleaveLayout?.factor ?? 1
            let dim = session.deviceCaches[name]!.interleaveLayout?.dimension ?? 0
            session.deviceCaches[name]!.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, strides in
                for layer in 0..<layers { for channel in 0..<channels {
                    let outer = layer * batch + row
                    let offset = (dim == 0 ? outer / factor : outer) * strides[0]
                        + (dim == 1 ? channel / factor : channel) * strides[1]
                        + (dim == 3 ? slot / factor : slot) * strides[3]
                        + (dim == 0 ? outer % factor : (dim == 1 ? channel % factor : (dim == 3 ? slot % factor : 0)))
                    ptr[offset] = values[(layer * batch + row) * channels + channel]
                } }
            }
        }
    }

    private func forward(tokens: [[Int32]], states: [ReadonlyBatchKV], deviceCaches: [String: NDArray]? = nil) async throws -> [String: [Float16]] {
        let b = tokens.count; let q = tokens[0].count; let key = "\(b):\(q)"
        guard let fn = functions[key], tokens.flatMap({$0}).allSatisfy({ $0 >= 0 && $0 < vocab }) else {
            throw InferenceRuntimeError.invalidState("Unsupported functional shape or token")
        }
        var inputs: [String: NDArray] = ["embedding_table": embeddingInputs[key]!]
        func add<T: BitwiseCopyable>(_ name: String, _ values: [T]) throws {
            guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidState("Missing input \(name)")
            }
            inputs[name] = try FunctionalTensorIO.make(values, descriptor: desc)
        }
        let ids = tokens.flatMap { $0 }
        var embeddings = [Float16](repeating: 0, count: ids.count * hidden)
        embedding.data.withUnsafeBytes { raw in
            for (i,id) in ids.enumerated() {
                for channel in 0..<hidden {
                    let value = raw.load(fromByteOffset: embedding.dataStart + Int(id) * hidden + channel, as: Int8.self)
                    embeddings[i * hidden + channel] = Float16(Float(value) * embeddingScale)
                }
            }
        }
        try add("transformer_input", embeddings)
        var perLayer = [Int8](repeating: 0, count: ids.count * ple.rowWidth)
        perLayer.withUnsafeMutableBufferPointer { ple.gather(tokenIDs: ids, batchSize: ids.count, into: $0) }
        try add("ple_embeddings", perLayer)
        var cosines: [Float16] = []; var sines: [Float16] = []
        for state in states { for j in 0..<q { for frequency in theta {
            let phase = Double(state.cursor + j) * frequency
            cosines.append(Float16(cos(phase))); sines.append(Float16(sin(phase)))
        } } }
        try add("rope_cos", cosines); try add("rope_sin", sines)
        try add("causal_mask", states.flatMap { $0.mask(queryWidth: q, sliding: false) })
        try add("sliding_causal_mask", states.flatMap { $0.mask(queryWidth: q, sliding: true) })
        for (name,path,layers,channels,capacity) in [
            ("key_cache", \ReadonlyBatchKV.globalK, geometry.globalLayers, geometry.globalChannels, geometry.context),
            ("value_cache", \ReadonlyBatchKV.globalV, geometry.globalLayers, geometry.globalChannels, geometry.context),
            ("sliding_key_cache", \ReadonlyBatchKV.slidingK, geometry.slidingLayers, geometry.slidingChannels, geometry.ring),
            ("sliding_value_cache", \ReadonlyBatchKV.slidingV, geometry.slidingLayers, geometry.slidingChannels, geometry.ring)] {
            if let cached = deviceCaches?[name] { inputs[name] = cached; continue }
            var flat: [Float16] = []
            for layer in 0..<layers { for state in states {
                let start = layer * channels * capacity
                flat.append(contentsOf: state[keyPath: path][start..<start + channels * capacity])
            } }
            try add(name, flat)
        }
        var output = try await fn.run(inputs: inputs)
        var result: [String: [Float16]] = [:]
        for name in ["logits","new_global_k","new_global_v","new_sliding_k","new_sliding_v"] {
            guard let array = output.remove(name)?.ndArray else { throw InferenceRuntimeError.invalidState("Missing output \(name)") }
            let values = FunctionalTensorIO.read(array)
            guard values.allSatisfy({ $0.isFinite }) else { throw InferenceRuntimeError.invalidState("Nonfinite functional output") }
            result[name] = values
        }
        return result
    }
}
