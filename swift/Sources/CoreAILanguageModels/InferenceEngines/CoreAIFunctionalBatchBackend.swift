// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause
import CoreAI
import CoreAIShared
import Foundation
import Synchronization

/// Experimental functional-KV backend. Calls must be serialized by BatchScheduler.
/// This asset ABI is separate from the shipping static-shape engine.
final class CoreAIFunctionalBatchBackend: IncrementalPrefillStep, @unchecked Sendable {
    struct StepOutput {
        var logits: [Float16]?
        var native: [String: NDArray] = [:]
        var dense: [String: [Float16]] = [:]
        subscript(_ name: String) -> [Float16]? { name == "logits" ? logits : dense[name] }
        @inline(__always)
        func withKV<R>(_ name: String, _ body: ((Int) -> Float16) -> R) -> R {
            if let array = native[name] { return FunctionalTensorIO.withReader(array, body) }
            let values = dense[name]!
            return body { values[$0] }
        }
    }
    private let nativeKV: Bool
    final class PrefillBank {
        var buffers: [String: NDArray] = [:]
        var descriptors: [String: NDArrayDescriptor] = [:]
        var cursor = 0
        var bytes = 0
    }
    final class Session: BatchSessionState {
        var rows: [ReadonlyBatchKV]
        var deviceCaches: [String: NDArray] = [:]
        var boundCursors: [Int?]
        var deviceRows: [Int] = []
        var deviceBatch = 0
        var prefillBanks: [Int: PrefillBank] = [:]
        let prefillCacheBytes = Atomic<Int>(0)
        let deviceCacheBytes = Atomic<Int>(0)
        let hostStateBytes: Int
        init(rows: [ReadonlyBatchKV]) {
            self.rows = rows; boundCursors = Array(repeating: nil, count: rows.count)
            hostStateBytes = rows.reduce(0) { $0 + $1.byteCount }
        }
        var estimatedStateBytes: Int { hostStateBytes + deviceCacheBytes.load(ordering: .relaxed) + prefillCacheBytes.load(ordering: .relaxed) }
        var estimatedLogitsScratchBytes: Int { 0 }
    }
    private let kvOnlyPrefill: Bool
    private let persistentPrefill: Bool
    private let io = Mutex((legacyKVBytes: 0, fullBindBytes: 0, updateBytes: 0, logitsReadBytes: 0))
    var prefillIOMetrics: (legacyKVBytes: Int, fullBindBytes: Int, updateBytes: Int, logitsReadBytes: Int) {
        io.withLock { $0 }
    }
    private let plan: FunctionalBatchPlan
    var maxBatchCapacity: Int { plan.capacity }
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

    init(assetURL: URL, prefixCacheBytes: Int = 256 * 1024 * 1024, prefillQueryLimit: Int? = nil, persistentPrefill: Bool = true, nativeKV: Bool = false) async throws {
        guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf:
            assetURL.appendingPathComponent("batch-manifest.json"))) as? [String: Any],
            let abi = manifest["abi"] as? String,
            ["gemma4-readonly-batch-v1", "gemma4-readonly-batch-v2"].contains(abi),
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
        let headless = abi == "gemma4-readonly-batch-v2"
        kvOnlyPrefill = headless
        self.persistentPrefill = persistentPrefill
        self.nativeKV = nativeKV
        guard let contracts = manifest["functions"] as? [[String: Any]] else {
            throw InferenceRuntimeError.invalidState("Missing functional graph contracts")
        }
        let shapes = try contracts.map { contract -> FunctionalBatchPlan.Shape in
            guard let batch = contract["request_batch"] as? Int,
                  let query = contract["query_width"] as? Int,
                  contract["context_capacity"] as? Int == context,
                  contract["name"] as? String == "decode_b\(batch)_q\(query)_c\(context)" else {
                throw InferenceRuntimeError.invalidState("Invalid functional graph contract")
            }
            if headless {
                let expected = (query > 1 ? [] : ["logits"]) + ["new_global_k", "new_global_v", "new_sliding_k", "new_sliding_v"]
                guard contract["outputs"] as? [String] == expected, query == 1 || batch == 1 else {
                    throw InferenceRuntimeError.invalidState("Invalid KV-only prefill contract")
                }
            }
            return .init(batch: batch, query: query)
        }
        plan = try FunctionalBatchPlan(shapes: shapes, context: context, prefillQueryLimit: prefillQueryLimit)
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
        let runtime = try await Self.loadRuntime(at: assetURL.appendingPathComponent("model.aimodel"), context: context, shapes: plan.shapes)
        model = runtime.model
        CLILogger.log("Functional AIModel loaded")
        let embeddingReader = embedding
        let packedURL = assetURL.appendingPathComponent("embedding_interleaved8.bin")
        let packed = FileManager.default.fileExists(atPath: packedURL.path) ? try Data(contentsOf: packedURL, options: .alwaysMapped) : nil
        var loaded: [String: InferenceFunction] = [:]
        var tables: [String: NDArray] = [:]
        var sharedDescriptor: NDArrayDescriptor?
        var sharedTable: NDArray?
        for shape in plan.shapes {
            let (b,q) = (shape.batch, shape.query)
            let key = "\(b):\(q)"
            CLILogger.log("Loading functional entrypoint \(key)")
            guard let fn = runtime.functions[key] else {
                throw InferenceRuntimeError.invalidState("Missing functional batch entrypoint \(key)")
            }
            let expectedOutputs = (headless && q > 1 ? [] : ["logits"]) + ["new_global_k", "new_global_v", "new_sliding_k", "new_sliding_v"]
            guard Set(fn.descriptor.outputNames) == Set(expectedOutputs) else {
                throw InferenceRuntimeError.invalidState("Unexpected functional outputs")
            }
            loaded[key] = fn
            if !fn.descriptor.inputNames.contains("embedding_table") {
                guard headless && q > 1 else { throw InferenceRuntimeError.invalidState("Missing decode embedding table") }
                continue
            }
            guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: "embedding_table") else {
                throw InferenceRuntimeError.invalidState("Invalid embedding descriptor")
            }
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
    private static func loadRuntime(at url: URL, context: Int, shapes: [FunctionalBatchPlan.Shape]) async throws
        -> (model: AIModel, functions: [String: InferenceFunction]) {
        for attempt in 0..<2 {
            let model = try await AIModel(contentsOf: url,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            let names = shapes.map { "decode_b\($0.batch)_q\($0.query)_c\(context)" }
            guard names.allSatisfy({ model.functionNames.contains($0) }) else {
                throw InferenceRuntimeError.invalidState("Functional asset is missing a required entrypoint")
            }
            do {
                var functions: [String: InferenceFunction] = [:]
                for shape in shapes {
                    let (b,q) = (shape.batch, shape.query)
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
        _ = try plan.decodeBatch(for: batchSize)
        return Session(rows: (0..<batchSize).map { _ in emptyState })
    }

    private func cast(_ state: any BatchSessionState) throws -> Session {
        guard let state = state as? Session else { throw InferenceRuntimeError.invalidState("Foreign functional batch session") }
        return state
    }

    func endPrefill(row: Int, session: any BatchSessionState) {
        guard let session = session as? Session else { return }
        if let bank = session.prefillBanks.removeValue(forKey: row) {
            _ = session.prefillCacheBytes.wrappingSubtract(bank.bytes, ordering: .relaxed)
        }
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
        if let old = session.prefillBanks.removeValue(forKey: row) {
            _ = session.prefillCacheBytes.wrappingSubtract(old.bytes, ordering: .relaxed)
        }
    }

    func prefillChunk(row: Int, promptTokens: [Int32], session: any BatchSessionState) async throws -> [LogitsScalarType]? {
        let session = try cast(session)
        var state = session.rows[row]
        // Remove the second host owner before committing new KV; otherwise Swift
        // COW copies the full 16K arrays on every chunk.
        if persistentPrefill { session.rows[row] = emptyState }
        defer { session.rows[row] = state }
        let remaining = promptTokens.count - state.cursor
        let shape = try plan.prefill(remaining: remaining)
        let (physicalBatch,q,valid) = (shape.batch,shape.query,shape.valid)
        var tokens = Array(promptTokens[state.cursor..<state.cursor + valid])
        tokens += Array(repeating: 0, count: q - valid)
        let bank = persistentPrefill && q > 1 ? try bindPrefillBank(session, row: row, query: q, state: state) : nil
        let position = state.cursor
        let result = try await forward(
            tokens: [tokens] + Array(repeating: Array(repeating: Int32(0), count: q), count: physicalBatch - 1),
            states: [state] + Array(repeating: emptyState, count: physicalBatch - 1),
            deviceCaches: bank?.buffers)
        try commit(result, row: 0, batch: physicalBatch, query: q, valid: valid, state: &state)
        if let bank {
            writePrefillUpdates(bank, outputs: result, position: position, query: q, valid: valid)
            bank.cursor = state.cursor
        }
        session.boundCursors[row] = nil
        if state.cursor < promptTokens.count {
            // Keep useful shared prefixes and sparse long checkpoints; copying
            // a full snapshot after every prefill chunk is quadratic work.
            let position = state.cursor
            let shortCheckpoint = position <= 512 && (position & (position - 1)) == 0
            if position == promptTokens.count - 1 || shortCheckpoint || position % 1024 == 0 {
                prefixCache.insert(tokens: Array(promptTokens.prefix(position)), state: state)
                counters.withLock { $0.bytes = prefixCache.residentBytes }
            }
            return nil
        }
        if let old = session.prefillBanks.removeValue(forKey: row) {
            _ = session.prefillCacheBytes.wrappingSubtract(old.bytes, ordering: .relaxed)
        }
        guard let logits = result["logits"] else { throw InferenceRuntimeError.invalidState("Final prefill needs logits") }
        return capped(Array(logits[(valid - 1) * vocab..<valid * vocab]))
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
        let selected = startPositions.indices.filter { startPositions[$0] > 0 }
        if selected.isEmpty { return Array(repeating: [], count: session.rows.count) }
        for row in selected {
            guard startPositions[row] == session.rows[row].cursor,
                  session.rows[row].cursor < geometry.context else {
                throw InferenceRuntimeError.invalidState("Functional row cursor mismatch or context exhausted")
            }
        }
        let batch = try plan.decodeBatch(for: selected.count)
        var states = selected.map { session.rows[$0] }
        var inputTokens = selected.map { tokensPerRow[$0] }
        var logicalRows = selected
        var active = Array(repeating: true, count: selected.count)
        for _ in selected.count..<batch {
            states.append(emptyState); inputTokens.append([0]); active.append(false); logicalRows.append(-1)
        }
        try bindDeviceRows(session, states: states, active: active, logicalRows: logicalRows)
        let result = try await forward(tokens: inputTokens, states: states, deviceCaches: session.deviceCaches)
        states.removeAll()
        var logits = Array(repeating: [LogitsScalarType](), count: session.rows.count)
        for (physical,row) in selected.enumerated() {
            try commit(result, row: physical, batch: batch, query: 1, valid: 1, state: &session.rows[row])
            writeDeviceUpdates(session, outputs: result, row: physical, position: startPositions[row], physicalBatch: batch)
            session.boundCursors[row] = session.rows[row].cursor
            logits[row] = capped(Array(result["logits"]![physical * vocab..<(physical + 1) * vocab]))
        }
        return logits
    }

    func allPositionLogits(tokensPerRow: [[Int32]], startPosition: Int, session: any BatchSessionState) async throws -> [LogitsScalarType] {
        throw InferenceRuntimeError.invalidState("Functional batch scoring is not enabled")
    }

    private func capped(_ logits: [Float16]) -> [LogitsScalarType] {
        guard softcap > 0 else { return logits }
        return logits.map { Float16(softcap * tanh(Float($0) / softcap)) }
    }

    private func commit(_ outputs: StepOutput, row: Int, batch: Int, query: Int, valid: Int, state: inout ReadonlyBatchKV) throws {
        let layouts = cacheLayouts
        let names = ["new_global_k","new_global_v","new_sliding_k","new_sliding_v"]
        try state.commit(queryWidth: query, validTokens: valid) { index, destination, capacity, cursor in
            let (_,_,layers,channels,_) = layouts[index]
            outputs.withKV(names[index]) { read in
                for layer in 0..<layers { for channel in 0..<channels { for j in 0..<valid {
                    let slot = index < 2 ? cursor + j : (cursor + j) % capacity
                    destination[(layer * channels + channel) * capacity + slot] =
                        read(((layer * batch + row) * channels + channel) * query + j)
                } } }
            }
        }
    }

    private var cacheLayouts: [(String, KeyPath<ReadonlyBatchKV, [Float16]>, Int, Int, Int)] {
        [("key_cache", \ReadonlyBatchKV.globalK, geometry.globalLayers, geometry.globalChannels, geometry.context),
         ("value_cache", \ReadonlyBatchKV.globalV, geometry.globalLayers, geometry.globalChannels, geometry.context),
         ("sliding_key_cache", \ReadonlyBatchKV.slidingK, geometry.slidingLayers, geometry.slidingChannels, geometry.ring),
         ("sliding_value_cache", \ReadonlyBatchKV.slidingV, geometry.slidingLayers, geometry.slidingChannels, geometry.ring)]
    }

    /// Rebind complete rows only at admission/restoration. Ordinary decode updates
    /// just the new K/V position in these persistent runtime-layout buffers.
    private func bindDeviceRows(_ session: Session, states: [ReadonlyBatchKV], active: [Bool], logicalRows: [Int]) throws {
        let batch = states.count
        if session.deviceBatch != batch {
            session.deviceCaches.removeAll()
            session.deviceCacheBytes.store(0, ordering: .relaxed)
            session.deviceBatch = batch
        }
        if session.deviceRows != logicalRows {
            session.boundCursors = Array(repeating: nil, count: session.rows.count)
            session.deviceRows = logicalRows
        }
        let fn = functions["\(batch):1"]!
        for (name,path,layers,channels,capacity) in cacheLayouts {
            if session.deviceCaches[name] == nil {
                guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) else {
                    throw InferenceRuntimeError.invalidState("Missing cache descriptor")
                }
                guard desc.shape == [layers * batch, channels, 1, capacity],
                      (desc.interleaveLayout?.factor ?? 1) == 1 || desc.interleaveLayout?.dimension != 2 else {
                    throw InferenceRuntimeError.invalidState("Unsupported persistent cache layout")
                }
                var array = NDArray(descriptor: desc)
                _ = array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in memset(ptr, 0, desc.minimumByteCount) }
                session.deviceCaches[name] = array
                _ = session.deviceCacheBytes.wrappingAdd(desc.minimumByteCount, ordering: .relaxed)
            }
            for row in states.indices where active[row] && session.boundCursors[logicalRows[row]] != states[row].cursor {
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
        for row in states.indices where active[row] { session.boundCursors[logicalRows[row]] = states[row].cursor }
    }

    /// A request owns one descriptor-checked bank. q32/q64 share it only when
    /// all physical descriptors agree. No layout or residency is inferred from shape.
    private func bindPrefillBank(_ session: Session, row: Int, query: Int, state: ReadonlyBatchKV) throws -> PrefillBank {
        let fn = functions["1:\(query)"]!
        var descriptors: [String: NDArrayDescriptor] = [:]
        for (name,_,layers,channels,capacity) in cacheLayouts {
            guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name),
                  desc.shape == [layers, channels, 1, capacity],
                  (desc.interleaveLayout?.factor ?? 1) == 1 || desc.interleaveLayout?.dimension != 2 else {
                throw InferenceRuntimeError.invalidState("Unsupported prefill KV descriptor")
            }
            descriptors[name] = desc
        }
        if let bank = session.prefillBanks[row], bank.descriptors == descriptors {
            guard bank.cursor == state.cursor else { throw InferenceRuntimeError.invalidState("Stale prefill bank") }
            return bank
        }
        if let old = session.prefillBanks.removeValue(forKey: row) {
            _ = session.prefillCacheBytes.wrappingSubtract(old.bytes, ordering: .relaxed)
        }
        let bank = PrefillBank()
        bank.descriptors = descriptors
        bank.cursor = state.cursor
        for (name,path,layers,channels,capacity) in cacheLayouts {
            let desc = descriptors[name]!
            var array = NDArray(descriptor: desc)
            _ = array.mutableRawView().withUnsafeMutableBytes { ptr,_,_ in memset(ptr,0,desc.minimumByteCount) }
            let used = min(state.cursor, capacity)
            if used > 0 {
                let source = state[keyPath: path]
                let factor = array.interleaveLayout?.factor ?? 1
                let dim = array.interleaveLayout?.dimension ?? 0
                array.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr,_,strides in
                    for layer in 0..<layers { for channel in 0..<channels {
                        let base = (dim == 0 ? layer / factor : layer) * strides[0]
                            + (dim == 1 ? channel / factor : channel) * strides[1]
                            + (dim == 0 ? layer % factor : (dim == 1 ? channel % factor : 0))
                        for position in 0..<used {
                            let offset = base + (dim == 3 ? position / factor : position) * strides[3]
                                + (dim == 3 ? position % factor : 0)
                            ptr[offset] = source[(layer * channels + channel) * capacity + position]
                        }
                    } }
                }
                io.withLock { $0.fullBindBytes += layers * channels * used * MemoryLayout<Float16>.stride }
            }
            bank.buffers[name] = array
            bank.bytes += desc.minimumByteCount
        }
        session.prefillBanks[row] = bank
        _ = session.prefillCacheBytes.wrappingAdd(bank.bytes, ordering: .relaxed)
        return bank
    }

    private func writePrefillUpdates(_ bank: PrefillBank, outputs: StepOutput, position: Int, query: Int, valid: Int) {
        let names = ["new_global_k","new_global_v","new_sliding_k","new_sliding_v"]
        for (index,layout) in cacheLayouts.enumerated() {
            let (name,_,layers,channels,capacity) = layout
            var destination = bank.buffers.removeValue(forKey: name)!
            defer { bank.buffers[name] = destination }
            outputs.withKV(names[index]) { read in
            let factor = destination.interleaveLayout?.factor ?? 1
            let dim = destination.interleaveLayout?.dimension ?? 0
            let view = destination.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr,_,strides in
                for layer in 0..<layers { for channel in 0..<channels {
                    let base = (dim == 0 ? layer / factor : layer) * strides[0]
                        + (dim == 1 ? channel / factor : channel) * strides[1]
                        + (dim == 0 ? layer % factor : (dim == 1 ? channel % factor : 0))
                    for j in 0..<valid {
                        let slot = index < 2 ? position + j : (position + j) % capacity
                        let offset = base + (dim == 3 ? slot / factor : slot) * strides[3]
                            + (dim == 3 ? slot % factor : 0)
                        ptr[offset] = read((layer * channels + channel) * query + j)
                    }
                } }
            }
            }
            io.withLock { $0.updateBytes += layers * channels * valid * MemoryLayout<Float16>.stride }
        }
    }

    private func writeDeviceUpdates(_ session: Session, outputs: StepOutput, row: Int, position: Int, physicalBatch batch: Int) {
        let outputNames = ["new_global_k","new_global_v","new_sliding_k","new_sliding_v"]
        for (index,layout) in cacheLayouts.enumerated() {
            let (name,_,layers,channels,capacity) = layout
            let slot = index < 2 ? position : position % capacity
            var destination = session.deviceCaches.removeValue(forKey: name)!
            defer { session.deviceCaches[name] = destination }
            outputs.withKV(outputNames[index]) { read in
            let factor = destination.interleaveLayout?.factor ?? 1
            let dim = destination.interleaveLayout?.dimension ?? 0
            let view = destination.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, _, strides in
                for layer in 0..<layers { for channel in 0..<channels {
                    let outer = layer * batch + row
                    let offset = (dim == 0 ? outer / factor : outer) * strides[0]
                        + (dim == 1 ? channel / factor : channel) * strides[1]
                        + (dim == 3 ? slot / factor : slot) * strides[3]
                        + (dim == 0 ? outer % factor : (dim == 1 ? channel % factor : (dim == 3 ? slot % factor : 0)))
                    ptr[offset] = read((layer * batch + row) * channels + channel)
                } }
            }
            }
        }
    }

    private func forward(tokens: [[Int32]], states: [ReadonlyBatchKV], deviceCaches: [String: NDArray]? = nil) async throws -> StepOutput {
        let b = tokens.count; let q = tokens[0].count; let key = "\(b):\(q)"
        guard let fn = functions[key], tokens.flatMap({$0}).allSatisfy({ $0 >= 0 && $0 < vocab }) else {
            throw InferenceRuntimeError.invalidState("Unsupported functional shape or token")
        }
        var inputs: [String: NDArray] = [:]
        if let table = embeddingInputs[key] { inputs["embedding_table"] = table }
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
            if q > 1 { io.withLock { $0.legacyKVBytes += flat.count * MemoryLayout<Float16>.stride } }
        }
        var output = try await fn.run(inputs: inputs)
        var result = StepOutput()
        let outputNames = (kvOnlyPrefill && q > 1 ? [] : ["logits"]) + ["new_global_k","new_global_v","new_sliding_k","new_sliding_v"]
        for name in outputNames {
            guard let array = output.remove(name)?.ndArray else { throw InferenceRuntimeError.invalidState("Missing output \(name)") }
            if name != "logits", nativeKV {
                let index = outputNames.filter { $0 != "logits" }.firstIndex(of: name)!
                let (_,_,layers,channels,_) = cacheLayouts[index]
                guard array.shape == [layers * b, channels, 1, q] else {
                    throw InferenceRuntimeError.invalidState("Invalid native KV output extent")
                }
                let finite = FunctionalTensorIO.withReader(array) { read in
                    (0..<(layers * b * channels * q)).allSatisfy { read($0).isFinite }
                }
                guard finite else { throw InferenceRuntimeError.invalidState("Nonfinite functional output") }
                result.native[name] = array
            } else {
                let values = FunctionalTensorIO.read(array)
                guard values.allSatisfy({ $0.isFinite }) else { throw InferenceRuntimeError.invalidState("Nonfinite functional output") }
                if name == "logits" {
                    result.logits = values
                    io.withLock { $0.logitsReadBytes += values.count * MemoryLayout<Float16>.stride }
                } else { result.dense[name] = values }
            }
        }
        return result
    }
}
