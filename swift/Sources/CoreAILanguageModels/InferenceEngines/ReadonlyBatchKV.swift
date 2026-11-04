// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause

/// Host-owned functional KV state. Array value semantics keep retained snapshots immutable.
/// Dense per-row storage is [storingLayer, channel, sequence]; device packing is separate.
struct ReadonlyBatchKV {
    struct Geometry: Equatable {
        let context: Int
        let ring: Int
        let window: Int
        let globalLayers: Int
        let slidingLayers: Int
        let globalChannels: Int
        let slidingChannels: Int
    }
    enum Failure: Error { case invalidGeometry, invalidUpdate, capacityExceeded }
    let geometry: Geometry
    private(set) var cursor = 0
    private(set) var globalK: [Float16]
    private(set) var globalV: [Float16]
    private(set) var slidingK: [Float16]
    private(set) var slidingV: [Float16]
    private(set) var slidingPositions: [Int]

    init(geometry g: Geometry) throws {
        guard g.context > 0, g.ring > 0, g.window > 0, g.ring >= g.window,
              g.globalLayers > 0, g.slidingLayers > 0,
              g.globalChannels > 0, g.slidingChannels > 0 else { throw Failure.invalidGeometry }
        geometry = g
        globalK = Array(repeating: 0, count: g.globalLayers * g.globalChannels * g.context)
        globalV = globalK
        slidingK = Array(repeating: 0, count: g.slidingLayers * g.slidingChannels * g.ring)
        slidingV = slidingK
        slidingPositions = Array(repeating: -1, count: g.ring)
    }

    var byteCount: Int {
        (globalK.count + globalV.count + slidingK.count + slidingV.count) * MemoryLayout<Float16>.stride
            + slidingPositions.count * MemoryLayout<Int>.stride
    }

    /// Compact immutable snapshot: retain only committed global positions and
    /// occupied ring slots, rather than charging every short prefix for 16K KV.
    struct Snapshot {
        let geometry: Geometry
        let cursor: Int
        let globalK: [Float16]
        let globalV: [Float16]
        let slidingK: [Float16]
        let slidingV: [Float16]
        let positions: [Int]
        var byteCount: Int {
            (globalK.count + globalV.count + slidingK.count + slidingV.count) * MemoryLayout<Float16>.stride
                + positions.count * MemoryLayout<Int>.stride
        }
    }

    func snapshot() -> Snapshot {
        func compact(_ values: [Float16], width: Int, used: Int) -> [Float16] {
            var result: [Float16] = []
            result.reserveCapacity(values.count / width * used)
            for row in 0..<(values.count / width) {
                result.append(contentsOf: values[row * width..<row * width + used])
            }
            return result
        }
        let usedRing = min(cursor, geometry.ring)
        return Snapshot(geometry: geometry, cursor: cursor,
            globalK: compact(globalK, width: geometry.context, used: cursor),
            globalV: compact(globalV, width: geometry.context, used: cursor),
            slidingK: compact(slidingK, width: geometry.ring, used: usedRing),
            slidingV: compact(slidingV, width: geometry.ring, used: usedRing),
            positions: Array(slidingPositions.prefix(usedRing)))
    }

    init(snapshot: Snapshot) throws {
        try self.init(geometry: snapshot.geometry)
        guard snapshot.cursor > 0, snapshot.cursor <= geometry.context,
              snapshot.positions.count == min(snapshot.cursor, geometry.ring) else { throw Failure.invalidUpdate }
        func restore(_ source: [Float16], into destination: inout [Float16], width: Int, used: Int) throws {
            guard source.count == destination.count / width * used else { throw Failure.invalidUpdate }
            for row in 0..<(destination.count / width) {
                destination.replaceSubrange(row * width..<row * width + used,
                    with: source[row * used..<(row + 1) * used])
            }
        }
        try restore(snapshot.globalK, into: &globalK, width: geometry.context, used: snapshot.cursor)
        try restore(snapshot.globalV, into: &globalV, width: geometry.context, used: snapshot.cursor)
        try restore(snapshot.slidingK, into: &slidingK, width: geometry.ring, used: snapshot.positions.count)
        try restore(snapshot.slidingV, into: &slidingV, width: geometry.ring, used: snapshot.positions.count)
        slidingPositions.replaceSubrange(0..<snapshot.positions.count, with: snapshot.positions)
        cursor = snapshot.cursor
    }

    /// Accept only valid query tokens (exclude padding from a fixed-width graph).
    /// Validate every extent before mutating so failed runs leave the row unchanged.
    mutating func commit(globalK newGK: [Float16], globalV newGV: [Float16],
                         slidingK newSK: [Float16], slidingV newSV: [Float16],
                         queryWidth: Int, validTokens: Int) throws {
        let g = geometry
        guard queryWidth > 0, validTokens > 0, validTokens <= queryWidth,
              newGK.count == g.globalLayers * g.globalChannels * queryWidth,
              newGV.count == newGK.count,
              newSK.count == g.slidingLayers * g.slidingChannels * queryWidth,
              newSV.count == newSK.count else { throw Failure.invalidUpdate }
        guard validTokens <= g.context - cursor else { throw Failure.capacityExceeded }
        for channel in 0..<(g.globalLayers * g.globalChannels) {
            for j in 0..<validTokens {
                globalK[channel * g.context + cursor + j] = newGK[channel * queryWidth + j]
                globalV[channel * g.context + cursor + j] = newGV[channel * queryWidth + j]
            }
        }
        for channel in 0..<(g.slidingLayers * g.slidingChannels) {
            for j in 0..<validTokens {
                let slot = (cursor + j) % g.ring
                slidingK[channel * g.ring + slot] = newSK[channel * queryWidth + j]
                slidingV[channel * g.ring + slot] = newSV[channel * queryWidth + j]
            }
        }
        for j in 0..<validTokens { slidingPositions[(cursor + j) % g.ring] = cursor + j }
        cursor += validTokens
    }

    /// The caller validates all source tensor extents and finiteness before entering.
    /// Each writer borrows its native source and writes into this row's dense snapshot.
    mutating func commit(queryWidth: Int, validTokens: Int,
                         write: (Int, inout [Float16], Int, Int) -> Void) throws {
        guard queryWidth > 0, validTokens > 0, validTokens <= queryWidth else { throw Failure.invalidUpdate }
        guard validTokens <= geometry.context - cursor else { throw Failure.capacityExceeded }
        write(0, &globalK, geometry.context, cursor)
        write(1, &globalV, geometry.context, cursor)
        write(2, &slidingK, geometry.ring, cursor)
        write(3, &slidingV, geometry.ring, cursor)
        for j in 0..<validTokens { slidingPositions[(cursor + j) % geometry.ring] = cursor + j }
        cursor += validTokens
    }

    /// Layout [pastCapacity + q, q] matches the graph's [B, S, 1, q].
    /// Mask per query position, including appended tokens outside the sliding window.
    func mask(queryWidth q: Int, sliding: Bool) -> [Float16] {
        precondition(q > 0)
        let capacity = sliding ? geometry.ring : geometry.context
        var result = [Float16](repeating: -40000, count: (capacity + q) * q)
        for j in 0..<q {
            let absolute = cursor + j
            for slot in 0..<capacity {
                let position = sliding ? slidingPositions[slot] : slot
                if position >= 0 && position < cursor && (!sliding || position > absolute - geometry.window) {
                    result[slot * q + j] = 0
                }
            }
            for k in 0...j where !sliding || cursor + k > absolute - geometry.window {
                result[(capacity + k) * q + j] = 0
            }
        }
        return result
    }
}

/// One instance belongs to one model/geometry. Access is serialized by the backend.
/// Stores exact token prefixes and complete ring state; a token match alone is not a KV hit.
final class ReadonlyPrefixCache {
    private struct Entry {
        let tokens: [Int32]
        let state: ReadonlyBatchKV.Snapshot
        var accessed: UInt64
        let shortTier: Bool
        var bytes: Int { state.byteCount + tokens.count * MemoryLayout<Int32>.stride }
    }
    let byteLimit: Int
    private var entries: [Entry] = []
    private var clock: UInt64 = 0
    private(set) var residentBytes = 0
    init(byteLimit: Int) { self.byteLimit = max(0, byteLimit) }

    func insert(tokens: [Int32], state: ReadonlyBatchKV) {
        guard byteLimit > 0, !tokens.isEmpty, tokens.count == state.cursor else { return }
        let snapshot = state.snapshot()
        let size = snapshot.byteCount + tokens.count * MemoryLayout<Int32>.stride
        // Reserve a quarter of a practical cache for reusable system prefixes.
        // Large one-off documents must not evict every short shared prefix.
        let partitioned = byteLimit >= 4 * 1024 * 1024
        let shortTier = partitioned && tokens.count <= 512
        let tierLimit = partitioned ? (shortTier ? byteLimit / 4 : byteLimit - byteLimit / 4) : byteLimit
        guard size <= tierLimit else { return }
        if let existing = entries.firstIndex(where: { $0.tokens == tokens }) {
            residentBytes -= entries.remove(at: existing).bytes
        }
        while entries.filter({ $0.shortTier == shortTier }).reduce(0, { $0 + $1.bytes }) > tierLimit - size {
            guard let victim = entries.indices.filter({ entries[$0].shortTier == shortTier }).min(by: {
                entries[$0].accessed < entries[$1].accessed
            }) else { return }
            residentBytes -= entries.remove(at: victim).bytes
        }
        clock &+= 1
        entries.append(Entry(tokens: tokens, state: snapshot, accessed: clock, shortTier: shortTier))
        residentBytes += size
    }

    func removeAll() { entries.removeAll(); residentBytes = 0; clock = 0 }

    /// Leave at least one prompt token to evaluate; snapshots intentionally omit logits.
    func lookup(prompt: [Int32], geometry: ReadonlyBatchKV.Geometry) -> ReadonlyBatchKV? {
        let matches = entries.indices.filter {
            entries[$0].state.geometry == geometry && entries[$0].tokens.count < prompt.count
                && prompt.starts(with: entries[$0].tokens)
        }
        guard let best = matches.max(by: { entries[$0].tokens.count < entries[$1].tokens.count }) else {
            return nil
        }
        clock &+= 1
        entries[best].accessed = clock
        return try? ReadonlyBatchKV(snapshot: entries[best].state)
    }
}
