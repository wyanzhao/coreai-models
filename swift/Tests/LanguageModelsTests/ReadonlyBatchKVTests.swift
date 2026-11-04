import Testing
@testable import CoreAILanguageModels

struct ReadonlyBatchKVTests {
    let geometry = ReadonlyBatchKV.Geometry(context: 32, ring: 4, window: 3,
        globalLayers: 1, slidingLayers: 1, globalChannels: 1, slidingChannels: 1)
    func append(_ values: [Float16], to state: inout ReadonlyBatchKV, valid: Int? = nil) throws {
        try state.commit(globalK: values, globalV: values, slidingK: values, slidingV: values,
            queryWidth: values.count, validTokens: valid ?? values.count)
    }

    @Test func snapshotSurvivesRingWrapAndRestore() throws {
        var state = try ReadonlyBatchKV(geometry: geometry)
        try append([1,2,3], to: &state)
        let snapshot = state
        try append([4,5,6,7,8], to: &state)
        #expect(state.slidingPositions == [4,5,6,7])
        #expect(snapshot.cursor == 3)
        #expect(snapshot.slidingK == [1,2,3,0])
        let compact = state.snapshot()
        let fromCompact = try ReadonlyBatchKV(snapshot: compact)
        #expect(fromCompact.globalK == state.globalK)
        #expect(fromCompact.slidingK == state.slidingK)
        #expect(fromCompact.slidingPositions == state.slidingPositions)
        #expect(compact.byteCount < state.byteCount)
        var restored = snapshot
        try append([9], to: &restored)
        #expect(restored.globalK.prefix(4) == [1,2,3,9])
        #expect(state.globalK.prefix(4) == [1,2,3,4])
    }

    @Test func slidingMaskUsesAbsolutePositionsAndQueryWindow() throws {
        var state = try ReadonlyBatchKV(geometry: geometry)
        try append([1,2,3,4,5,6], to: &state)
        let mask = state.mask(queryWidth: 4, sliding: true)
        // Ring slots map to absolute positions [4,5,2,3]. Query 6 sees 4,5,6.
        #expect((0..<8).filter { mask[$0 * 4] == 0 } == [0,1,4])
        // Query 9 sees only appended 7,8,9, never stale ring slots or query 6.
        #expect((0..<8).filter { mask[$0 * 4 + 3] == 0 } == [5,6,7])
    }

    @Test func paddingAndInvalidCommitDoNotPolluteCache() throws {
        var state = try ReadonlyBatchKV(geometry: geometry)
        try append([1,2,99,99], to: &state, valid: 2)
        #expect(state.cursor == 2)
        #expect(state.globalK.prefix(4) == [1,2,0,0])
        #expect(throws: ReadonlyBatchKV.Failure.self) {
            try state.commit(globalK: [], globalV: [], slidingK: [], slidingV: [],
                queryWidth: 1, validTokens: 1)
        }
        #expect(state.cursor == 2)
    }

    @Test func prefixCacheReturnsLongestCompatibleSnapshotWithinBudget() throws {
        var state = try ReadonlyBatchKV(geometry: geometry)
        let cache = ReadonlyPrefixCache(byteLimit: 140)
        try append([1,2], to: &state)
        cache.insert(tokens: [1,2], state: state)
        try append([3], to: &state)
        cache.insert(tokens: [1,2,3], state: state)
        #expect(cache.lookup(prompt: [1,2,3,4], geometry: geometry)?.cursor == 3)
        #expect(cache.lookup(prompt: [1,2], geometry: geometry) == nil)
        #expect(cache.lookup(prompt: [8,9,10], geometry: geometry) == nil)
        try append([4], to: &state)
        cache.insert(tokens: [1,2,3,4], state: state)
        #expect(cache.residentBytes <= cache.byteLimit)
        #expect(cache.lookup(prompt: [1,2,9], geometry: geometry) == nil)
        #expect(cache.lookup(prompt: [1,2,3,4,5], geometry: geometry)?.cursor == 4)
    }
}

extension ReadonlyBatchKVTests {
    @Test func longDocumentsDoNotEvictProtectedShortPrefixes() throws {
        let geometry = ReadonlyBatchKV.Geometry(context: 1024, ring: 4, window: 3,
            globalLayers: 1, slidingLayers: 1, globalChannels: 512, slidingChannels: 1)
        func make(_ count: Int) throws -> ReadonlyBatchKV {
            var state = try ReadonlyBatchKV(geometry: geometry)
            let global = [Float16](repeating: 1, count: count * 512)
            let sliding = [Float16](repeating: 1, count: count)
            try state.commit(globalK: global, globalV: global, slidingK: sliding, slidingV: sliding,
                queryWidth: count, validTokens: count)
            return state
        }
        let cache = ReadonlyPrefixCache(byteLimit: 4 * 1024 * 1024)
        let short = [Int32](repeating: 1, count: 256)
        cache.insert(tokens: short, state: try make(256))
        let longState = try make(900)
        for id in 2...5 { cache.insert(tokens: Array(repeating: Int32(id), count: 900), state: longState) }
        #expect(cache.lookup(prompt: short + [9], geometry: geometry)?.cursor == 256)
        #expect(cache.lookup(prompt: Array(repeating: 2, count: 901), geometry: geometry) == nil)
        #expect(cache.lookup(prompt: Array(repeating: 5, count: 901), geometry: geometry)?.cursor == 900)
        #expect(cache.residentBytes <= cache.byteLimit)
    }
}
