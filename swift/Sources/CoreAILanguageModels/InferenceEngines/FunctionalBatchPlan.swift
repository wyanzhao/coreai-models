// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause

/// Static graph capabilities, separate from scheduler occupancy.
struct FunctionalBatchPlan {
    struct Shape: Hashable {
        let batch: Int
        let query: Int
        var key: String { "\(batch):\(query)" }
    }
    let shapes: [Shape]
    let decodeBatches: [Int]
    let prefillWidths: [Int]
    var capacity: Int { decodeBatches.last! }

    init(shapes: [Shape], context: Int, prefillQueryLimit: Int? = nil) throws {
        guard context > 0, !shapes.isEmpty, Set(shapes).count == shapes.count,
              shapes.allSatisfy({ (1...8).contains($0.batch) && (1...256).contains($0.query) }) else {
            throw InferenceRuntimeError.invalidState("Invalid functional graph shapes")
        }
        // Preserve the measured numerical exclusion on the current 16K runtime.
        self.shapes = shapes.filter { !(context >= 16384 && $0.batch == 1 && $0.query == 1) }
        decodeBatches = self.shapes.filter { $0.query == 1 }.map(\.batch).sorted()
        prefillWidths = self.shapes.filter { $0.batch == 1 && $0.query > 1 && $0.query <= (prefillQueryLimit ?? 256) }.map(\.query).sorted()
        guard !decodeBatches.isEmpty, !prefillWidths.isEmpty else {
            throw InferenceRuntimeError.invalidState("Functional asset needs decode and prefill graphs")
        }
    }

    func decodeBatch(for rows: Int) throws -> Int {
        guard rows > 0, let batch = decodeBatches.first(where: { $0 >= rows }) else {
            throw InferenceRuntimeError.invalidState("Request rows exceed exported batch capacity")
        }
        return batch
    }

    func prefill(remaining: Int) throws -> (batch: Int, query: Int, valid: Int) {
        guard remaining > 0 else { throw InferenceRuntimeError.invalidState("Prefill already complete") }
        let valid = remaining == 1 ? 1 : min(prefillWidths.last!, remaining - 1)
        if valid == 1 { return (try decodeBatch(for: 1), 1, 1) }
        return (1, prefillWidths.first(where: { $0 >= valid })!, valid)
    }
}
