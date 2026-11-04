import Testing
@testable import CoreAILanguageModels

struct FunctionalBatchPlanTests {
    @Test func sparseCapacitiesPadAndExcludeUnsafeSingleton() throws {
        let plan = try FunctionalBatchPlan(shapes: [.init(batch: 1, query: 1),
            .init(batch: 2, query: 1), .init(batch: 4, query: 1),
            .init(batch: 1, query: 32), .init(batch: 1, query: 64)], context: 16384)
        #expect(plan.capacity == 4)
        #expect(try plan.decodeBatch(for: 1) == 2)
        #expect(try plan.decodeBatch(for: 3) == 4)
        #expect(!plan.shapes.contains(.init(batch: 1, query: 1)))
        #expect(throws: InferenceRuntimeError.self) { try plan.decodeBatch(for: 5) }
        let tail = try plan.prefill(remaining: 1)
        #expect(tail.batch == 2 && tail.query == 1 && tail.valid == 1)
        let short = try plan.prefill(remaining: 18)
        #expect(short.query == 32 && short.valid == 17)
        let long = try plan.prefill(remaining: 66)
        #expect(long.query == 64 && long.valid == 64)
    }
    @Test func malformedOrMissingGraphsAreRejected() throws {
        #expect(throws: InferenceRuntimeError.self) {
            try FunctionalBatchPlan(shapes: [.init(batch: 1, query: 1), .init(batch: 1, query: 32)], context: 16384)
        }
        #expect(throws: InferenceRuntimeError.self) {
            try FunctionalBatchPlan(shapes: [.init(batch: 2, query: 1), .init(batch: 2, query: 1)], context: 2048)
        }
        let small = try FunctionalBatchPlan(shapes: [.init(batch: 1, query: 1), .init(batch: 1, query: 32)], context: 2048)
        #expect(try small.decodeBatch(for: 1) == 1)
    }
}
