// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause

/// Bridge the existing dynamic cache layout to the backend-neutral scheduler contract.
extension CoreAISequentialEngine {
    func makeBatchSessionState(batchSize: Int) throws -> any BatchSessionState {
        try makeSessionState(batchSize: batchSize)
    }

    private func sequentialSession(_ session: any BatchSessionState) throws -> GenerationSessionState {
        guard let session = session as? GenerationSessionState else {
            throw InferenceRuntimeError.invalidState("Sequential backend received a foreign batch session")
        }
        return session
    }

    func decodeBatch(
        tokensPerRow: [[Int32]], startPositions: [Int], session: any BatchSessionState
    ) async throws -> [[LogitsScalarType]] {
        try await decodeBatch(
            tokensPerRow: tokensPerRow, startPositions: startPositions,
            session: sequentialSession(session))
    }

    func prefillRow(
        row: Int, promptTokens: [Int32], session: any BatchSessionState
    ) async throws -> [LogitsScalarType] {
        try await prefillRow(row: row, promptTokens: promptTokens, session: sequentialSession(session))
    }

    func allPositionLogits(
        tokensPerRow: [[Int32]], startPosition: Int, session: any BatchSessionState
    ) async throws -> [LogitsScalarType] {
        try await allPositionLogits(
            tokensPerRow: tokensPerRow, startPosition: startPosition,
            session: sequentialSession(session))
    }
}
