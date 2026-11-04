// Copyright 2026 Apple Inc.
// SPDX-License-Identifier: BSD-3-Clause
import CoreAI
import Foundation

/// Logical row-major values to/from the runtime's padded and interleaved layout.
enum FunctionalTensorIO {
    static func make<T: BitwiseCopyable>(_ values: [T], descriptor: NDArrayDescriptor) throws -> NDArray {
        try values.withUnsafeBytes { try make($0, as: T.self, descriptor: descriptor) }
    }

    static func make<T: BitwiseCopyable>(
        _ source: UnsafeRawBufferPointer, as type: T.Type, descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        let shape = descriptor.shape
        guard shape.reduce(1, *) * MemoryLayout<T>.stride == source.count else {
            throw InferenceRuntimeError.invalidState("Functional tensor logical extent mismatch")
        }
        var array = NDArray(descriptor: descriptor)
        let factor = array.interleaveLayout?.factor ?? 1
        let dimension = array.interleaveLayout?.dimension ?? 0
        _ = array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            memset(ptr, 0, descriptor.minimumByteCount)
        }
        array.mutableView(as: T.self).withUnsafeMutablePointer { ptr, _, strides in
            if factor == 1, let run = shape.last, strides[shape.count - 1] == 1 {
                for row in 0..<(shape.reduce(1, *) / run) {
                    var remaining = row; var offset = 0
                    for d in (0..<(shape.count - 1)).reversed() {
                        offset += (remaining % shape[d]) * strides[d]; remaining /= shape[d]
                    }
                    memcpy(ptr.advanced(by: offset), source.baseAddress!.advanced(by: row * run * MemoryLayout<T>.stride), run * MemoryLayout<T>.stride)
                }
                return
            }
            for i in 0..<shape.reduce(1, *) {
                var remaining = i
                var offset = 0
                for d in (0..<shape.count).reversed() {
                    let coordinate = remaining % shape[d]
                    remaining /= shape[d]
                    offset += d == dimension
                        ? coordinate / factor * strides[d] + coordinate % factor
                        : coordinate * strides[d]
                }
                ptr[offset] = source.loadUnaligned(fromByteOffset: i * MemoryLayout<T>.stride, as: T.self)
            }
        }
        return array
    }

    /// Borrow physical storage for the duration of body; never materialize a dense copy.
    @inline(__always)
    static func withReader<R>(_ array: NDArray, _ body: ((Int) -> Float16) throws -> R) rethrows -> R {
        let shape = array.shape
        let factor = array.interleaveLayout?.factor ?? 1
        let dimension = array.interleaveLayout?.dimension ?? 0
        return try array.view(as: Float16.self).withUnsafePointer { ptr, _, strides in
            // Auto-layout outputs can already be contiguous. Do not decode an
            // address per scalar when the logical and physical indices agree.
            if factor == 1, shape.count == 4, shape[2] == 1,
               strides[3] == 1, strides[1] == shape[3],
               strides[0] == shape[1] * shape[3] {
                return try body { ptr[$0] }
            }
            // KV uses rank4 BC1S with power-of-two channel/query extents.
            // Hoist layout decoding; integer division per scalar is expensive.
            if shape.count == 4, shape[2] == 1,
               shape[1].nonzeroBitCount == 1, shape[3].nonzeroBitCount == 1,
               factor.nonzeroBitCount == 1 {
                let qShift = shape[3].trailingZeroBitCount
                let outerShift = qShift + shape[1].trailingZeroBitCount
                let channelMask = shape[1] - 1, queryMask = shape[3] - 1
                let s0 = strides[0], s1 = strides[1], s3 = strides[3]
                let fShift = factor.trailingZeroBitCount, fMask = factor - 1
                if factor == 1 {
                    return try body { i in
                        ptr[(i >> outerShift) * s0 + ((i >> qShift) & channelMask) * s1 + (i & queryMask) * s3]
                    }
                }
                if dimension == 0 {
                    return try body { i in
                        let outer = i >> outerShift
                        return ptr[(outer >> fShift) * s0 + ((i >> qShift) & channelMask) * s1
                            + (i & queryMask) * s3 + (outer & fMask)]
                    }
                }
                if dimension == 1 {
                    return try body { i in
                        let channel = (i >> qShift) & channelMask
                        return ptr[(i >> outerShift) * s0 + (channel >> fShift) * s1
                            + (i & queryMask) * s3 + (channel & fMask)]
                    }
                }
                if dimension == 3 {
                    return try body { i in
                        let query = i & queryMask
                        return ptr[(i >> outerShift) * s0 + ((i >> qShift) & channelMask) * s1
                            + (query >> fShift) * s3 + (query & fMask)]
                    }
                }
            }
            return try body { index in
                var remaining = index
                var offset = 0
                for d in (0..<shape.count).reversed() {
                    let coordinate = remaining % shape[d]
                    remaining /= shape[d]
                    offset += d == dimension
                        ? coordinate / factor * strides[d] + coordinate % factor
                        : coordinate * strides[d]
                }
                return ptr[offset]
            }
        }
    }

    static func read(_ array: NDArray) -> [Float16] {
        let shape = array.shape
        let factor = array.interleaveLayout?.factor ?? 1
        let dimension = array.interleaveLayout?.dimension ?? 0
        var result = [Float16](repeating: 0, count: shape.reduce(1, *))
        array.view(as: Float16.self).withUnsafePointer { ptr, _, strides in
            if factor == 1, let run = shape.last, strides[shape.count - 1] == 1 {
                result.withUnsafeMutableBufferPointer { destination in
                    for row in 0..<(shape.reduce(1, *) / run) {
                        var remaining = row; var offset = 0
                        for d in (0..<(shape.count - 1)).reversed() {
                            offset += (remaining % shape[d]) * strides[d]; remaining /= shape[d]
                        }
                        destination.baseAddress!.advanced(by: row * run).update(from: ptr.advanced(by: offset), count: run)
                    }
                }
                return
            }
            for i in result.indices {
                var remaining = i
                var offset = 0
                for d in (0..<shape.count).reversed() {
                    let coordinate = remaining % shape[d]
                    remaining /= shape[d]
                    offset += d == dimension
                        ? coordinate / factor * strides[d] + coordinate % factor
                        : coordinate * strides[d]
                }
                result[i] = ptr[offset]
            }
        }
        return result
    }
}
