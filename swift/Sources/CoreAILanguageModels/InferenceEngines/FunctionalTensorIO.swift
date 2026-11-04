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
