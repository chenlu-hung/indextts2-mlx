import Foundation
import MLX

/// Reverse-index based reflect padding along a single axis.
///
/// Mirrors the manual reflect pad used in the Python reference (slicing +
/// reversal), since MLX's `padded` only supports `.constant` / `.edge`.
///
/// For pad `p` along `axis` of length `T`:
///   prefix = x[1 ... p] reversed   (indices p, p-1, ..., 1)
///   suffix = x[T-p-1 ... T-2] reversed (indices T-2, T-3, ..., T-p-1)
func reflectPad(_ x: MLXArray, axis: Int, pad: Int) -> MLXArray {
    if pad == 0 { return x }
    let t = x.dim(axis)
    let prefixIdx = MLXArray((0 ..< pad).map { Int32(pad - $0) })          // p, p-1, ..., 1
    let suffixIdx = MLXArray((0 ..< pad).map { Int32(t - 2 - $0) })        // T-2, ..., T-p-1
    let prefix = x.take(prefixIdx, axis: axis)
    let suffix = x.take(suffixIdx, axis: axis)
    return concatenated([prefix, x, suffix], axis: axis)
}

/// Additive causal mask of shape `[n, n]`: 0 on/below diagonal, -inf above.
/// Broadcasts over (batch, heads, n, n) inside scaled-dot-product attention.
func causalMask(_ n: Int, dtype: DType = .float32) -> MLXArray {
    let rinds = MLXArray(Array(0 ..< Int32(n)))
    let linds = rinds.reshaped([n, 1])
    let allow = linds .>= rinds.reshaped([1, n])          // bool [n, n]
    let zeros = MLXArray.zeros([n, n], dtype: dtype)
    let neg = MLXArray(-Float.infinity).asType(dtype)
    return MLX.which(allow, zeros, neg)
}
