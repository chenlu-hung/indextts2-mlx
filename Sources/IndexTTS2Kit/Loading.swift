import Foundation
import MLX
import MLXNN

public enum IndexTTS2Error: Error, CustomStringConvertible {
    case missingWeights(String)
    case missingFile(String)
    case badConfig(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "missing weights: \(s)"
        case .missingFile(let s): return "missing file: \(s)"
        case .badConfig(let s): return "bad config: \(s)"
        }
    }
}

/// Load a `.safetensors` file into `module`, then report any checkpoint keys that
/// went unused or module parameters left unfilled. `allowMissingSuffixes` lists
/// parameter-name suffixes that are *expected* to be missing (computed buffers
/// such as positional-encoding / RoPE tables that are not stored in the
/// checkpoint).
@discardableResult
public func loadWeights(
    into module: Module,
    from url: URL,
    allowMissingSuffixes: [String] = [
        "pos_enc.pe", "freqs_cis", "rope.freqs_cis", "freqs",
        "upsample.filter", "lowpass.filter",  // kaiser anti-alias buffers (computed at init)
    ],
    label: String = "",
    verbose: Bool = false
) throws -> [String: MLXArray] {
    let weights = try loadArrays(url: url)
    let params = ModuleParameters.unflattened(weights)
    module.update(parameters: params)
    eval(module)

    let modelKeys = Set(module.parameters().flattened().map { $0.0 })
    let weightKeys = Set(weights.keys)
    let missing = modelKeys.subtracting(weightKeys).sorted()
    let unused = weightKeys.subtracting(modelKeys).sorted()
    let unexpectedMissing = missing.filter { key in
        !allowMissingSuffixes.contains { key.hasSuffix($0) }
    }
    let tag = label.isEmpty ? "" : "[\(label)] "
    if !unexpectedMissing.isEmpty {
        FileHandle.standardError.write(
            Data("⚠️  \(tag)missing weights (\(unexpectedMissing.count)): \(unexpectedMissing.prefix(20))\n".utf8))
    }
    if !unused.isEmpty {
        FileHandle.standardError.write(
            Data("⚠️  \(tag)unused checkpoint keys (\(unused.count)): \(unused.prefix(20))\n".utf8))
    }
    if verbose && unexpectedMissing.isEmpty && unused.isEmpty {
        FileHandle.standardError.write(Data("✓ \(tag)all \(weightKeys.count) keys mapped\n".utf8))
    }
    return weights
}

/// Cast every float32 parameter/buffer of `module` (recursively) to `dtype` and
/// re-materialize. Used to run inference-only modules (vocoder, DiT) in lower
/// precision (bf16) for ~2× memory-bandwidth throughput. Non-float parameters are
/// left untouched. bf16 is preferred over fp16 here: it keeps fp32's exponent
/// range, so SnakeBeta's `1/exp(beta)` and RMSNorm's sum-of-squares don't
/// overflow/underflow the way they do in fp16.
public func castParameters(_ module: Module, to dtype: DType) {
    var d = [String: MLXArray]()
    for (k, v) in module.parameters().flattened() {
        d[k] = v.dtype == .float32 ? v.asType(dtype) : v
    }
    module.update(parameters: ModuleParameters.unflattened(d))
    eval(module)
}
