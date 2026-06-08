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
