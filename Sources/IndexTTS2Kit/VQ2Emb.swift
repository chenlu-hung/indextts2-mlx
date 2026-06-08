import Foundation
import MLX

/// `vq2emb`: converts GPT mel codes (semantic tokens) to 1024-d embeddings.
///
/// This is the inference-only replacement for the PyTorch
/// `semantic_codec.quantizer.vq2emb`: a codebook lookup (8192 → 8) followed by a
/// kernel-size-1 Conv1d (8 → 1024). Weights live in `vq2emb.safetensors`:
///   * `codebook.weight`     (8192, 8)
///   * `out_project.weight`  (1024, 8, 1)   PyTorch Conv1d OIK
///   * `out_project.bias`    (1024,)
///
/// Mirrors `generate_v2.py::_vq2emb_forward`.
public struct VQ2Emb {
    let codebook: MLXArray      // (8192, 8)
    let weight2d: MLXArray      // (1024, 8)
    let bias: MLXArray          // (1024,)

    public init(codebook: MLXArray, outProjectWeight: MLXArray, outProjectBias: MLXArray) {
        self.codebook = codebook
        self.weight2d = outProjectWeight.squeezed(axis: -1)  // (1024, 8, 1) -> (1024, 8)
        self.bias = outProjectBias
    }

    /// Load from a `vq2emb.safetensors` file.
    public static func load(from url: URL) throws -> VQ2Emb {
        let w = try MLX.loadArrays(url: url)
        guard let cb = w["codebook.weight"],
              let opw = w["out_project.weight"],
              let opb = w["out_project.bias"]
        else {
            throw IndexTTS2Error.missingWeights("vq2emb: codebook.weight / out_project.{weight,bias}")
        }
        return VQ2Emb(codebook: cb, outProjectWeight: opw, outProjectBias: opb)
    }

    /// `codes`: (batch, length) Int32 mel codes. Returns (batch, 1024, length).
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let emb = codebook[codes]                  // (B, T, 8)
        let out = matmul(emb, weight2d.T) + bias   // (B, T, 1024)
        return out.transposed(0, 2, 1)             // (B, 1024, T)
    }
}
