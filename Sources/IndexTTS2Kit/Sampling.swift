import Foundation
import MLX
import MLXRandom

/// Autoregressive mel-token sampler: repetition penalty -> temperature -> top-k,
/// then categorical sampling. Mirrors `gpt_v2.py::_sample` (top-p is currently
/// approximated by top-k; exact RNG parity with the Python reference is not
/// possible anyway, so this favours a fast, robust implementation).
///
/// `logits`: (vocab,) for a single position. `generated` are the previously
/// emitted token ids (for the repetition penalty).
func sampleMelToken(
    _ logits0: MLXArray,
    temperature: Float,
    topK: Int,
    topP: Float,
    repetitionPenalty: Float,
    generated: [Int]
) -> Int {
    var logits = logits0.reshaped([-1]).asType(.float32)
    let vocab = logits.dim(0)

    // Repetition penalty: positive logits divided, negative multiplied.
    if repetitionPenalty != 1.0 && !generated.isEmpty {
        let uniq = Array(Set(generated)).filter { $0 >= 0 && $0 < vocab }
        if !uniq.isEmpty {
            let idx = MLXArray(uniq.map { Int32($0) })
            let g = logits[idx]
            let newg = MLX.which(g .> 0, g / repetitionPenalty, g * repetitionPenalty)
            logits[idx] = newg
        }
    }

    if temperature == 0 {
        return argMax(logits, axis: -1).item(Int.self)
    }
    logits = logits / temperature

    // Top-k filter.
    if topK > 0 && topK < vocab {
        let ascending = sorted(logits, axis: -1)
        let threshold = ascending[vocab - topK]
        logits = MLX.which(logits .>= threshold, logits, MLXArray(-Float.infinity))
    }

    let token = categorical(logits)
    return token.item(Int.self)
}
