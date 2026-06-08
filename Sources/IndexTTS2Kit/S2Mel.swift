import Foundation
import MLX
import MLXNN

/// MLP projecting the GPT latent (1280) to semantic content (1024).
/// `layers`: Linear(1280→256) → Linear(256→128) → Linear(128→1024).
final class GPTLayer: Module {
    let layers: [Linear]
    override init() {
        self.layers = [Linear(1280, 256), Linear(256, 128), Linear(128, 1024)]
        super.init()
    }
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for layer in layers { x = layer(x) }
        return x
    }
}

/// Semantic-to-Mel: GPT-latent projection → length regulator → CFM diffusion.
/// Port of models/s2mel/s2mel.py. Loads `s2mel.safetensors`.
public final class S2Mel: Module {
    let gpt_layer: GPTLayer
    let length_regulator: InterpolateRegulator
    let cfm: CFM

    public override init() {
        self.gpt_layer = GPTLayer()
        self.length_regulator = InterpolateRegulator()
        self.cfm = CFM()
        super.init()
    }

    public static func fromPretrained(directory: URL, verbose: Bool = false) throws -> S2Mel {
        let model = S2Mel()
        try loadWeights(
            into: model, from: directory.appendingPathComponent("s2mel.safetensors"),
            label: "s2mel", verbose: verbose)
        return model
    }
}
