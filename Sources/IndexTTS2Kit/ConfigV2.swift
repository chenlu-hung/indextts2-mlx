import Foundation

/// Minimal config for IndexTTS-2. Only the handful of values that actually vary
/// between checkpoints are read from `config.json`; the rest of the architecture
/// (S2Mel DiT, BigVGAN v2, conformer/perceiver dims) is fixed for this model
/// family and matches the hard-coded constructors in the MLX-Python reference.
public struct IndexTTS2Config: Codable {
    public struct GPT: Codable {
        public var model_dim: Int = 1280
        public var heads: Int = 20
        public var layers: Int = 24
        public var max_mel_tokens: Int = 1815
        public var max_text_tokens: Int = 600
        public var number_text_tokens: Int = 12000
        public var number_mel_codes: Int = 8194
        public var start_mel_token: Int = 8192
        public var stop_mel_token: Int = 8193
        public var start_text_token: Int = 0
        public var stop_text_token: Int = 1
        public var mel_length_compression: Int = 1024
        public var condition_num_latent: Int = 32
    }

    public var gpt: GPT = GPT()
    public var version: Double? = 2.0
    public var sample_rate: Int? = 22050
    /// Set when the GPT backbone was pre-quantized (`nn.quantize` group_size 64).
    public var quantize_bits: Int? = nil

    public var sampleRate: Int { sample_rate ?? 22050 }

    public static func load(from url: URL) throws -> IndexTTS2Config {
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        // Tolerate the many extra keys in config.json (bigvgan/mel/s2mel sections).
        return try dec.decode(IndexTTS2Config.self, from: data)
    }
}
