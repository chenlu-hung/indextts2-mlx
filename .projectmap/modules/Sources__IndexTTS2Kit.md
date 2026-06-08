# Module: `Sources/IndexTTS2Kit`

## Summary
`IndexTTS2Kit` is the core Swift library implementing the full IndexTTS-2 text-to-speech pipeline on Apple MLX with no PyTorch dependency. It contains every neural network layer and model: W2VBert (semantic features), CAMPPlus (speaker style embedding), GPT/GPTv2 (autoregressive token decoder), DiT/CFM (Conditional Flow Matching mel synthesizer), BigVGANV2 (neural vocoder), RepCodec/VQ2Emb (semantic codec), plus pure-Swift preprocessing (KaldiFbank, Mel, DSP, TextTokenizer). The `Pipeline` / `IndexTTSv2` class is the main orchestrator that wires all modules together for end-to-end synthesis from text + reference audio to waveform.

<!-- projectmap:auto:start (generated — do not edit by hand) -->
## Files (30)
- `Sources/IndexTTS2Kit/Activations.swift`
- `Sources/IndexTTS2Kit/Attention.swift`
- `Sources/IndexTTS2Kit/AudioIO.swift`
- `Sources/IndexTTS2Kit/BigVGANV2.swift`
- `Sources/IndexTTS2Kit/CAMPPlus.swift`
- `Sources/IndexTTS2Kit/CFM.swift`
- `Sources/IndexTTS2Kit/ConfigV2.swift`
- `Sources/IndexTTS2Kit/Conformer.swift`
- `Sources/IndexTTS2Kit/DSP.swift`
- `Sources/IndexTTS2Kit/DiT.swift`
- `Sources/IndexTTS2Kit/GPT2.swift`
- `Sources/IndexTTS2Kit/GPTV2.swift`
- `Sources/IndexTTS2Kit/GenerateHelpers.swift`
- `Sources/IndexTTS2Kit/Helpers.swift`
- `Sources/IndexTTS2Kit/KaldiFbank.swift`
- `Sources/IndexTTS2Kit/LengthRegulator.swift`
- `Sources/IndexTTS2Kit/Loading.swift`
- `Sources/IndexTTS2Kit/Mel.swift`
- `Sources/IndexTTS2Kit/Normalize.swift`
- `Sources/IndexTTS2Kit/Perceiver.swift`
- `Sources/IndexTTS2Kit/Pipeline.swift`
- `Sources/IndexTTS2Kit/ReferenceEncoder.swift`
- `Sources/IndexTTS2Kit/RepCodec.swift`
- `Sources/IndexTTS2Kit/S2Mel.swift`
- `Sources/IndexTTS2Kit/Sampling.swift`
- `Sources/IndexTTS2Kit/TextTokenizer.swift`
- `Sources/IndexTTS2Kit/Tokenizer.swift`
- `Sources/IndexTTS2Kit/VQ2Emb.swift`
- `Sources/IndexTTS2Kit/W2VBert.swift`
- `Sources/IndexTTS2Kit/WaveNet.swift`

## Public symbols (261)
- `function normalizeWeight` — Sources/IndexTTS2Kit/Activations.swift:7
- `class WNConv1d` — Sources/IndexTTS2Kit/Activations.swift:12
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:37
- `class WNConvTranspose1d` — Sources/IndexTTS2Kit/Activations.swift:45
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:70
- `class Snake` — Sources/IndexTTS2Kit/Activations.swift:82
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:92
- `class SnakeBeta` — Sources/IndexTTS2Kit/Activations.swift:100
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:112
- `function besselI0` — Sources/IndexTTS2Kit/Activations.swift:126
- `function kaiserWindow` — Sources/IndexTTS2Kit/Activations.swift:140
- `function kaiserSincFilter1d` — Sources/IndexTTS2Kit/Activations.swift:150
- `function sinc` — Sources/IndexTTS2Kit/Activations.swift:175
- `class LowPassFilter1d` — Sources/IndexTTS2Kit/Activations.swift:187
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:202
- `class UpSample1d` — Sources/IndexTTS2Kit/Activations.swift:210
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:230
- `class DownSample1d` — Sources/IndexTTS2Kit/Activations.swift:240
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:250
- `class Activation1d` — Sources/IndexTTS2Kit/Activations.swift:253
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:266
- `protocol UnaryLayerLike` — Sources/IndexTTS2Kit/Activations.swift:275
- `function forward` — Sources/IndexTTS2Kit/Activations.swift:276
- `extension Snake` — Sources/IndexTTS2Kit/Activations.swift:278
- `extension SnakeBeta` — Sources/IndexTTS2Kit/Activations.swift:279
- `class AMPBlock1` — Sources/IndexTTS2Kit/Activations.swift:283
- `function callAsFunction` — Sources/IndexTTS2Kit/Activations.swift:306
- `class MultiHeadAttention` — Sources/IndexTTS2Kit/Attention.swift:7
- `function callAsFunction` — Sources/IndexTTS2Kit/Attention.swift:29
- `class RelPositionMultiHeadAttention` — Sources/IndexTTS2Kit/Attention.swift:52
- `function callAsFunction` — Sources/IndexTTS2Kit/Attention.swift:80
- `class RelPositionalEncoding` — Sources/IndexTTS2Kit/Attention.swift:113
- `function makePE` — Sources/IndexTTS2Kit/Attention.swift:130
- `function callAsFunction` — Sources/IndexTTS2Kit/Attention.swift:145
- `class LearnedPositionEncoding` — Sources/IndexTTS2Kit/Attention.swift:158
- `function callAsFunction` — Sources/IndexTTS2Kit/Attention.swift:166
- `enum AudioIO` — Sources/IndexTTS2Kit/AudioIO.swift:5
- `function loadAudio` — Sources/IndexTTS2Kit/AudioIO.swift:8
- `function writeWAV` — Sources/IndexTTS2Kit/AudioIO.swift:62
- `function appendStr` — Sources/IndexTTS2Kit/AudioIO.swift:70
- `function appendU32` — Sources/IndexTTS2Kit/AudioIO.swift:71
- `function appendU16` — Sources/IndexTTS2Kit/AudioIO.swift:72
- `struct BigVGANV2Config` — Sources/IndexTTS2Kit/BigVGANV2.swift:21
- `class AMPBlock1V2` — Sources/IndexTTS2Kit/BigVGANV2.swift:38
- `function callAsFunction` — Sources/IndexTTS2Kit/BigVGANV2.swift:64
- `function getPadding` — Sources/IndexTTS2Kit/BigVGANV2.swift:75
- `class BigVGANV2` — Sources/IndexTTS2Kit/BigVGANV2.swift:79
- `function callAsFunction` — Sources/IndexTTS2Kit/BigVGANV2.swift:137
- `class CampBatchNorm2d` — Sources/IndexTTS2Kit/CAMPPlus.swift:18
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:32
- `class CampBatchNorm1d` — Sources/IndexTTS2Kit/CAMPPlus.swift:41
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:58
- `class BNReLU` — Sources/IndexTTS2Kit/CAMPPlus.swift:67
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:70
- `class NCLConv1d` — Sources/IndexTTS2Kit/CAMPPlus.swift:77
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:83
- `class CampResBlock` — Sources/IndexTTS2Kit/CAMPPlus.swift:90
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:117
- `class CampFCM` — Sources/IndexTTS2Kit/CAMPPlus.swift:129
- `function callAsFunction` — Sources/IndexTTS2Kit/CAMPPlus.swift:154
- …and 201 more

## Dependencies (imports)
- `AVFoundation`
- `Foundation`
- `MLX`
- `MLXFFT`
- `MLXFast`
- `MLXNN`
- `MLXRandom`
<!-- projectmap:auto:end -->
