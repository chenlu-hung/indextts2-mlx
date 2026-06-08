# IndexTTS-2 — MLX-Swift (torch-free)

A native Swift / [MLX](https://github.com/ml-explore/mlx-swift) port of
**IndexTTS-2** — zero-shot voice cloning with emotion and duration control —
running entirely on Apple Silicon with **no PyTorch dependency** (neither at
runtime nor for weight conversion).

The whole pipeline is ported, including the preprocessing models that the
upstream MLX port still ran through PyTorch (W2V-BERT 2.0, RepCodec semantic
codec, CAMPPlus). Weight conversion is done torch-free in numpy
(`scripts/`).

## Pipeline

```
reference.wav ─┬─ 16k ─ W2V-BERT 2.0 ── hidden_states[17] ─ norm ─→ spk_cond_emb
               │                                       └─ RepCodec ─→ S_ref ─→ length_reg ─→ prompt_condition
               ├─ 16k ─ kaldi fbank ── CAMPPlus ───────────────────→ style
               └─ 22k ─ mel ───────────────────────────────────────→ ref_mel

text ─ tokenizer ─→ GPT v2 (UnifiedVoiceV2)  [+ spk_cond_emb, emotion]  ─→ semantic mel-codes
        ─→ vq2emb + GPT latent ─→ S2Mel (gpt_layer → length_reg → CFM/DiT + CFG) ─→ 80-mel
        ─→ BigVGAN v2 ─→ 22.05 kHz waveform
```

## Build

Requires Xcode (MLX needs the Metal toolchain — plain `swift build` cannot
compile the kernels).

```bash
./build.sh Debug        # -> .build/xcode/Build/Products/Debug/indextts2
```

## Models

The IndexTTS-2 model dir (`models/mlx-indextts2-standard-8bit`, HF
`vanch007/mlx-indextts2-standard-8bit`) holds the generation stack (GPT v2 8-bit,
S2Mel, BigVGAN, vq2emb, tokenizer, `wav2vec2bert_stats.pt`).

The preprocessing weights come from other HF repos and are converted to MLX
layout (torch-free) into `models/preprocessing/`:

```bash
# CAMPPlus  (funasr/campplus, ~27 MB)
curl -L https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin \
     -o models/preprocessing/campplus_cn_common.bin
uv run --with numpy --with safetensors python scripts/torch_bin_to_safetensors.py \
     models/preprocessing/campplus_cn_common.bin models/preprocessing/campplus.safetensors
uv run --with numpy --with safetensors python scripts/convert_campplus.py \
     models/preprocessing/campplus.safetensors models/preprocessing/campplus_mlx.safetensors

# RepCodec semantic codec  (amphion/MaskGCT, ~169 MB)
curl -L https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors \
     -o models/preprocessing/semantic_codec.safetensors
uv run --with numpy --with safetensors python scripts/convert_semantic_codec.py \
     models/preprocessing/semantic_codec.safetensors models/preprocessing/semantic_codec_mlx.safetensors

# W2V-BERT 2.0  (facebook/w2v-bert-2.0, ~2.3 GB; only 17/24 layers are kept)
curl -L https://huggingface.co/facebook/w2v-bert-2.0/resolve/main/model.safetensors \
     -o models/preprocessing/w2v-bert.safetensors
uv run --with numpy --with safetensors python scripts/convert_w2vbert.py \
     models/preprocessing/w2v-bert.safetensors models/preprocessing/w2vbert_mlx.safetensors

# W2V-BERT normalization stats (from the IndexTTS-2 model dir)
uv run --with numpy --with safetensors python scripts/torch_bin_to_safetensors.py \
     models/mlx-indextts2-standard-8bit/wav2vec2bert_stats.pt \
     models/preprocessing/w2vbert_stats.safetensors
```

After conversion the runtime needs only the `*_mlx.safetensors` +
`w2vbert_stats.safetensors` files; the raw torch-layout downloads can be deleted.

## Synthesis

```bash
./.build/xcode/Build/Products/Debug/indextts2 \
    --model models/mlx-indextts2-standard-8bit \
    --ref  speaker.wav \
    --text "Hello, this is index TTS speaking." \
    --out  out.wav
```

Options: `--emo-ref <wav>` (separate emotion reference), `--steps 25`
(diffusion), `--cfg 0.7`, `--seed N`, `--temperature 0.8`, `--top-p 0.8`,
`--top-k 30`, `--speed 1.0`, `--max-mel-tokens 1500`, `--preproc-dir <dir>`.

## Verification (torch-free)

Each preprocessing model was verified against a pure-numpy reference reading the
same weights (no torch). Run e.g.:

```bash
uv run --with numpy --with safetensors --with scipy python scripts/ref_campplus.py \
    models/preprocessing/campplus.safetensors /tmp/campref
./.build/xcode/Build/Products/Debug/indextts2 --campplus-test --out /tmp/campref
```

Measured agreement (Swift vs numpy float64 reference):

| Component | metric |
|---|---|
| kaldi fbank | max abs 7e-4 |
| CAMPPlus `style` | max abs 4e-4 |
| RepCodec `S_ref` | max abs 1e-6, indices exact |
| W2V-BERT `spk_cond_emb` (broadband) | corr 1.0000, max abs 2e-3 |

(Diagnostic CLI modes: `--smoke`, `--gen-smoke`, `--mel-dump`,
`--campplus-test`, `--repcodec-test`, `--w2vbert-test`.)

## Not yet ported

- `emo_matrix` (feat2.pt) 8-emotion-weight control — the reference-audio and
  separate-emotion-reference paths work; explicit emotion-weight vectors do not.
- SRT-timed batch synthesis.
