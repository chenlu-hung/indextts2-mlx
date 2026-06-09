# Module: `scripts`

## Summary
The `scripts/` directory contains Python utilities for one-time model preparation and numerical verification. The `convert_*.py` scripts load PyTorch `.bin` checkpoints (CAMPPlus, W2VBert, semantic codec) and re-save them as `.safetensors`, the format MLX reads natively; `torch_bin_to_safetensors.py` is a torch-free pickle reader used by the converters. The `ref_*.py` scripts are pure-NumPy reference implementations of each preprocessing stage used to validate that the Swift port produces bit-accurate results.

<!-- projectmap:auto:start (generated — do not edit by hand) -->
## Files (7)
- `scripts/convert_campplus.py`
- `scripts/convert_semantic_codec.py`
- `scripts/convert_w2vbert.py`
- `scripts/ref_campplus.py`
- `scripts/ref_repcodec.py`
- `scripts/ref_w2vbert.py`
- `scripts/torch_bin_to_safetensors.py`

## Public symbols (63)
- `namespace np` — scripts/convert_campplus.py:13
- `function main` — scripts/convert_campplus.py:17
- `namespace np` — scripts/convert_semantic_codec.py:12
- `function main` — scripts/convert_semantic_codec.py:16
- `namespace np` — scripts/convert_w2vbert.py:12
- `function main` — scripts/convert_w2vbert.py:16
- `namespace np` — scripts/ref_campplus.py:9
- `function mel_scale` — scripts/ref_campplus.py:14
- `function inverse_mel_scale` — scripts/ref_campplus.py:18
- `function povey_window` — scripts/ref_campplus.py:22
- `function get_mel_banks` — scripts/ref_campplus.py:28
- `function kaldi_fbank` — scripts/ref_campplus.py:48
- `function conv1d` — scripts/ref_campplus.py:86
- `function conv2d` — scripts/ref_campplus.py:105
- `function bn` — scripts/ref_campplus.py:126
- `function relu` — scripts/ref_campplus.py:139
- `function basic_resblock` — scripts/ref_campplus.py:144
- `function fcm` — scripts/ref_campplus.py:155
- `function bn_relu` — scripts/ref_campplus.py:169
- `function cam_layer` — scripts/ref_campplus.py:173
- `function seg_pooling` — scripts/ref_campplus.py:182
- `function dense_tdnn_layer` — scripts/ref_campplus.py:194
- `function dense_block` — scripts/ref_campplus.py:202
- `function transit` — scripts/ref_campplus.py:209
- `function xvector` — scripts/ref_campplus.py:215
- `function campplus` — scripts/ref_campplus.py:237
- `function synth_wave` — scripts/ref_campplus.py:245
- `function main` — scripts/ref_campplus.py:251
- `namespace np` — scripts/ref_repcodec.py:11
- `function fold_wn` — scripts/ref_repcodec.py:15
- `function conv1d_nlc` — scripts/ref_repcodec.py:26
- `function layernorm` — scripts/ref_repcodec.py:47
- `function gelu` — scripts/ref_repcodec.py:53
- `function convnext` — scripts/ref_repcodec.py:59
- `function vocos_backbone` — scripts/ref_repcodec.py:70
- `function normalize` — scripts/ref_repcodec.py:80
- `function quantize` — scripts/ref_repcodec.py:85
- `function main` — scripts/ref_repcodec.py:104
- `namespace np` — scripts/ref_w2vbert.py:11
- `function mel_scale` — scripts/ref_w2vbert.py:22
- `function povey` — scripts/ref_w2vbert.py:26
- `function mel_banks` — scripts/ref_w2vbert.py:31
- `function fbank80` — scripts/ref_w2vbert.py:47
- `function extract_features` — scripts/ref_w2vbert.py:65
- `function layernorm` — scripts/ref_w2vbert.py:77
- `function lin` — scripts/ref_w2vbert.py:83
- `function swish` — scripts/ref_w2vbert.py:87
- `function gelu` — scripts/ref_w2vbert.py:91
- `function ffn` — scripts/ref_w2vbert.py:95
- `function attention` — scripts/ref_w2vbert.py:102
- `function conv_module` — scripts/ref_w2vbert.py:120
- `function encoder_layer` — scripts/ref_w2vbert.py:141
- `function model` — scripts/ref_w2vbert.py:161
- `function synth_wave` — scripts/ref_w2vbert.py:169
- `function main` — scripts/ref_w2vbert.py:175
- `namespace np` — scripts/torch_bin_to_safetensors.py:12
- `class _Storage` — scripts/torch_bin_to_safetensors.py:29
- `function load_state_dict` — scripts/torch_bin_to_safetensors.py:36
- `function load_storage_bytes` — scripts/torch_bin_to_safetensors.py:43
- `class Unpickler` — scripts/torch_bin_to_safetensors.py:46
- …and 3 more

## Dependencies (imports)
- `collections`
- `io`
- `math`
- `numpy`
- `os`
- `pickle`
- `re`
- `safetensors`
- `scipy`
- `struct`
- `sys`
- `zipfile`
<!-- projectmap:auto:end -->
