#!/usr/bin/env python3
"""Torch-free numpy reference for the W2V-BERT 2.0 speaker-feature pipeline:
SeamlessM4T 80-mel + stride-2 stacking -> feature_projection -> 17 conformer
layers -> hidden_states[17] -> normalize by stats -> spk_cond_emb.

Usage: uv run --with numpy --with safetensors --with scipy python ref_w2vbert.py \
           w2v-bert.safetensors w2vbert_stats.safetensors out_dir
"""
import os
import sys
import numpy as np
from safetensors.numpy import load_file
from scipy.special import erf

NUM_LAYERS = 17
HEADS = 16
HEAD = 64
LEFT, RIGHT = 64, 8


# --- fbank (kaldi/seamless: scale 2^15, povey window, kaldi mel, log) ---
def mel_scale(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def povey(n):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return np.power(w, 0.85)


def mel_banks(num_bins=80, padded=512, sr=16000, low=20.0, high=8000.0):
    nfft = padded // 2
    width = sr / padded
    ml, mh = mel_scale(low), mel_scale(high)
    delta = (mh - ml) / (num_bins + 1)
    b = np.arange(num_bins).reshape(-1, 1)
    left = ml + b * delta
    center = ml + (b + 1) * delta
    right = ml + (b + 2) * delta
    mel = mel_scale(width * np.arange(nfft)).reshape(1, -1)
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    fb = np.maximum(0.0, np.minimum(up, down))
    return np.pad(fb, ((0, 0), (0, 1)))  # (80, 257)


def fbank80(wave, floor=1.192092955078125e-07):
    wave = wave.astype(np.float64) * (2 ** 15)
    win_size, shift, padded = 400, 160, 512
    n = wave.shape[0]
    m = 1 + (n - win_size) // shift
    idx = np.arange(win_size)[None, :] + shift * np.arange(m)[:, None]
    fr = wave[idx]
    fr = fr - fr.mean(1, keepdims=True)
    pre = fr.copy()
    pre[:, 1:] -= 0.97 * fr[:, :-1]
    pre[:, 0] *= (1 - 0.97)
    pre = pre * povey(win_size)[None, :]
    pre = np.pad(pre, ((0, 0), (0, padded - win_size)))
    spec = np.abs(np.fft.rfft(pre, n=padded, axis=1)) ** 2
    mel = spec @ mel_banks().T
    return np.log(np.maximum(mel, floor)).astype(np.float32)  # (m,80)


def extract_features(wave, stride=2):
    f = fbank80(wave)                      # (m,80)
    f = (f - f.mean(0, keepdims=True)) / np.sqrt(f.var(0, ddof=1, keepdims=True) + 1e-7)
    m = f.shape[0]
    rem = m % stride
    if rem:
        f = f[: m - rem]
    f = f.reshape(m // stride, 80 * stride)  # (m//2, 160)
    return f[None]                            # (1, T, 160)


# --- model ---
def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def lin(x, sd, p):
    return x @ sd[p + ".weight"].T + sd[p + ".bias"]


def swish(x):
    return x / (1.0 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def ffn(x, sd, p):
    x = lin(x, sd, p + ".intermediate_dense")
    x = swish(x)
    x = lin(x, sd, p + ".output_dense")
    return x


def attention(x, sd, p):
    B, T, H = x.shape
    q = (x @ sd[p + ".linear_q.weight"].T + sd[p + ".linear_q.bias"]).reshape(B, T, HEADS, HEAD).transpose(0, 2, 1, 3)
    k = (x @ sd[p + ".linear_k.weight"].T + sd[p + ".linear_k.bias"]).reshape(B, T, HEADS, HEAD).transpose(0, 2, 1, 3)
    v = (x @ sd[p + ".linear_v.weight"].T + sd[p + ".linear_v.bias"]).reshape(B, T, HEADS, HEAD).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HEAD)
    pl = np.arange(T)[:, None]
    pr = np.arange(T)[None, :]
    dist = np.clip(pr - pl, -LEFT, RIGHT) + LEFT
    pos = sd[p + ".distance_embedding.weight"][dist]   # (T,T,HEAD)
    rel = np.einsum("bhld,lrd->bhlr", q, pos)
    scores = scores + rel / np.sqrt(HEAD)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs = probs / probs.sum(-1, keepdims=True)
    out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, HEADS * HEAD)
    return out @ sd[p + ".linear_out.weight"].T + sd[p + ".linear_out.bias"]


def conv_module(x, sd, p):
    x = layernorm(x, sd[p + ".layer_norm.weight"], sd[p + ".layer_norm.bias"])
    # pointwise_conv1 (2048,1024,1) -> linear
    w1 = sd[p + ".pointwise_conv1.weight"][:, :, 0]      # (2048,1024)
    h = x @ w1.T                                          # (B,T,2048)
    a, b = h[..., :1024], h[..., 1024:]
    h = a / (1.0 + np.exp(-b))                            # glu
    # causal pad left 30, depthwise k31
    B, T, C = h.shape
    hp = np.pad(h, ((0, 0), (30, 0), (0, 0)))
    dw = sd[p + ".depthwise_conv.weight"][:, 0, :]        # (1024,31)
    out = np.zeros((B, T, C))
    for kk in range(31):
        out += hp[:, kk:kk + T, :] * dw[:, kk][None, None, :]
    h = layernorm(out, sd[p + ".depthwise_layer_norm.weight"], sd[p + ".depthwise_layer_norm.bias"])
    h = swish(h)
    w2 = sd[p + ".pointwise_conv2.weight"][:, :, 0]       # (1024,1024)
    h = h @ w2.T
    return h


def encoder_layer(x, sd, p):
    res = x
    x = layernorm(x, sd[p + ".ffn1_layer_norm.weight"], sd[p + ".ffn1_layer_norm.bias"])
    x = ffn(x, sd, p + ".ffn1")
    x = x * 0.5 + res
    res = x
    x = layernorm(x, sd[p + ".self_attn_layer_norm.weight"], sd[p + ".self_attn_layer_norm.bias"])
    x = attention(x, sd, p + ".self_attn")
    x = x + res
    res = x
    x = conv_module(x, sd, p + ".conv_module")
    x = res + x
    res = x
    x = layernorm(x, sd[p + ".ffn2_layer_norm.weight"], sd[p + ".ffn2_layer_norm.bias"])
    x = ffn(x, sd, p + ".ffn2")
    x = x * 0.5 + res
    x = layernorm(x, sd[p + ".final_layer_norm.weight"], sd[p + ".final_layer_norm.bias"])
    return x


def model(feat, sd):
    x = layernorm(feat, sd["feature_projection.layer_norm.weight"], sd["feature_projection.layer_norm.bias"])
    x = lin(x, sd, "feature_projection.projection")
    for i in range(NUM_LAYERS):
        x = encoder_layer(x, sd, f"encoder.layers.{i}")
    return x


def synth_wave(n=16000):
    t = np.arange(n) / 16000.0
    return (0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)


def main():
    w_path, stat_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    sd = {k: v.astype(np.float64) for k, v in load_file(w_path).items()
          if k.startswith("feature_projection") or k.startswith("encoder.layers.")}
    stats = load_file(stat_path)
    mean = stats["mean"].astype(np.float64)
    std = np.sqrt(stats["var"].astype(np.float64))

    if len(sys.argv) > 4 and sys.argv[4] == "noise":
        rng = np.random.RandomState(1)
        wave = (0.1 * rng.randn(16000)).astype(np.float32)
    else:
        wave = synth_wave(16000)
    os.makedirs(out_dir, exist_ok=True)
    wave.astype(np.float32).tofile(os.path.join(out_dir, "wave.bin"))
    feat = extract_features(wave)
    h17 = model(feat, sd)
    spk = (h17 - mean) / std
    os.makedirs(out_dir, exist_ok=True)
    feat.astype(np.float32).tofile(os.path.join(out_dir, "feat160.bin"))
    spk.astype(np.float32).tofile(os.path.join(out_dir, "spk.bin"))
    print("feat160:", feat.shape, "mean=%.5f std=%.5f" % (feat.mean(), feat.std()))
    print("h17:", h17.shape, "mean=%.5f min=%.5f max=%.5f" % (h17.mean(), h17.min(), h17.max()))
    print("spk:", spk.shape, "mean=%.5f min=%.5f max=%.5f" % (spk.mean(), spk.min(), spk.max()))


if __name__ == "__main__":
    main()
