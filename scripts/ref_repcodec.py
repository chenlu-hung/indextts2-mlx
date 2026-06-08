#!/usr/bin/env python3
"""Torch-free numpy reference for RepCodec semantic_codec `quantize()`.
Loads semantic_codec.safetensors (torch layout). Generates a fixed random input,
computes S_ref (B,T,1024) + indices, and dumps input/sref/indices for the Swift
parity test.

Usage: uv run --with numpy --with safetensors python ref_repcodec.py semantic_codec.safetensors out_dir
"""
import os
import sys
import numpy as np
from safetensors.numpy import load_file


def fold_wn(sd):
    for k in list(sd):
        if k.endswith("_g"):
            base = k[:-2]
            wv = sd[base + "_v"].astype(np.float64)
            wg = sd[base + "_g"].astype(np.float64)
            norm = np.sqrt((wv ** 2).sum(axis=tuple(range(1, wv.ndim)), keepdims=True))
            sd[base] = wg * (wv / (norm + 1e-8))
    return sd


def conv1d_nlc(x, w, b=None, pad=0, groups=1):
    # x: (B,T,Cin), w torch layout (Cout,Cin/groups,K)
    B, T, Cin = x.shape
    Cout, Cing, K = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    Tp = x.shape[1]
    Tout = Tp - K + 1
    out = np.zeros((B, Tout, Cout), np.float64)
    if groups == 1:
        cols = np.stack([x[:, k:k + Tout, :] for k in range(K)], axis=2)  # (B,Tout,K,Cin)
        out = np.einsum("btkc,ock->bto", cols, w)
    else:
        # depthwise (groups==Cin==Cout, Cing==1)
        for k in range(K):
            out += x[:, k:k + Tout, :] * w[:, 0, k][None, None, :]
    if b is not None:
        out += b[None, None, :]
    return out


def layernorm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def convnext(x, sd, p):
    res = x
    x = conv1d_nlc(x, sd[p + ".dwconv.weight"], sd[p + ".dwconv.bias"], pad=3, groups=x.shape[-1])
    x = layernorm(x, sd[p + ".norm.weight"], sd[p + ".norm.bias"])
    x = x @ sd[p + ".pwconv1.weight"].T + sd[p + ".pwconv1.bias"]
    x = gelu(x)
    x = x @ sd[p + ".pwconv2.weight"].T + sd[p + ".pwconv2.bias"]
    x = sd[p + ".gamma"] * x
    return res + x


def vocos_backbone(x, sd, p):
    # x: (B,T,1024) NLC
    x = conv1d_nlc(x, sd[p + ".embed.weight"], sd[p + ".embed.bias"], pad=3)
    x = layernorm(x, sd[p + ".norm.weight"], sd[p + ".norm.bias"])
    for i in range(12):
        x = convnext(x, sd, f"{p}.convnext.{i}")
    x = layernorm(x, sd[p + ".final_layer_norm.weight"], sd[p + ".final_layer_norm.bias"])
    return x


def normalize(x, eps=1e-12):
    n = np.sqrt((x ** 2).sum(-1, keepdims=True))
    return x / np.maximum(n, eps)


def quantize(x, sd):
    # x: (B,T,1024)
    z = vocos_backbone(x, sd, "encoder.0")
    z = z @ sd["encoder.1.weight"].T + sd["encoder.1.bias"]   # (B,T,1024)
    # in_project conv1x1
    z_e = conv1d_nlc(z, sd["quantizer.quantizers.0.in_project.weight"],
                     sd["quantizer.quantizers.0.in_project.bias"])  # (B,T,8)
    B, T, D = z_e.shape
    enc = normalize(z_e.reshape(B * T, D))
    cb = sd["quantizer.quantizers.0.codebook.weight"].astype(np.float64)  # (8192,8)
    cbn = normalize(cb)
    sims = enc @ cbn.T
    idx = np.argmax(sims, axis=1).reshape(B, T)
    z_q = cb[idx]  # (B,T,8) raw codebook
    s_ref = conv1d_nlc(z_q, sd["quantizer.quantizers.0.out_project.weight"],
                       sd["quantizer.quantizers.0.out_project.bias"])  # (B,T,1024)
    return s_ref, idx


def main():
    sd_path, out_dir = sys.argv[1], sys.argv[2]
    sd = fold_wn({k: v.astype(np.float64) for k, v in load_file(sd_path).items()})
    T = 50
    rng = np.random.RandomState(0)
    x = rng.randn(1, T, 1024).astype(np.float32)
    s_ref, idx = quantize(x.astype(np.float64), sd)
    os.makedirs(out_dir, exist_ok=True)
    x.astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))
    s_ref.astype(np.float32).tofile(os.path.join(out_dir, "sref.bin"))
    idx.astype(np.int32).tofile(os.path.join(out_dir, "indices.bin"))
    print("S_ref:", s_ref.shape, "mean=%.5f min=%.5f max=%.5f" % (s_ref.mean(), s_ref.min(), s_ref.max()))
    print("indices[:10]:", idx.reshape(-1)[:10].tolist())


if __name__ == "__main__":
    main()
