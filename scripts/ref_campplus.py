#!/usr/bin/env python3
"""Torch-free numpy reference for kaldi-fbank + CAMPPlus, used to verify the
Swift port. Loads campplus.safetensors (torch layout). Produces `style` (1,192)
and dumps fbank + style as raw float32 for Swift to diff against.

Usage: uv run --with numpy --with safetensors python ref_campplus.py campplus.safetensors out_dir
"""
import sys
import numpy as np
from safetensors.numpy import load_file


# ----------------------------- kaldi fbank -----------------------------------
def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def povey_window(n):
    # hann periodic=False, raised to 0.85
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    return np.power(w, 0.85).astype(np.float64)


def get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq):
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low = mel_scale(low_freq)
    mel_high = mel_scale(high_freq)
    mel_freq_delta = (mel_high - mel_low) / (num_bins + 1)
    bin_idx = np.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low + bin_idx * mel_freq_delta
    center_mel = mel_low + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low + (bin_idx + 2.0) * mel_freq_delta
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins)).reshape(1, -1)
    up = (mel - left_mel) / (center_mel - left_mel)
    down = (right_mel - mel) / (right_mel - center_mel)
    bins = np.maximum(0.0, np.minimum(up, down))  # (num_bins, num_fft_bins)
    return bins


def kaldi_fbank(wave, sr=16000, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                preemph=0.97, low_freq=20.0, high_freq=0.0):
    wave = wave.astype(np.float64)
    win_shift = int(sr * frame_shift * 0.001)   # 160
    win_size = int(sr * frame_length * 0.001)    # 400
    padded = 1
    while padded < win_size:
        padded <<= 1                              # 512
    n = wave.shape[0]
    if n < win_size:
        return np.zeros((0, num_mel_bins), np.float32)
    m = 1 + (n - win_size) // win_shift
    # frame
    idx = np.arange(win_size)[None, :] + win_shift * np.arange(m)[:, None]
    frames = wave[idx]  # (m, win_size)
    # remove dc offset (row mean)
    frames = frames - frames.mean(axis=1, keepdims=True)
    # preemphasis (per frame, max(0,j-1))
    pre = frames.copy()
    pre[:, 1:] -= preemph * frames[:, :-1]
    pre[:, 0] -= preemph * frames[:, 0]
    # window
    win = povey_window(win_size)
    pre = pre * win[None, :]
    # pad to padded window size
    if padded > win_size:
        pre = np.pad(pre, ((0, 0), (0, padded - win_size)))
    # power spectrum
    spec = np.abs(np.fft.rfft(pre, n=padded, axis=1)) ** 2   # (m, padded//2+1)
    mb = get_mel_banks(num_mel_bins, padded, sr, low_freq, high_freq)  # (80, 256)
    mb = np.pad(mb, ((0, 0), (0, 1)))  # (80, 257)
    eps = np.finfo(np.float32).eps
    mel_e = spec @ mb.T  # (m, 80)
    mel_e = np.log(np.maximum(mel_e, eps))
    return mel_e.astype(np.float32)


# ----------------------------- conv helpers ----------------------------------
def conv1d(x, w, b=None, stride=1, padding=0, dilation=1):
    # x: (Cin, L), w: (Cout, Cin, K)
    Cin, L = x.shape
    Cout, _, K = w.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding)))
    Lp = x.shape[1]
    Lout = (Lp - dilation * (K - 1) - 1) // stride + 1
    # im2col: (Cin, K, Lout)
    cols = np.empty((Cin, K, Lout), np.float64)
    for k in range(K):
        start = k * dilation
        cols[:, k, :] = x[:, start:start + stride * Lout:stride]
    out = np.einsum("ock,ckl->ol", w.astype(np.float64), cols)
    if b is not None:
        out += b.astype(np.float64)[:, None]
    return out


def conv2d(x, w, b=None, stride=(1, 1), padding=(1, 1)):
    # x: (Cin, H, W), w: (Cout, Cin, KH, KW)
    Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    ph, pw = padding
    sh, sw = stride
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    Hp, Wp = xp.shape[1], xp.shape[2]
    Hout = (Hp - KH) // sh + 1
    Wout = (Wp - KW) // sw + 1
    cols = np.empty((Cin, KH, KW, Hout, Wout), np.float64)
    for i in range(KH):
        for j in range(KW):
            cols[:, i, j, :, :] = xp[:, i:i + sh * Hout:sh, j:j + sw * Wout:sw]
    out = np.einsum("oijk,cijhw->ohw".replace("k", "C"), w.astype(np.float64).reshape(Cout, Cin, KH, KW), cols) \
        if False else np.einsum("oCij,Cijhw->ohw", w.astype(np.float64), cols)
    if b is not None:
        out += b.astype(np.float64)[:, None, None]
    return out


def bn(x, sd, prefix, axis, eps=1e-5, affine=True):
    rm = sd[prefix + ".running_mean"].astype(np.float64)
    rv = sd[prefix + ".running_var"].astype(np.float64)
    shape = [1] * x.ndim
    shape[axis] = -1
    y = (x - rm.reshape(shape)) / np.sqrt(rv.reshape(shape) + eps)
    if affine:
        w = sd[prefix + ".weight"].astype(np.float64).reshape(shape)
        b = sd[prefix + ".bias"].astype(np.float64).reshape(shape)
        y = y * w + b
    return y


def relu(x):
    return np.maximum(x, 0.0)


# ----------------------------- CAMPPlus forward ------------------------------
def basic_resblock(x, sd, p, stride):
    out = relu(bn(conv2d(x, sd[p + ".conv1.weight"], stride=(stride, 1), padding=(1, 1)), sd, p + ".bn1", 0))
    out = bn(conv2d(out, sd[p + ".conv2.weight"], stride=(1, 1), padding=(1, 1)), sd, p + ".bn2", 0)
    if (p + ".shortcut.0.weight") in sd:
        sc = conv2d(x, sd[p + ".shortcut.0.weight"], stride=(stride, 1), padding=(0, 0))
        sc = bn(sc, sd, p + ".shortcut.1", 0)
    else:
        sc = x
    return relu(out + sc)


def fcm(x, sd):
    # x: (F=80, T) -> (1, 80, T)
    x = x[None, :, :]
    out = relu(bn(conv2d(x, sd["head.conv1.weight"], stride=(1, 1), padding=(1, 1)), sd, "head.bn1", 0))
    out = basic_resblock(out, sd, "head.layer1.0", 2)
    out = basic_resblock(out, sd, "head.layer1.1", 1)
    out = basic_resblock(out, sd, "head.layer2.0", 2)
    out = basic_resblock(out, sd, "head.layer2.1", 1)
    out = relu(bn(conv2d(out, sd["head.conv2.weight"], stride=(2, 1), padding=(1, 1)), sd, "head.bn2", 0))
    C, H, W = out.shape
    out = out.reshape(C * H, W)  # (320, T)
    return out


def bn_relu(x, sd, prefix):
    return relu(bn(x, sd, prefix + ".batchnorm", 0))


def cam_layer(x, sd, p, dilation, kernel=3):
    pad = (kernel - 1) // 2 * dilation
    y = conv1d(x, sd[p + ".linear_local.weight"], stride=1, padding=pad, dilation=dilation)
    context = x.mean(axis=-1, keepdims=True) + seg_pooling(x, 100)
    context = relu(conv1d(context, sd[p + ".linear1.weight"], sd[p + ".linear1.bias"]))
    m = 1.0 / (1.0 + np.exp(-conv1d(context, sd[p + ".linear2.weight"], sd[p + ".linear2.bias"])))
    return y * m


def seg_pooling(x, seg_len=100):
    # avg_pool1d ceil_mode=True, kernel=stride=seg_len
    C, L = x.shape
    nseg = -(-L // seg_len)  # ceil
    out = np.zeros((C, nseg))
    for s in range(nseg):
        chunk = x[:, s * seg_len:(s + 1) * seg_len]
        out[:, s] = chunk.mean(axis=1)
    seg = np.repeat(out, seg_len, axis=1)[:, :L]
    return seg


def dense_tdnn_layer(x, sd, p, dilation):
    y = bn_relu(x, sd, p + ".nonlinear1")
    y = conv1d(y, sd[p + ".linear1.weight"])  # bn_channels
    y = bn_relu(y, sd, p + ".nonlinear2")
    y = cam_layer(y, sd, p + ".cam_layer", dilation)
    return y


def dense_block(x, sd, p, num_layers, dilation):
    for i in range(num_layers):
        y = dense_tdnn_layer(x, sd, f"{p}.tdnnd{i+1}", dilation)
        x = np.concatenate([x, y], axis=0)
    return x


def transit(x, sd, p):
    x = bn_relu(x, sd, p + ".nonlinear")
    x = conv1d(x, sd[p + ".linear.weight"])
    return x


def xvector(x, sd):
    # tdnn: conv1d stride2 pad2 + bn-relu
    x = conv1d(x, sd["xvector.tdnn.linear.weight"], stride=2, padding=2, dilation=1)
    x = bn_relu(x, sd, "xvector.tdnn.nonlinear")
    x = dense_block(x, sd, "xvector.block1", 12, 1)
    x = transit(x, sd, "xvector.transit1")
    x = dense_block(x, sd, "xvector.block2", 24, 2)
    x = transit(x, sd, "xvector.transit2")
    x = dense_block(x, sd, "xvector.block3", 16, 2)
    x = transit(x, sd, "xvector.transit3")
    x = bn_relu(x, sd, "xvector.out_nonlinear")
    # stats pooling: mean + std(unbiased) over time
    mean = x.mean(axis=-1)
    std = x.std(axis=-1, ddof=1)
    stats = np.concatenate([mean, std])  # (2C,)
    # dense: conv1d 1x1 then bn (affine=False), no relu
    s = stats[:, None]
    s = conv1d(s, sd["xvector.dense.linear.weight"])
    s = bn(s, sd, "xvector.dense.nonlinear.batchnorm", 0, affine=False)
    return s[:, 0]  # (192,)


def campplus(feat, sd):
    # feat: (T, 80) -> permute -> (80, T)
    x = feat.T
    x = fcm(x, sd)
    emb = xvector(x, sd)
    return emb[None, :]  # (1,192)


def synth_wave(n=16000):
    t = np.arange(n) / 16000.0
    return (0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)


def main():
    sd_path, out_dir = sys.argv[1], sys.argv[2]
    sd = load_file(sd_path)
    wave = synth_wave(16000)
    feat = kaldi_fbank(wave)
    feat_ms = feat - feat.mean(axis=0, keepdims=True)
    style = campplus(feat_ms, sd)
    print("fbank:", feat.shape, "mean=%.5f min=%.5f max=%.5f" % (feat.mean(), feat.min(), feat.max()))
    print("style:", style.shape, "mean=%.5f min=%.5f max=%.5f" % (style.mean(), style.min(), style.max()))
    import os
    os.makedirs(out_dir, exist_ok=True)
    feat.astype(np.float32).tofile(os.path.join(out_dir, "fbank.bin"))
    style.astype(np.float32).tofile(os.path.join(out_dir, "style.bin"))
    print("wrote fbank.bin (%d) style.bin (%d)" % (feat.size, style.size))


if __name__ == "__main__":
    main()
