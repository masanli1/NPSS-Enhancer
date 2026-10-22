#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Shared DSP for the three NPSS stages (white / impulsive / correlated)."""
import argparse
import wave
import numpy as np

try:
    import soundfile as _sf
except Exception:
    _sf = None


# ---------------------------------------------------------------- WAV I/O
def read_wav(path):
    try:
        with wave.open(path, "rb") as w:
            nch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if sw == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / float(1 << 23)
        elif sw == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
        else:
            raise ValueError("unsupported sample width: %d" % (sw * 8))
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        return np.ascontiguousarray(x.astype(np.float32)), int(sr)
    except Exception:
        if _sf is None:
            raise
        data, sr = _sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1).astype(np.float32)), int(sr)


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


# ---------------------------------------------------------------- STFT
def stft(x, n_fft=512, hop=128):
    w = np.hanning(n_fft).astype(np.float32)
    xp = np.pad(np.asarray(x, dtype=np.float32), (n_fft, n_fft), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop] * w
    return np.fft.rfft(frames, axis=1).T.astype(np.complex64)


def istft(X, length, n_fft=512, hop=128):
    w = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1).astype(np.float32) * w
    T = frames.shape[0]
    y = np.zeros((T - 1) * hop + n_fft, dtype=np.float32)
    ws = np.zeros_like(y)
    for k in range(T):
        y[k * hop:k * hop + n_fft] += frames[k]
        ws[k * hop:k * hop + n_fft] += w * w
    y = y / np.maximum(ws, 1e-8)
    return y[n_fft:n_fft + length]


# ---------------------------------------------------------------- noise / gain
def medfilt2(a, k=3, chunk=256):
    if k <= 1:
        return a.astype(np.float32)
    p = k // 2
    Ap = np.pad(a, ((p, p), (p, p)), mode="edge")
    Wv = np.lib.stride_tricks.sliding_window_view(Ap, (k, k))
    out = np.empty_like(a)
    for t0 in range(0, a.shape[1], chunk):
        t1 = min(t0 + chunk, a.shape[1])
        out[:, t0:t1] = np.median(Wv[:, t0:t1], axis=(-2, -1))
    return out.astype(np.float32)


def estimate_noise(mag, bias=1.5):
    P = mag * mag
    Ps = medfilt2(P, 3)
    return (np.percentile(Ps, 10, axis=1, keepdims=True).astype(np.float32)
            * float(bias))


def _exp1(x):
    """Exponential integral E1(x), A&S 5.1.53/5.1.56 approximations.
    float64 + clip to avoid overflow; E1(x) ~ 0 for x > 50."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-8)
    xc = np.minimum(x, 1e6)
    small = (-np.log(xc) - 0.57721566 +
             xc * (0.99999193 + xc * (-0.24991055 +
             xc * (0.05519968 + xc * (-0.00976004 + xc * 0.00107857)))))
    big = (np.exp(-xc) / xc *
           (xc ** 4 + 8.5733287401 * xc ** 3 + 18.0590169730 * xc ** 2 +
            8.6347608925 * xc + 0.2677737343) /
           (xc ** 4 + 9.5733223454 * xc ** 3 + 25.6329561486 * xc ** 2 +
            21.0996530827 * xc + 3.9584969228))
    out = np.where(xc <= 1.0, small, big)
    out = np.where(x > 50.0, 0.0, out)
    return out.astype(np.float32)


def backend_gain(mag, mode="wiener", floor=0.05, noise_bias=1.5, npow=None):
    m2 = mag * mag
    if npow is None:
        npow = estimate_noise(mag, noise_bias)
    if mode == "specsub":
        g = 1.0 - np.sqrt(npow) / (mag + 1e-8)
    elif mode == "mmse":
        xi = np.maximum(m2 / (npow + 1e-12) - 1.0, 0.0)
        g = xi / (1.0 + xi)
    elif mode == "mmse_lsa":
        # MMSE-LSA gain
        xi = np.maximum(m2 / (npow + 1e-12) - 1.0, 0.0)
        gamma = m2 / (npow + 1e-12)
        v = xi * gamma / (1.0 + xi)
        g = xi / (1.0 + xi) * np.exp(0.5 * _exp1(v))
    elif mode == "omlsa":
        # simplified OM-LSA: speech-presence-weighted Wiener
        xi = np.maximum(m2 / (npow + 1e-12) - 1.0, 0.0)
        gamma = m2 / (npow + 1e-12)
        g_h1 = xi / (1.0 + xi)
        p = 1.0 / (1.0 + (1.0 + xi) * np.exp(-gamma * xi / (1.0 + xi)))
        g = (float(floor) ** (1.0 - p)) * (g_h1 ** p)
    else:
        g = 1.0 - npow / (m2 + 1e-12)
    return np.clip(g, float(floor), 1.0).astype(np.float32)


def extremum_position(mag):
    F, T = mag.shape
    Mp = np.pad(mag, ((2, 2), (2, 2)), mode="edge")
    B = np.lib.stride_tricks.sliding_window_view(Mp, (2, 2))[2:2 + F, 2:2 + T]
    C = np.lib.stride_tricks.sliding_window_view(Mp, (4, 4))[1:1 + F, 1:1 + T]
    m2, m4 = B.mean(axis=(2, 3)), C.mean(axis=(2, 3))
    flat = B.reshape(F, T, 4)
    return np.where(m2 > m4, flat.argmax(axis=2), flat.argmin(axis=2)).astype(np.int8)


def prepare(x, n_fft=512, hop=128, noise_bias=1.5, floor=0.05, backend="wiener"):
    X = stft(x, n_fft, hop)
    M = np.abs(X).astype(np.float32)
    npow = estimate_noise(M, noise_bias)
    p_sig = max(float(np.mean(M * M) - np.mean(npow)), 1e-12)
    snr_in = float(10.0 * np.log10(p_sig / (float(np.mean(npow)) + 1e-12)))
    gb = backend_gain(M, backend, floor, noise_bias, npow)
    Y0 = istft(gb * X, len(x), n_fft, hop)
    R = np.asarray(x, dtype=np.float32) - Y0
    return X, M, gb, Y0, R, snr_in


# ---------------------------------------------------------------- families
def family_noise(kind, n, rng):
    if kind == "white":
        return rng.normal(size=n)
    if kind == "correlated":
        w = rng.normal(size=n)
        y = np.convolve(w, np.ones(80) / 80.0, "same")
        return y / (np.std(y) + 1e-12)
    if kind == "impulsive":
        y = 0.02 * rng.normal(size=n)
        idx = rng.random(n) < 0.01
        y[idx] += rng.normal(0.0, 4.0, idx.sum())
        return y / (np.std(y) + 1e-12)
    raise ValueError("unknown family: %s" % kind)


def stability_for_family(x, M, family, K=10, base=0.05, seed=0,
                         n_fft=512, hop=128):
    P0 = extremum_position(M)
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    acc = np.zeros_like(M, dtype=np.float32)
    for k, a in enumerate(np.logspace(-0.3, 1.0, int(K))):
        rng = np.random.default_rng(int(seed) + 700 + k)
        nk = family_noise(family, len(x), rng)
        nk = nk / (np.std(nk) + 1e-12) * base * rms * a
        xk = np.asarray(x, dtype=np.float32) + nk.astype(np.float32)
        Pk = extremum_position(np.abs(stft(xk, n_fft, hop)))
        acc += (Pk == P0).astype(np.float32)
    return acc / float(max(1, int(K)))


def stage_denoise(x, sr, family, K=10, theta=0.90, support=0.50,
                  base=0.05, seed=0, n_fft=512, hop=128,
                  noise_bias=1.5, floor=0.05, backend="wiener"):
    """One stage: one family, K levels, hard threshold, gated backend."""
    del sr
    X, M, gb, Y0, R, snr_in = prepare(x, n_fft, hop, noise_bias, floor, backend)
    S = stability_for_family(x, M, family, K, base, seed, n_fft, hop)
    piv = (S >= float(theta)).astype(np.float32)
    G = gb * ((1.0 - float(support)) + float(support) * piv)
    y = istft(G * X, len(x), n_fft, hop)
    return y.astype(np.float32), snr_in, S, piv


def cascade_denoise(x, sr, family, passes=3, theta=0.90, support=0.50,
                    K=10, base=0.05, seed=0, n_fft=512, hop=128,
                    noise_bias=1.5, floor=0.05, backend="wiener",
                    combine="vote"):
    """Pass 1 + pass 2/3 support re-extraction, combined by majority vote.
    Noise PSD stays from pass 1 (residual re-estimation was harmful)."""
    del sr
    X, M, gb, Y0, R, snr_in = prepare(x, n_fft, hop, noise_bias, floor, backend)
    y, _, S1, piv1 = stage_denoise(x, 0, family, K=K, theta=theta,
                                   support=support, base=base, seed=seed,
                                   n_fft=n_fft, hop=hop, noise_bias=noise_bias,
                                   floor=floor, backend=backend)
    pivots = [piv1.astype(np.float32)]
    for p in range(2, int(passes) + 1):
        Y = stft(y, n_fft, hop)
        Sp = stability_for_family(y, np.abs(Y), family, K=K, base=base,
                                  seed=seed + p * 100, n_fft=n_fft, hop=hop)
        pp = (Sp >= theta).astype(np.float32)
        pivots.append(pp)
        G = gb * ((1.0 - support) + support * pp)
        y = istft(G * X, len(x), n_fft, hop)
    if combine == "replace":
        vote = pivots[-1]
    elif combine == "union":
        vote = np.max(np.stack(pivots, 0), axis=0).astype(np.float32)
    else:
        need = len(pivots) // 2 + 1
        vote = (np.sum(pivots, axis=0) >= need).astype(np.float32)
    G = gb * ((1.0 - support) + support * vote)
    y = istft(G * X, len(x), n_fft, hop)
    return y, snr_in, pivots, vote


def contrast_gate(S, lo_pct=30, hi_pct=90, floor=0.1):
    """Relative invariant/variable contrast: low-S sets the lower reference,
    high-S the upper reference; the transition is the edge."""
    lo = float(np.percentile(S, lo_pct))
    hi = float(np.percentile(S, hi_pct))
    return np.clip((S - lo) / (hi - lo + 1e-8), float(floor), 1.0).astype(np.float32)


def cascade_contrast(x, sr, family, passes=3, theta=0.90, support=0.50,
                     K=10, base=0.05, seed=0, n_fft=512, hop=128,
                     noise_bias=1.5, floor=0.05, backend="wiener",
                     lo_pct=30, hi_pct=90, gate_floor=0.1):
    """Pass 1..P stability maps -> average -> relative contrast gate."""
    del sr
    X, M, gb, Y0, R, snr_in = prepare(x, n_fft, hop, noise_bias, floor, backend)
    y, _, S1, piv1 = stage_denoise(x, 0, family, K=K, theta=theta,
                                   support=support, base=base, seed=seed,
                                   n_fft=n_fft, hop=hop, noise_bias=noise_bias,
                                   floor=floor, backend=backend)
    S_list = [S1]
    for p in range(2, int(passes) + 1):
        Y = stft(y, n_fft, hop)
        Sp = stability_for_family(y, np.abs(Y), family, K=K, base=base,
                                  seed=seed + p * 100, n_fft=n_fft, hop=hop)
        S_list.append(Sp)
        pp = (Sp >= theta).astype(np.float32)
        y = istft(gb * ((1.0 - support) + support * pp) * X, len(x), n_fft, hop)
    Savg = np.mean(S_list, axis=0).astype(np.float32)
    gate = contrast_gate(Savg, lo_pct, hi_pct, gate_floor)
    y = istft(gb * gate * X, len(x), n_fft, hop)
    return y, snr_in, Savg, gate


# ---------------------------------------------------------------- metrics
def si_sdr(est, ref):
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t, e = a * ref, est - a * ref
    return 10.0 * np.log10(np.sum(t * t) / (np.sum(e * e) + 1e-12))


def snr(est, ref):
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return 10.0 * np.log10(np.sum(ref ** 2) /
                           (np.sum((est - ref) ** 2) + 1e-12))


def stage_cli(family):
    p = argparse.ArgumentParser(description="NPSS stage: %s" % family)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--theta", type=float, default=0.90)
    p.add_argument("--support", type=float, default=0.50)
    p.add_argument("--base", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    x, sr = read_wav(a.input)
    y, se, S, piv = stage_denoise(x, sr, family, K=a.K, theta=a.theta,
                                  support=a.support, base=a.base, seed=a.seed)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak * 0.99
    write_wav(a.output, y, sr)
    print("stage=%s  input SNR est=%.2f dB  pivots=%.3f  -> %s" %
          (family, se, float(piv.mean()), a.output))
