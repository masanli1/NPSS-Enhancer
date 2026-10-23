#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Multi-stat Gmin + single blend (v2.1): noise statistics, not voice."""

import numpy as np
import common as C
import common_win as W
import nmf_blend

SR = 16000
NFFT, HOP = 2048, 512
THETA, SUPPORT, FLOOR = 0.95, 0.60, 0.30
K, BASE = 10, 0.05
PASSES = 3
LOW_HZ = 250.0
LOW_FLOOR_16 = 0.30
LOW_FLOOR_48_MUSIC = 0.70
LOW_FLOOR_VOICE = 0.30


def _protect_low(gate, n_fft, sr, floor):
    if floor <= 0:
        return gate
    g = gate.copy()
    f = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    m = f < LOW_HZ
    if np.any(m):
        g[m, :] = np.maximum(g[m, :], float(floor))
    return g


def _norm(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.mean(x)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def _color(n, rng, slope):
    w = rng.normal(size=n)
    Wf = np.fft.rfft(w)
    f = np.arange(len(Wf), dtype=np.float64); f[0] = 1.0
    Wf = Wf / (f ** (float(slope) / 2.0))
    return np.fft.irfft(Wf, n=n).astype(np.float32)


def _impulsive(n, rng):
    y = 0.02 * rng.normal(size=n)
    idx = rng.random(n) < 0.005
    y[idx] += rng.normal(0.0, 4.0, int(idx.sum()))
    return y


def _am(n, rng, rate=4.0, depth=0.8):
    t = np.arange(n) / float(SR)
    return (rng.normal(size=n).astype(np.float32) *
            (1.0 + depth * np.sin(2 * np.pi * rate * t)))


def _bursty(n, rng):
    env = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        on = int(rng.uniform(0.02, 0.08) * SR)
        off = int(rng.uniform(0.05, 0.20) * SR)
        env[pos:min(n, pos + on)] = 1.0
        pos += on + off
    return rng.normal(size=n).astype(np.float32) * env


def _traffic(n, rng):
    low = np.convolve(rng.normal(size=n), np.ones(300) / 300.0, "same")
    hum = 0.25 * np.sin(2 * np.pi * 60.0 * np.arange(n) / SR)
    return (2.0 * low + hum).astype(np.float32)


def make_stats_bank(n, seed=7):
    """Structureless but statistically different probes (no voice/babble)."""
    rng = np.random.default_rng(seed)
    bank = {
        "white": _color(n, rng, 0.0),
        "student_t": rng.standard_t(3.0, size=n).astype(np.float32),
        "impulsive": _impulsive(n, rng),
        "am": _am(n, rng, 4.0, 0.8),
        "bursty": _bursty(n, rng),
        "traffic": _traffic(n, rng),
    }
    return {k: _norm(v) for k, v in bank.items()}


def _stab_one(x, M, p, fi, seed=0):
    P0 = C.extremum_position(M)
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    acc = np.zeros_like(M, dtype=np.float32)
    for k, a in enumerate(np.logspace(-0.3, 1.0, int(K))):
        rng = np.random.default_rng(seed + 1000 * fi + k)
        nk = np.roll(p, int(rng.integers(0, len(p))))
        xk = x + (nk * (BASE * rms * a)).astype(np.float32)
        Pk = C.extremum_position(np.abs(C.stft(xk, NFFT, HOP)))
        acc += (Pk == P0).astype(np.float32)
    return acc / float(K)


def _cascade_ext(x, M, p, gb, X, fi, seed=0, passes=PASSES):
    S_list = []
    yy = x
    MM = M
    for j in range(int(passes)):
        S = _stab_one(yy, MM, p, fi, seed=seed + 10 * j)
        S_list.append(S)
        if j < int(passes) - 1:
            piv = (S >= THETA).astype(np.float32)
            yy = C.istft(gb * ((1.0 - SUPPORT) + SUPPORT * piv) * X,
                         len(x), NFFT, HOP)
            MM = np.abs(C.stft(yy, NFFT, HOP))
    return np.mean(S_list, axis=0).astype(np.float32)


def single_full(x, low_floor=LOW_FLOOR_16):
    y1, _, S1, _ = W.cascade_contrast_win(
        x, SR, "correlated", wb=2, wc=4, passes=PASSES,
        theta=THETA, support=SUPPORT, n_fft=NFFT, hop=HOP,
        backend="specsub", gate_floor=FLOOR)
    S2 = W.cascade_contrast_win(
        y1, SR, "correlated", wb=2, wc=4, passes=PASSES,
        theta=THETA, support=SUPPORT, n_fft=NFFT, hop=HOP,
        backend="specsub", gate_floor=FLOOR)[2]
    X, M, gb, _, _, _ = C.prepare(x, NFFT, HOP, 1.5, 0.05, "specsub")
    gate = C.contrast_gate(0.5 * (S1 + S2), 30, 90, FLOOR)
    gate = _protect_low(gate, NFFT, SR, low_floor)
    return C.istft(gb * gate * X, len(x), NFFT, HOP)


def gmin_full(x, seed=7, low_floor=LOW_FLOOR_16):
    X, M, gb, _, _, _ = C.prepare(x, NFFT, HOP, 1.5, 0.05, "specsub")
    bank = make_stats_bank(len(x), seed)
    S1 = {k: _cascade_ext(x, M, p, gb, X, i, seed=seed)
          for i, (k, p) in enumerate(bank.items())}
    g1 = np.min(np.stack([C.contrast_gate(S1[k], 30, 90, FLOOR) for k in S1], 0), axis=0)
    g1 = _protect_low(g1, NFFT, SR, low_floor)
    y1 = C.istft(gb * g1 * X, len(x), NFFT, HOP)
    M1 = np.abs(C.stft(y1, NFFT, HOP))
    S2 = {k: _cascade_ext(y1, M1, p, gb, X, i, seed=seed + 100)
          for i, (k, p) in enumerate(bank.items())}
    S = {k: 0.5 * (S1[k] + S2[k]) for k in S1}
    g = np.min(np.stack([C.contrast_gate(S[k], 30, 90, FLOOR) for k in S], 0), axis=0)
    g = _protect_low(g, NFFT, SR, low_floor)
    return C.istft(gb * g * X, len(x), NFFT, HOP)


def _resample_fft(x, sr_in, sr_out):
    """FFT resample (zero-pad/truncate); good enough for the 16k/48k bridge."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr_in) == int(sr_out) or len(x) == 0:
        return x.astype(np.float32)
    n_in = len(x)
    n_out = int(round(n_in * float(sr_out) / float(sr_in)))
    if n_out < 2:
        return np.zeros(2, dtype=np.float32)
    X = np.fft.rfft(x.astype(np.float64))
    Y = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    m = min(len(X), len(Y))
    Y[:m] = X[:m]
    y = np.fft.irfft(Y, n=n_out)
    y *= float(n_out) / float(n_in)
    return y.astype(np.float32)


def _fit_len(y, n):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n].copy()
    return np.pad(y, (0, n - len(y))).astype(np.float32)


def denoise_fullband(x, sr=48000, voice=False, seed=7):
    """v2.3: 16k multi-stat for 0-8k + untouched 8k+ high band; full-band out."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    x16 = _resample_fft(x, sr, 16000)
    low = LOW_FLOOR_VOICE if voice else LOW_FLOOR_48_MUSIC
    y16 = denoise_best(x16, 16000, voice=voice, seed=seed, low_floor=low)
    ylow = _resample_fft(y16, 16000, sr)
    n_fft = 4096 if sr >= 32000 else 2048
    hop = n_fft // 4
    X = C.stft(x, n_fft, hop)
    f = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    Xh = X.copy()
    Xh[f < 8000.0] = 0.0
    xh = C.istft(Xh, len(x), n_fft, hop)
    n = len(x)
    ylow = _fit_len(ylow, n)
    xh = _fit_len(xh, n)
    y = ylow + xh
    rms_in = float(np.sqrt(np.mean(x[:n] ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = y * (rms_in / rms_out)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def denoise_best(x, sr=SR, voice=False, seed=7, low_floor=None):
    """v2.3: 50% single + 50% multi-stat Gmin; voice adds 40% NMF."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        return denoise_fullband(x, sr, voice=voice, seed=seed)
    if low_floor is None:
        low_floor = LOW_FLOOR_VOICE if voice else LOW_FLOOR_16
    ys = single_full(x, low_floor=low_floor)
    yg = gmin_full(x, seed=seed, low_floor=low_floor)
    yb = 0.5 * (ys + yg)
    if voice:
        yn = nmf_blend.enhance(x)
        yb = 0.6 * yb + 0.4 * yn
    return np.clip(yb, -1.0, 1.0).astype(np.float32)
