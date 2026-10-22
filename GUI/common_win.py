#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Parameterized local window for NPSS: wb x wb block, wc x wc context."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C


def extremum_position_win(mag, wb=2, wc=None):
    """P(f,t) in [0, wb*wb-1]: argmax/argmin of the wb x wb block,
    polarity chosen by block mean vs centered wc x wc mean."""
    if wc is None:
        wc = wb + 2
    F, T = mag.shape
    p = wc + 1
    Mp = np.pad(mag, ((p, p), (p, p)), mode="edge")
    B = np.lib.stride_tricks.sliding_window_view(Mp, (wb, wb))[p:p + F, p:p + T]
    off = max(0, (wc - wb) // 2)
    Cc = np.lib.stride_tricks.sliding_window_view(Mp, (wc, wc))[
        p - off:p - off + F, p - off:p - off + T]
    m2 = B.mean(axis=(2, 3))
    mc = Cc.mean(axis=(2, 3))
    flat = B.reshape(F, T, wb * wb)
    P = np.where(m2 > mc, flat.argmax(axis=2), flat.argmin(axis=2))
    return P.astype(np.int16)


def stability_win(x, M, family, wb=2, wc=None, K=10, base=0.05, seed=0,
                  n_fft=512, hop=128):
    P0 = extremum_position_win(M, wb, wc)
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    acc = np.zeros_like(M, dtype=np.float32)
    for k, a in enumerate(np.logspace(-0.3, 1.0, int(K))):
        rng = np.random.default_rng(int(seed) + 700 + k)
        nk = C.family_noise(family, len(x), rng)
        nk = nk / (np.std(nk) + 1e-12) * base * rms * a
        xk = np.asarray(x, dtype=np.float32) + nk.astype(np.float32)
        Pk = extremum_position_win(np.abs(C.stft(xk, n_fft, hop)), wb, wc)
        acc += (Pk == P0).astype(np.float32)
    return acc / float(max(1, int(K)))


def stage_denoise_win(x, sr, family, wb=2, wc=None, K=10, theta=0.90,
                      support=0.50, base=0.05, seed=0, n_fft=512, hop=128,
                      noise_bias=1.5, floor=0.05, backend="wiener"):
    del sr
    X, M, gb, Y0, R, snr_in = C.prepare(x, n_fft, hop, noise_bias, floor, backend)
    S = stability_win(x, M, family, wb, wc, K, base, seed, n_fft, hop)
    piv = (S >= theta).astype(np.float32)
    G = gb * ((1.0 - support) + support * piv)
    y = C.istft(G * X, len(x), n_fft, hop)
    return y.astype(np.float32), snr_in, S, piv


def cascade_contrast_win(x, sr, family, wb=2, wc=None, passes=3, theta=0.90,
                         support=0.50, K=10, base=0.05, seed=0,
                         n_fft=512, hop=128, noise_bias=1.5, floor=0.05,
                         backend="wiener", lo_pct=30, hi_pct=90,
                         gate_floor=0.1):
    """Pass 1..P stability -> average -> relative contrast gate."""
    del sr
    X, M, gb, Y0, R, snr_in = C.prepare(x, n_fft, hop, noise_bias, floor, backend)
    y, _, S1, piv1 = stage_denoise_win(
        x, 0, family, wb, wc, K=K, theta=theta, support=support, base=base,
        seed=seed, n_fft=n_fft, hop=hop, noise_bias=noise_bias,
        floor=floor, backend=backend)
    S_list = [S1]
    for p in range(2, int(passes) + 1):
        Y = C.stft(y, n_fft, hop)
        Sp = stability_win(y, np.abs(Y), family, wb, wc, K, base,
                           seed + p * 100, n_fft, hop)
        S_list.append(Sp)
        pp = (Sp >= theta).astype(np.float32)
        y = C.istft(gb * ((1.0 - support) + support * pp) * X,
                    len(x), n_fft, hop)
    Savg = np.mean(S_list, axis=0).astype(np.float32)
    gate = C.contrast_gate(Savg, lo_pct, hi_pct, gate_floor)
    y = C.istft(gb * gate * X, len(x), n_fft, hop)
    return y.astype(np.float32), snr_in, Savg, gate
