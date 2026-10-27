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

from collections import namedtuple

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


def _stab_one(x, M, p, fi, seed=0, klevels=K):
    P0 = C.extremum_position(M)
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    acc = np.zeros_like(M, dtype=np.float32)
    for k, a in enumerate(np.logspace(-0.3, 1.0, int(klevels))):
        rng = np.random.default_rng(seed + 1000 * fi + k)
        nk = np.roll(p, int(rng.integers(0, len(p))))
        xk = x + (nk * (BASE * rms * a)).astype(np.float32)
        Pk = C.extremum_position(np.abs(C.stft(xk, NFFT, HOP)))
        acc += (Pk == P0).astype(np.float32)
    return acc / float(klevels)


def _cascade_ext(x, M, p, gb, X, fi, seed=0, passes=PASSES, klevels=K):
    S_list = []
    yy = x
    MM = M
    for j in range(int(passes)):
        S = _stab_one(yy, MM, p, fi, seed=seed + 10 * j, klevels=klevels)
        S_list.append(S)
        if j < int(passes) - 1:
            piv = (S >= THETA).astype(np.float32)
            yy = C.istft(gb * ((1.0 - SUPPORT) + SUPPORT * piv) * X,
                         len(x), NFFT, HOP)
            MM = np.abs(C.stft(yy, NFFT, HOP))
    return np.mean(S_list, axis=0).astype(np.float32)


def _region_weights(S, nr=16, bands=8):
    """Soft cross-region topology weight: 0.8 + 0.2 * normalized degree."""
    F, T = S.shape
    R = int(np.ceil(T / float(nr)))
    pad = np.pad(S, ((0, 0), (0, R * nr - T)), mode="edge")
    blocks = pad.reshape(F, R, nr)
    edges = np.linspace(0, F, bands + 1).astype(int)
    feat = np.zeros((R, bands), dtype=np.float32)
    for b in range(bands):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        feat[:, b] = blocks[lo:hi].mean(axis=(0, 2))
    feat = feat - feat.mean(axis=1, keepdims=True)
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    deg = np.zeros(R, dtype=np.float32)
    for r in range(R):
        if r > 0:
            deg[r] += max(0.0, float(feat[r] @ feat[r - 1]))
        if r < R - 1:
            deg[r] += max(0.0, float(feat[r] @ feat[r + 1]))
    w = 0.8 + 0.2 * deg / (deg.max() + 1e-8)
    frames = np.repeat(w, nr)[:T]
    return np.tile(frames[None, :], (F, 1)).astype(np.float32)


def _complete_pivots(S):
    """High-contribution pivots fill full; low-contribution only auxiliary."""
    F, T = S.shape
    piv = S >= THETA
    anchor = np.zeros_like(piv)
    if T > 1:
        anchor[:, 1:] |= piv[:, :-1]
        anchor[:, :-1] |= piv[:, 1:]
    anchor &= piv
    fill = np.zeros_like(piv)
    for g in (1, 2):
        if T > 2 * g:
            both = piv[:, :T - 2 * g] & piv[:, 2 * g:]
            side = anchor[:, :T - 2 * g] | anchor[:, 2 * g:]
            fill[:, g:T - g] |= both & side
    fill &= S >= 0.60
    if F > 2:
        both = piv[:-2, :] & piv[2:, :]
        side = anchor[:-2, :] | anchor[2:, :]
        fill[1:-1, :] |= both & side & (S[1:-1, :] >= 0.60)
    out = S.copy()
    out[fill] = np.maximum(out[fill], THETA)
    # auxiliary: low-contribution bin between two high pivots
    lowm = (S >= 0.65) & (~piv)
    aux = np.zeros_like(piv)
    if T > 2:
        aux[:, 1:-1] |= lowm[:, 1:-1] & piv[:, :-2] & piv[:, 2:]
    if F > 2:
        aux[1:-1, :] |= lowm[1:-1, :] & piv[:-2, :] & piv[2:, :]
    out[aux] = np.maximum(out[aux], 0.82)
    return out

def _energy_gate(gate, M):
    """Energy/SNR weighting: high-SNR bins keep more, low-SNR bins less."""
    m2 = M * M
    npow = C.estimate_noise(M, 1.5)
    snr = 10.0 * np.log10(m2 / (npow + 1e-12) + 1e-12)
    e = 1.0 / (1.0 + np.exp(-(snr - 6.0) / 4.0))
    return (gate * (0.8 + 0.2 * e)).astype(np.float32)


Loads = namedtuple("Loads", "X M gb S G Ss Sg Gs Gg")


def _single_gate(x, X, M, gb, low_floor, klevels, passes, energy):
    """Legacy single-family gate: 2 cascades + completion + topology."""
    y1, _, S1, _ = W.cascade_contrast_win(
        x, SR, "correlated", wb=2, wc=4, passes=passes, K=klevels,
        theta=THETA, support=SUPPORT, n_fft=NFFT, hop=HOP,
        backend="specsub", gate_floor=FLOOR)
    S2 = W.cascade_contrast_win(
        y1, SR, "correlated", wb=2, wc=4, passes=passes, K=klevels,
        theta=THETA, support=SUPPORT, n_fft=NFFT, hop=HOP,
        backend="specsub", gate_floor=FLOOR)[2]
    S = 0.5 * (S1 + S2)
    S = _complete_pivots(S)
    S = S * _region_weights(S)
    gate = C.contrast_gate(S, 30, 90, FLOOR)
    if energy:
        gate = _energy_gate(gate, M)
    gate = _protect_low(gate, NFFT, SR, low_floor)
    return S.astype(np.float32), (gb * gate).astype(np.float32)


def _gmin_gate(x, X, M, gb, seed, low_floor, klevels, passes, energy):
    """Multi-stat Gmin gate: min over 6 probe statistics, 2 rounds."""
    bank = make_stats_bank(len(x), seed)
    S1 = {k: _cascade_ext(x, M, p, gb, X, i, seed=seed,
                          klevels=klevels, passes=passes)
          for i, (k, p) in enumerate(bank.items())}
    g1 = np.min(np.stack([C.contrast_gate(S1[k], 30, 90, FLOOR) for k in S1], 0), axis=0)
    if energy:
        g1 = _energy_gate(g1, M)
    g1 = _protect_low(g1, NFFT, SR, low_floor)
    y1 = C.istft(gb * g1 * X, len(x), NFFT, HOP)
    M1 = np.abs(C.stft(y1, NFFT, HOP))
    S2 = {k: _cascade_ext(y1, M1, p, gb, X, i, seed=seed + 100,
                          klevels=klevels, passes=passes)
          for i, (k, p) in enumerate(bank.items())}
    S = {k: _complete_pivots(0.5 * (S1[k] + S2[k])) for k in S1}
    w = _region_weights(np.mean(np.stack(list(S.values()), 0), axis=0))
    S = {k: S[k] * w for k in S}
    g = np.min(np.stack([C.contrast_gate(S[k], 30, 90, FLOOR) for k in S], 0), axis=0)
    if energy:
        g = _energy_gate(g, M)
    g = _protect_low(g, NFFT, SR, low_floor)
    Sm = np.mean(np.stack([S[k] for k in S], 0), axis=0)
    return Sm.astype(np.float32), (gb * g).astype(np.float32)


def _profile_gate(G, M, S, x, profile="freq", beta=0.5):
    """v2.9 quality profile: soft four-view modulation of the base gate.

    ``freq`` gave the best PESQ/STOI in the VoiceBank four-view tune; ``off``
    reproduces v2.8; ``energy`` / ``struct`` are the other measured profiles.
    """
    if not profile or profile == "off":
        return G
    if profile == "blend":
        Gf = _profile_gate(G, M, S, x, profile="freq", beta=beta)
        Gs = _profile_gate(G, M, S, x, profile="struct", beta=beta)
        return (0.5 * (Gf + Gs)).astype(np.float32)
    import four_views as FV
    freqs = np.fft.rfftfreq(NFFT, 1.0 / float(SR)).astype(np.float32)
    npow = C.estimate_noise(M, 1.5)
    if profile == "freq":
        I, _ = FV.freq_view(M, npow, freqs)
    elif profile == "energy":
        I, _ = FV.energy_view(M, npow)
    elif profile == "struct":
        I, _ = FV.struct_view(x, M, S, freqs, SR, NFFT, HOP)
    else:
        return G
    I = FV.smooth_protect(I, 1)
    e = np.clip(1.0 + float(beta) * (0.5 - I), 0.25, 2.0)
    return np.clip(np.power(np.maximum(G, 1e-6), e), 0.0, 1.0).astype(np.float32)


def _stoi_guard(G, M, alpha=0.25):
    """Band-envelope guard: only restore over-suppressed band energy."""
    import four_views as FV
    freqs = np.fft.rfftfreq(NFFT, 1.0 / float(SR)).astype(np.float32)
    band_of, nb = FV.band_index(freqs, 24)
    Pi = np.asarray(M, dtype=np.float32) ** 2
    Po = (np.asarray(G, dtype=np.float32) * M) ** 2
    Ein = np.zeros((nb, M.shape[1]), dtype=np.float32)
    Eout = np.zeros_like(Ein)
    for b in range(nb):
        m = band_of == b
        if np.any(m):
            Ein[b] = Pi[m].sum(axis=0)
            Eout[b] = Po[m].sum(axis=0)
    k = np.ones(5, dtype=np.float32) / 5.0
    for b in range(nb):
        Ein[b] = np.convolve(Ein[b], k, mode="same")
        Eout[b] = np.convolve(Eout[b], k, mode="same")
    r = np.sqrt((Ein + 1e-12) / (Eout + 1e-12))
    r = np.clip(r, 1.0, 1.30)
    return np.clip(G * np.power(r[band_of], float(alpha)), 0.0, 1.0).astype(np.float32)


def loads(x, sr=SR, voice=False, seed=7, low_floor=None, fast=False, energy=True,
          view_profile="freq", view_beta=0.5, stoi_guard=False, stoi_alpha=0.25):
    """All intermediate state needed to re-gate without recomputation.

    ``denoise_best`` is ``istft(G * X)`` with ``G = 0.5 * (Gs + Gg)``; exposing
    the loads lets the target controller re-derive the output for a new dose at
    the cost of one ISTFT instead of a full 180-STFT cascade.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        raise ValueError("loads() expects %d Hz" % SR)
    if low_floor is None:
        low_floor = LOW_FLOOR_VOICE if voice else LOW_FLOOR_16
    kl = 6 if fast else K
    ps = 2 if fast else PASSES
    X, M, gb, _, _, _ = C.prepare(x, NFFT, HOP, 1.5, 0.05, "specsub")
    Ss, Gs = _single_gate(x, X, M, gb, low_floor, kl, ps, energy)
    Sg, Gg = _gmin_gate(x, X, M, gb, seed, low_floor, kl, ps, energy)
    S = 0.5 * (Ss + Sg)
    G = 0.5 * (Gs + Gg)
    G = _profile_gate(G, M, S, x, profile=view_profile, beta=view_beta)
    if stoi_guard:
        G = _stoi_guard(G, M, stoi_alpha)
    return Loads(X=X, M=M, gb=gb, S=S.astype(np.float32), G=G.astype(np.float32),
                 Ss=Ss, Sg=Sg, Gs=Gs, Gg=Gg)


def single_full(x, low_floor=LOW_FLOOR_16, klevels=K, passes=PASSES, energy=True):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    X, M, gb, _, _, _ = C.prepare(x, NFFT, HOP, 1.5, 0.05, "specsub")
    _, G = _single_gate(x, X, M, gb, low_floor, klevels, passes, energy)
    return C.istft(G * X, len(x), NFFT, HOP)


def gmin_full(x, seed=7, low_floor=LOW_FLOOR_16, klevels=K, passes=PASSES, energy=True):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    X, M, gb, _, _, _ = C.prepare(x, NFFT, HOP, 1.5, 0.05, "specsub")
    _, G = _gmin_gate(x, X, M, gb, seed, low_floor, klevels, passes, energy)
    return C.istft(G * X, len(x), NFFT, HOP)


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


def denoise_fullband(x, sr=48000, voice=False, seed=7, fast=False,
                     target_nr_db=None, target_weights=None, target_dose=None,
                     info_out=None):
    """v2.7: 16k multi-stat for 0-8k + untouched 8k+ high band; full-band out.

    The target applies to the 0-8 kHz band: the 8k+ band is deliberately
    preserved for music, so it adds its own (small) noise back on top.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    x16 = _resample_fft(x, sr, 16000)
    low = LOW_FLOOR_VOICE if voice else LOW_FLOOR_48_MUSIC
    y16 = denoise_best(x16, 16000, voice=voice, seed=seed, low_floor=low,
                       fast=fast, energy=False, target_nr_db=target_nr_db,
                       target_weights=target_weights, target_dose=target_dose,
                       info_out=info_out)
    ylow = _resample_fft(y16, 16000, sr)
    n_fft = 4096 if sr >= 32000 else 2048
    hop = n_fft // 4
    X, M48, gb48, _, _, _ = C.prepare(x, n_fft, hop, 1.5, 0.05, "specsub")
    f = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    Xh = X.copy()
    Xh[f < 8000.0] = 0.0
    # gentle high-band gate: preserve music, suppress hiss
    hi = f >= 8000.0
    Xh[hi, :] *= np.maximum(gb48[hi, :], 0.5) * 0.70
    xh = C.istft(Xh, len(x), n_fft, hop)
    n = len(x)
    ylow = _fit_len(ylow, n)
    xh = _fit_len(xh, n)
    y = ylow + xh
    rms_in = float(np.sqrt(np.mean(x[:n] ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = y * (rms_in / rms_out)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def denoise_best(x, sr=SR, voice=False, seed=7, low_floor=None, fast=False,
                 energy=True, target_nr_db=None, target_weights=None,
                 target_dose=None, target_clean=None, info_out=None,
                 view_profile="freq", view_beta=0.5, stoi_guard=False,
                 stoi_alpha=0.25):
    """v2.7: multi-stat Gmin + topology + role completion + energy; voice adds 40% NMF.

    ``target_nr_db`` switches to the 定果寻因 path: the noise floor is driven to
    exactly that many dB of attenuation, with the four-view protect map
    deciding where the dose may land.  ``target_dose`` applies an already
    solved dose (one-slice calibration, applied to every chunk).  The measured
    outcome is written into ``info_out`` when given.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        return denoise_fullband(x, sr, voice=voice, seed=seed, fast=fast,
                                target_nr_db=target_nr_db,
                                target_weights=target_weights,
                                target_dose=target_dose,
                                info_out=info_out)
    if low_floor is None:
        low_floor = LOW_FLOOR_VOICE if voice else LOW_FLOOR_16
    L = loads(x, SR, voice=voice, seed=seed, low_floor=low_floor,
              fast=fast, energy=energy, view_profile=view_profile,
              view_beta=view_beta, stoi_guard=stoi_guard,
              stoi_alpha=stoi_alpha)
    G = L.G
    if target_nr_db is not None or target_dose is not None:
        import target_nr as TG
        G, info = TG.shape_gate(L, x, SR, target_db=target_nr_db,
                                dose=target_dose, weights=target_weights,
                                clean=target_clean)
        if info_out is not None:
            info_out.clear()
            info_out.update({k: v for k, v in info.items() if k != "views"})
    yb = C.istft(G * L.X, len(x), NFFT, HOP)
    if voice:
        yn = nmf_blend.enhance(x)
        yb = 0.6 * yb + 0.4 * yn
    return np.clip(yb, -1.0, 1.0).astype(np.float32)
