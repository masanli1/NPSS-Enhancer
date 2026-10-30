#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""人声特征提取与分离 (v3.3, pure numpy).

降噪回答的是“噪声有多稳定”，这里回答的是“这一格像不像人在说话”。

分析链路
    1. 基频：倒谱候选 + Viterbi 平滑，60–400 Hz，抑制八度跳变；
    2. 谐波梳：按 f0 生成时变谐波掩码（梳齿之间给一个可控下限）；
    3. 清浊：归一化自相关峰 → 浊音概率；
    4. 共振峰：每帧 LPC 求根，F1/F2/F3 轨迹（用于报告与话音包络）；
    5. 语音段：浊音核心（谐波 + 音节调制 + 周期度 + 信噪比）向两侧扩张，
       清辅音跟着元音走；短于 80 ms 的孤立核心丢掉；
    6. 长期话音谱（LTASS）：本段录音自己的音色指纹。

分离链路
    G = G_npss * G_voice
    G_voice = floor + (1-floor) * [VAD * (浊音 ? 谐波梳 : 全通)]
    G_npss 仍是 v2.7 的多统计 Gmin（结构稳定性），所以语音质量不倒退；
    G_voice 负责把“人声之外”的部分按语音特征压下去：停顿时接近静音，
    浊音段只留谐波梳，清辅音按 VAD 放行。

本模块不训练、不加载模型，16 kHz 内部分析，输出回原采样率。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SR = 16000
NFFT = 1024
HOP = 160
FMIN, FMAX = 60.0, 400.0
N_HARM = 26

MIN_SPEECH = 8            # 80 ms 最短语音段


# ---------------------------------------------------------------- helpers
def resample(x, sr_in, sr_out=SR):
    """FFT resampling with anti-alias truncation (speech band)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr_in) == int(sr_out) or len(x) == 0:
        return x.astype(np.float32)
    n_in = len(x)
    n_out = int(round(n_in * float(sr_out) / float(sr_in)))
    if n_out < 2:
        return np.zeros(2, dtype=np.float32)
    Xf = np.fft.rfft(x.astype(np.float64))
    Yf = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    m = min(len(Xf), len(Yf))
    Yf[:m] = Xf[:m]
    y = np.fft.irfft(Yf, n=n_out) * (float(n_out) / float(n_in))
    return y.astype(np.float32)


def _fit(y, n):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y))).astype(np.float32)


def _smooth(x, k, axis=-1):
    if k <= 1:
        return x
    p = k // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (p, p)
    xp = np.pad(x, pad, mode="edge")
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.apply_along_axis(
        lambda m: np.convolve(m, ker, mode="valid"), axis, xp).astype(np.float32)


# ---------------------------------------------------------------- pitch
def _window_acf(n=NFFT):
    w = np.hanning(n)
    r = np.fft.irfft(np.abs(np.fft.rfft(w)) ** 2, n=n)
    return (r / r[0]).astype(np.float64)


def _acf_voicing(M, sr=SR, n_fft=NFFT):
    """Absolute periodicity: energy-normalised autocorrelation peak."""
    P = (M * M).astype(np.float64)
    ac = np.fft.irfft(P, n=n_fft, axis=0)
    ac = ac / (ac[0:1] + 1e-12)
    acn = ac / np.maximum(_window_acf(n_fft)[:, None], 0.08)
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    peak = acn[lo:hi + 1].max(axis=0)
    return np.clip((peak - 0.20) / 0.55, 0.0, 1.0).astype(np.float32)


def _pitch_candidates(M, sr=SR, n_fft=NFFT, fmin=FMIN, fmax=FMAX, top=3):
    """Cepstral peaks: a chord shows several periods, a voice mostly one."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    prior = np.interp(freqs, [0, 90, 150, 400, 2000, 4000, 8000],
                      [0.02, 0.5, 1.0, 1.0, 0.9, 0.6, 0.0])
    cep = np.fft.irfft(np.log(M * prior[:, None] + 1e-5), n=n_fft, axis=0)
    cep = cep[: n_fft // 2]
    qmin, qmax = int(sr / fmax), int(sr / fmin)
    T = M.shape[1]
    cand = np.full((T, top), np.nan)
    score = np.zeros((T, top), np.float32)
    for t in range(T):
        seg = cep[qmin - 1:qmax + 2, t]
        idx = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
        if len(idx) == 0:
            continue
        med = float(np.median(seg))
        for j, i in enumerate(idx[np.argsort(seg[idx])[::-1][:top]]):
            a, b, c = seg[i - 1], seg[i], seg[i + 1]
            d = np.clip(0.5 * (a - c) / (a - 2 * b + c + 1e-12), -0.5, 0.5)
            cand[t, j] = sr / (qmin - 1 + i + d)
            score[t, j] = b - med
    mx = score.max(axis=1, keepdims=True)
    score = np.where(mx > 0, score / np.maximum(mx, 1e-9), 0.0)
    return cand, score


def _viterbi(cand, score, lo=70.0, hi=360.0, trans=3.0, ucost=0.35):
    """Pick a pitch track with a range prior and a smoothness penalty."""
    T, NS = cand.shape
    lf = np.log2(np.where(np.isnan(cand), 1.0, cand))
    valid = ~np.isnan(cand)
    pri = np.where((cand >= lo) & (cand <= hi), 0.0, 0.35)
    cost = np.full((T, NS + 1), -1e9)
    back = np.zeros((T, NS + 1), np.int8)
    for j in range(NS):
        if valid[0, j]:
            cost[0, j] = 2.5 * score[0, j] - pri[0, j]
    cost[0, NS] = 0.55
    for t in range(1, T):
        for j in range(NS + 1):
            if j < NS:
                if not valid[t, j]:
                    continue
                em = 2.5 * score[t, j] - pri[t, j]
            else:
                em = 0.55
            best, bi = -1e9, 0
            for i in range(NS + 1):
                if i < NS and j < NS:
                    tr = min(1.0, trans * abs(lf[t, j] - lf[t - 1, i]))
                elif i == NS and j == NS:
                    tr = 0.0
                else:
                    tr = ucost
                v = cost[t - 1, i] - tr
                if v > best:
                    best, bi = v, i
            cost[t, j] = best + em
            back[t, j] = bi
    s = int(np.argmax(cost[-1]))
    states = np.zeros(T, np.int8)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = back[t, s]
    f0 = np.zeros(T, np.float32)
    voicing = np.zeros(T, np.float32)
    for t in range(T):
        if states[t] < NS:
            f0[t] = cand[t, states[t]]
            voicing[t] = np.clip(score[t, states[t]], 0.0, 1.0)
    return f0, voicing


def _harmonic_comb(f0, voicing, freqs, n_harm=N_HARM):
    """Time-varying comb; between-teeth energy is left to the caller's floor."""
    H = np.zeros((len(freqs), len(f0)), np.float32)
    for t in range(len(f0)):
        if f0[t] < 50.0 or voicing[t] < 0.12:
            continue
        for k in range(1, n_harm + 1):
            fc = k * f0[t]
            if fc > 0.46 * SR:
                break
            sig = max(18.0, 0.022 * fc)
            H[:, t] = np.maximum(H[:, t], np.exp(-0.5 * ((freqs - fc) / sig) ** 2))
    return _smooth(H, 3, 1)


# ---------------------------------------------------------------- voice colours
def _lpc_formants(frame, sr=SR, order=16):
    """Levinson-Durbin + root solving; returns F1..F3 (Hz) or zeros."""
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 1e-12:
        return (0.0, 0.0, 0.0)
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0])
    for i in range(1, order + 1):
        prev = a[:i].copy()
        k = -(r[i] + float(np.dot(prev[1:], r[i - 1:0:-1]))) / (e + 1e-12)
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 1e-12:
            break
    try:
        roots = np.roots(a)
    except Exception:
        return (0.0, 0.0, 0.0)
    freqs = []
    for z in roots:
        if z.imag <= 1e-6:
            continue
        f = abs(np.angle(z)) * sr / (2.0 * np.pi)
        bw = -np.log(max(abs(z), 1e-9)) * sr / np.pi
        if 90.0 < f < 4000.0 and bw < 500.0:
            freqs.append(float(f))
    freqs.sort()
    vals = []
    for f in freqs:
        if not vals or f - vals[-1] > 120.0:
            vals.append(f)
        if len(vals) == 3:
            break
    while len(vals) < 3:
        vals.append(0.0)
    return tuple(vals[:3])


def _modulation(energy_db, sr_frames=100.0, lo=2.0, hi=8.0):
    """2–8 Hz syllable-rate modulation share of the frame-energy envelope."""
    e = np.asarray(energy_db, dtype=np.float64)
    e = e - e.mean()
    win = 64
    out = np.zeros(len(e), np.float32)
    for t in range(len(e)):
        seg = e[max(0, t - win // 2):t + win // 2]
        if len(seg) < 32:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1.0 / sr_frames)
        tot = S[(f > 0.5) & (f <= 20.0)].sum() + 1e-12
        out[t] = S[(f >= lo) & (f <= hi)].sum() / tot
    return out


# ---------------------------------------------------------------- analysis
def analyze(x, sr=SR):
    """Extract voice features. ``x`` may be any sample rate."""
    x16 = resample(x, sr, SR)
    X = C.stft(x16, NFFT, HOP)
    M = np.abs(X).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    T = M.shape[1]

    cand, cscore = _pitch_candidates(M)
    f0, _track_score = _viterbi(cand, cscore)
    voicing = _acf_voicing(M)
    H = _harmonic_comb(f0, voicing, freqs)

    npow = C.estimate_noise(M, 1.5)
    M2 = M * M
    band = (freqs > 80) & (freqs < 5000)
    harm = (M2 * H)[band].sum(axis=0) / (M2[band].sum(axis=0) + 1e-12)

    b_hi = (freqs > 2000) & (freqs < 7800)
    Mh = np.maximum(M[b_hi], 1e-8)
    flat = np.exp(np.mean(np.log(Mh), axis=0)) / (np.mean(Mh, axis=0) + 1e-12)

    snr_f = 10.0 * np.log10(
        np.maximum((M2 - npow).sum(axis=0), 1e-12) / (npow.sum(axis=0) + 1e-12))
    energy_db = 10.0 * np.log10(M2.sum(axis=0) + 1e-12)
    mod = _modulation(energy_db)

    # "Voice core": a frame must be harmonic *and* syllable-modulated to count
    # as a human voice.  Music is harmonic but not syllabic; noise is syllabic
    # but not harmonic.  Only the core is dilated into speech segments, so
    # unvoiced consonants ride along with their vowel neighbours.
    core = ((harm > 0.24) & (mod > 0.18) & (voicing > 0.25)
            & (snr_f > -6.0))
    vad = _segments(core, before=10, after=25)
    if core.mean() < 0.02:
        vad[:] = 0.0
    # Clip-level music guard: a tonal, non-syllabic clip contains no human
    # voice, so the whole clip goes to the floor (BGM-only video/audio).
    vm = voicing > 0.25
    if vm.sum() > 10:
        if (float(np.median(mod[vm])) < 0.10
                and float(np.median(flat[vm])) < 0.30
                and float(np.median(harm[vm])) > 0.45):
            vad[:] = 0.0
    vad = _smooth(vad[None, :], 7)[0]

    formants = []
    step = 5       # every 50 ms is enough for a formant track
    for t in range(0, T, step):
        if voicing[t] < 0.15:
            formants.append((0.0, 0.0, 0.0))
            continue
        i = t * HOP
        seg = x16[i:i + NFFT].astype(np.float64)
        if len(seg) < NFFT:
            seg = np.pad(seg, (0, NFFT - len(seg)))
        seg = seg * np.hamming(NFFT)
        formants.append(_lpc_formants(seg))

    voiced_frames = voicing > 0.25
    if voiced_frames.sum() >= 3:
        ltass = M[:, voiced_frames].mean(axis=1)
    else:
        ltass = M.mean(axis=1)
    ltass = _smooth(ltass[None, :], 9)[0]

    return dict(x16=x16, X=X, M=M, freqs=freqs, f0=f0, voicing=voicing, H=H,
                npow=npow, harm=harm, flat=flat, snr=snr_f,
                mod=mod, vad=vad, formants=formants,
                ltass=ltass, step=step, sr=SR)


def _segments(core, before=10, after=25):
    """Dilate voice-core frames into speech segments + drop short blips."""
    on = np.asarray(core, dtype=bool)
    T = len(on)
    out = np.zeros(T, np.float32)
    i = 0
    while i < T:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < T and on[j]:
            j += 1
        if j - i >= MIN_SPEECH:
            out[max(0, i - before):min(T, j + after)] = 1.0
        i = j
    return out


# ---------------------------------------------------------------- gate
def _sharpen_comb(H, sharp):
    """Steepen the comb edges (0 = off, 1 = hard); used for BGM removal."""
    if sharp <= 0.0:
        return H
    lo = 0.20 + 0.25 * float(sharp)
    hi = 0.50 + 0.20 * float(sharp)
    return np.clip((H - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def voice_gate(F, strength=0.8, floor_db=-40.0, between=0.9, sharp=0.0):
    """Return the voice gain on the analysis grid (F x T)."""
    vad = F["vad"]
    H = _sharpen_comb(F["H"], sharp)
    v = _smooth(F["voicing"][None, :], 5)[0]
    unvoiced = np.clip(1.0 - v, 0.0, 1.0)
    teeth = between + (1.0 - between) * H
    conf = vad[None, :] * (v[None, :] * teeth + unvoiced[None, :])
    fl = 10.0 ** (float(floor_db) / 20.0)
    G = (fl + (1.0 - fl) * conf) ** float(strength)
    return np.clip(G, 0.0, 1.0).astype(np.float32)


def _upsample_gate(Gv, F2, T2, hop_v=HOP, hop2=512):
    """Map a 1024/160 gate onto the NPSS 2048/512 grid."""
    F1, T1 = Gv.shape
    idx = np.clip(np.round(np.arange(T2) * hop2 / float(hop_v)).astype(int),
                  0, T1 - 1)
    Gt = Gv[:, idx]
    x1 = np.linspace(0.0, 1.0, F1)
    x2 = np.linspace(0.0, 1.0, F2)
    out = np.empty((F2, T2), dtype=np.float32)
    for t in range(T2):
        out[:, t] = np.interp(x2, x1, Gt[:, t])
    return out


def _to_output(y16, sr):
    if int(sr) == SR:
        return np.clip(y16, -1.0, 1.0).astype(np.float32)
    return np.clip(resample(y16, SR, sr), -1.0, 1.0).astype(np.float32)


def separate(x, sr=SR, strength=0.8, floor_db=-40.0, between=0.9,
             sharp=0.0, use_base=True, info_out=None):
    """Voice-feature separation.

    ``use_base`` keeps the NPSS multi-stat gate as the floor of speech quality
    and multiplies the voice gate on top.  Set False for a fast, gate-only pass.
    ``strength`` 0..1 scales the whole voice gate.
    """
    x16 = resample(x, sr, SR)
    F = analyze(x16, SR)
    Gv = voice_gate(F, strength=strength, floor_db=floor_db, between=between,
                    sharp=sharp)
    if use_base:
        import multi_stat as MS
        L = MS.loads(x16, SR)
        G = L.G * _upsample_gate(Gv, L.G.shape[0], L.G.shape[1])
        y16 = C.istft(G * L.X, len(x16), 2048, 512)
    else:
        y16 = C.istft(Gv * F["X"], len(x16), NFFT, HOP)
    y = _to_output(y16.astype(np.float32), sr)
    # 48k -> 16k -> 48k 的取整会差 1 个样本；输出长度必须和输入严格一致
    y = _fit(y, len(np.asarray(x, dtype=np.float32).reshape(-1)))
    if info_out is not None:
        v = F["voicing"]
        f0v = F["f0"][v > 0.25]
        info_out.clear()
        info_out.update({
            "speech_ratio": float((F["vad"] > 0.5).mean()),
            "voiced_ratio": float((v > 0.30).mean()),
            "f0_median": float(np.median(f0v)) if len(f0v) else 0.0,
            "f0_low": float(np.percentile(f0v, 10)) if len(f0v) else 0.0,
            "f0_high": float(np.percentile(f0v, 90)) if len(f0v) else 0.0,
            "formants": F["formants"],
            "floor_db": float(floor_db),
            "between": float(between),
            "strength": float(strength),
        })
    return y, F


def describe(F, info=None):
    """One-line Chinese summary for the GUI log."""
    v = F["voicing"]
    f0v = F["f0"][v > 0.25]
    f0 = float(np.median(f0v)) if len(f0v) else 0.0
    fms = [f for f in F["formants"] if f[0] > 0]
    f1 = float(np.median([f[0] for f in fms])) if fms else 0.0
    f2 = float(np.median([f[1] for f in fms])) if fms else 0.0
    return ("人声特征：F0 %.0f Hz，浊音 %.0f%%，语音段 %.0f%%，F1≈%.0f F2≈%.0f Hz"
            % (f0, 100.0 * (v > 0.30).mean(),
               100.0 * (F["vad"] > 0.5).mean(), f1, f2))
