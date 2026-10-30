#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音色特征提取与转换（纯 numpy，无模型、无训练）。

分析参考音色：谱包络（倒谱）、基频分布、共振峰、节奏（起音/停顿）
转换一段源朗读：保内容与时长，把音高与谱包络换成参考音色。
"""
from __future__ import annotations

import numpy as np

import common as C

NFFT = 1024
HOP = 256


def _frames(x):
    return C.stft(x, NFFT, HOP)


def _env(mag, order=32):
    logm = np.log(mag + 1e-8)
    cep = np.fft.irfft(logm, axis=0)
    cep[order:] = 0.0
    return np.exp(np.fft.rfft(cep, axis=0).real)


def f0_track(mag, sr, fmin=60, fmax=400):
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    lo, hi = np.searchsorted(freqs, fmin), np.searchsorted(freqs, fmax)
    out = np.zeros(mag.shape[1], np.float32)
    for t in range(mag.shape[1]):
        m = mag[:, t]
        if m.sum() < 1e-6:
            continue
        h = m.copy()
        for k in (2, 3, 4):
            d = m[::k]
            h[:len(d)] *= d
        out[t] = freqs[lo + np.argmax(h[lo:hi])] if hi > lo else 0.0
    return out


def analyze(x, sr):
    mag = np.abs(_frames(x))
    env = _env(mag)
    e = mag.sum(0)
    loud = [t for t in range(mag.shape[1]) if e[t] > np.percentile(e, 55)]
    f0 = f0_track(mag, sr)
    v = f0[f0 > 0]
    lt = env[:, loud].mean(1) if loud else env.mean(1)
    # 共振峰：谱包络的前几个峰
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    logl = 20 * np.log10(lt / lt.max() + 1e-9)
    peaks = []
    for i in range(2, len(logl) - 2):
        if freqs[i] > 4000:
            break
        if logl[i] > logl[i-1] and logl[i] >= logl[i+1] and logl[i] > -30:
            if not peaks or freqs[i] - peaks[-1][0] > 150:
                peaks.append((float(freqs[i]), float(logl[i])))
    # 节奏
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(0)
    thr = flux.mean() + 0.8 * flux.std()
    onsets = np.where((flux[:-1] < thr) & (flux[1:] >= thr))[0]
    rms = np.sqrt((x[:len(x)//HOP*HOP].reshape(-1, HOP) ** 2).mean(1))
    pause = float((rms < 0.12 * rms.max()).mean()) if len(rms) else 0.0
    return dict(
        f0_median=float(np.median(v)) if len(v) else 0.0,
        f0_low=float(np.percentile(v, 10)) if len(v) else 0.0,
        f0_high=float(np.percentile(v, 90)) if len(v) else 0.0,
        envelope=lt.astype(np.float32),
        formants=peaks[:4],
        rate=float(len(onsets) / (len(x) / sr)) if len(x) else 0.0,
        pause=pause,
        centroid=float((lt * freqs).sum() / (lt.sum() + 1e-9)),
        duration=len(x) / sr,
    )


def _wsola(x, stretch, sr, frame=640, search=200):
    """保持音高、改变时长：stretch = 输出长度 / 输入长度。"""
    if abs(stretch - 1.0) < 1e-3:
        return x
    out = []
    pos = 0.0
    hop = frame // 2
    win = np.hanning(frame).astype(np.float32)
    n = len(x)
    while pos + frame < n:
        i = int(pos)
        # 在附近搜索最相似的帧（保持波形连续）
        best, boff = -1e9, 0
        for off in range(-search, search + 1, 8):
            j = i + off
            if j < 0 or j + frame > n:
                continue
            seg = x[j:j+frame]
            score = float(np.dot(seg, win))
            if score > best:
                best, boff = score, off
        seg = x[i+boff:i+boff+frame] * win
        out.append(seg)
        pos += hop / stretch
    if not out:
        return x
    y = np.zeros(int(len(out) * hop) + frame, np.float32)
    for k, seg in enumerate(out):
        y[k*hop:k*hop+frame] += seg
    env = np.zeros_like(y)
    for k in range(len(out)):
        env[k*hop:k*hop+frame] += win
    return (y / np.maximum(env, 1e-6)).astype(np.float32)


def shift_pitch(x, factor, sr):
    """变调不变速：先重采样变调，再用 WSOLA 恢复时长。"""
    if abs(factor - 1.0) < 1e-2:
        return x
    idx = np.arange(0, len(x) - 1, factor)
    y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return _wsola(y, factor, sr)


def convert(ref_path, src_path, out_path, strength=1.0, match_pitch=True):
    ref, sr = C.read_wav(ref_path)
    src, sr2 = C.read_wav(src_path)
    if sr2 != sr:
        idx = np.arange(0, len(src) - 1, sr2 / sr)
        src = np.interp(idx, np.arange(len(src)), src).astype(np.float32)
    A = analyze(ref, sr)
    B = analyze(src, sr)

    # 1) 音高
    y = src
    if match_pitch and B["f0_median"] > 0 and A["f0_median"] > 0:
        factor = float(np.clip(A["f0_median"] / B["f0_median"], 0.5, 2.0))
        y = shift_pitch(y, factor, sr)

    # 2) 谱倾斜匹配：用两者的长期平均谱做一条平滑 EQ
    X = _frames(y)
    mag = np.abs(X)
    e = mag.sum(0)
    loud = [t for t in range(mag.shape[1]) if e[t] > np.percentile(e, 55)]
    Es = _env(mag)
    src_lt = Es[:, loud].mean(1) if loud else Es.mean(1)
    tgt_lt = A["envelope"]
    ratio = np.clip((tgt_lt / (src_lt + 1e-9)) ** strength,
                    10 ** (-24 / 20), 10 ** (24 / 20))
    k = int(41)
    ratio = np.convolve(ratio, np.ones(k) / k, mode="same")
    ratio[:10] = 1.0
    Y = X * ratio[:, None]
    out = C.istft(Y, len(y), NFFT, HOP)
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.9
    C.write_wav(out_path, out.astype(np.float32), sr)
    return out_path, A, B
