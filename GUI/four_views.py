#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四视角不变特征：能量 / 频率 / 形态 / 结构。

The "定果寻因" pipeline fixes an *effect* (a required noise reduction, e.g.
15 dB) and then searches for a *cause* (the control that produces it).  The
cause space has an awkward property: almost any gain curve can hit a numeric
target by destroying the signal, so the target alone is not enough -- we need
to know **which time-frequency cells must survive**.

That is what this module answers, from four independent angles.  Each view
returns a soft score in [0, 1] per (f, t) cell:

    1 = the cell carries invariant structure -> protect it
    0 = the cell is expendable              -> let the controller eat it

    能量 (energy)    : the cell is far above the estimated noise floor and the
                       frame carries energy at all.
    频率 (frequency) : the cell sits on a band, or on a persistent spectral
                       peak, that stands out from that band's noise level.
    形态 (morphology): the local extremum *shape* of the spectrogram survives
                       random perturbation (the NPSS stability map S).
    结构 (structure) : the cell belongs to a long-range structure -- a harmonic
                       comb, a region-to-region topology, a transient onset.

The four maps are fused by a noisy-OR into one protect map ``I``.  The
controller applies its extra attenuation as ``G0 ** (1 + c * (1 - I))``: the
scalar ``c`` moves the noise floor, ``I`` decides where that movement lands.

``discover`` measures, for every named feature, how invariant it is (stability
across the multi-family perturbation bank: white, Student-t, impulsive, AM,
bursty, traffic) and how well it separates signal from noise (AUC), and turns
those measurements into the fusion weights.  Nothing is hand-tuned to a corpus:
the weights are measured on the input itself.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ helpers
def _sig(x, center=0.0, width=1.0):
    z = np.clip((np.asarray(x, dtype=np.float32) - center) / max(width, 1e-6),
                -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _db(p):
    return 10.0 * np.log10(np.maximum(np.asarray(p, np.float64), EPS)).astype(np.float32)


def band_index(freqs, n_bands=24, fmin=80.0, fmax=None):
    """Log-spaced band edges -> (band_of[F] in [-1..nb-1], nb)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if fmax is None or fmax > freqs[-1]:
        fmax = float(freqs[-1])
    lo = max(fmin, float(freqs[1]))
    edges = np.geomspace(lo, max(fmax, lo * 1.01), n_bands + 1)
    band_of = np.full(len(freqs), -1, dtype=np.int16)
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        band_of[m] = b
    band_of[freqs >= edges[-1]] = n_bands - 1
    return band_of, n_bands


def _band_power(P, band_of, nb):
    out = np.zeros((nb, P.shape[1]), dtype=np.float32)
    for b in range(nb):
        m = band_of == b
        if np.any(m):
            out[b] = P[m].sum(axis=0)
    return out


def _expand_bands(band_vals, band_of, n_bins):
    out = np.zeros((n_bins, band_vals.shape[1]), dtype=np.float32)
    for b in range(band_vals.shape[0]):
        m = band_of == b
        if np.any(m):
            out[m] = band_vals[b]
    valid = band_of >= 0
    if np.any(valid) and np.any(~valid):
        out[~valid] = out[valid].mean(axis=0)
    return out


def _median3(a):
    """Cheap 3x3 median (edge padded), chunked over time."""
    p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    W = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
    out = np.empty_like(a)
    T = a.shape[1]
    for t0 in range(0, T, 256):
        t1 = min(t0 + 256, T)
        out[:, t0:t1] = np.median(W[:, t0:t1], axis=(-2, -1))
    return out.astype(np.float32)


# ------------------------------------------------------------------ view 1
def energy_view(M, npow):
    """能量视角：cell SNR + 帧能量包络。"""
    m2 = np.asarray(M, dtype=np.float32) ** 2
    snr = _db(m2) - _db(npow)
    cell = _sig(snr, center=1.5, width=3.0)
    env = _db(m2.sum(axis=0))
    floor = float(np.percentile(env, 15))
    frame = _sig(env - floor, center=3.0, width=4.0)
    I = np.clip(0.65 * cell + 0.35 * frame[None, :], 0.0, 1.0)
    return I.astype(np.float32), {
        "cell_snr_db": snr,
        "frame_energy_db": env,
        "cell_invariance": cell,
    }


# ------------------------------------------------------------------ view 2
def freq_view(M, npow, freqs, n_bands=24):
    """频率视角：频带 SNR + 持续谱峰 + 低频保护。"""
    M = np.asarray(M, dtype=np.float32)
    band_of, nb = band_index(freqs, n_bands)
    E = _band_power(M * M, band_of, nb)
    N = _band_power(np.asarray(npow, dtype=np.float32), band_of, nb)
    bsnr = _db(E) - _db(N)
    Iband = _expand_bands(_sig(bsnr, center=3.0, width=3.0), band_of, M.shape[0])

    P = np.zeros_like(M, dtype=bool)
    P[1:-1, :] = (M[1:-1, :] >= M[:-2, :]) & (M[1:-1, :] > M[2:, :])
    cnt = np.zeros_like(M, dtype=np.float32)
    for dt in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dt == 0 and df == 0:
                continue
            cnt += np.roll(np.roll(P, dt, axis=1), df, axis=0).astype(np.float32)
    Ipeak = np.clip((cnt - 2.0) / 4.0, 0.0, 1.0)

    low = np.clip(1.0 - np.asarray(freqs, np.float32) / 250.0, 0.0, 1.0)[:, None]
    Ilow = 0.55 * np.broadcast_to(low, M.shape).astype(np.float32)
    # spectral flatness: tonal cells have low flatness, noise has high flatness
    P = (M.astype(np.float64) ** 2) + EPS
    flat = np.exp(np.mean(np.log(P), axis=0)) / (np.mean(P, axis=0) + EPS)
    flat = flat.astype(np.float32)
    Iflat = np.broadcast_to(_sig(0.32 - flat, center=0.0, width=0.08)[None, :], M.shape)
    I = np.maximum(np.maximum(np.maximum(Iband, Ipeak), Ilow), Iflat)
    return np.clip(I, 0.0, 1.0).astype(np.float32), {
        "band_snr_db": bsnr,
        "peak_persistence": Ipeak,
        "spectral_flatness": np.broadcast_to(flat[None, :], M.shape).copy(),
        "spectral_centroid_hz": _centroid(M, freqs),
        "band_invariance": Iband,
    }


def _centroid(M, freqs):
    de = M.sum(axis=0) + EPS
    return ((np.asarray(freqs, np.float32)[:, None] * M).sum(axis=0) / de).astype(np.float32)


# ------------------------------------------------------------------ view 3
def form_view(S, theta=0.6):
    """形态视角：NPSS 稳定性场 + 形态学闭运算补全。"""
    S = np.asarray(S, dtype=np.float32)
    lo, hi = np.percentile(S, 10), np.percentile(S, 95)
    if hi - lo < 0.02:
        lo, hi = float(S.min()), float(S.max())
    base = np.clip((S - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)
    piv = (S >= theta).astype(np.float32)
    dil = np.zeros_like(piv)
    for dt in (-1, 0, 1):
        for df in (-1, 0, 1):
            dil = np.maximum(dil, np.roll(np.roll(piv, dt, axis=1), df, axis=0))
    closed = np.minimum(dil, np.maximum(piv, 0.85 * base))
    # temporal persistence of pivots: long runs are structure, isolated points are detail
    run = np.zeros_like(S, dtype=np.float32)
    for t in range(1, S.shape[1]):
        run[:, t] = np.where(piv[:, t] > 0, run[:, t - 1] + 1.0, 0.0)
    Irun = np.clip(run / 8.0, 0.0, 1.0).astype(np.float32)
    I = np.maximum(np.maximum(base, closed), Irun)
    return np.clip(I, 0.0, 1.0).astype(np.float32), {
        "stability_S": S,
        "pivot_density": np.array([float(piv.mean())], dtype=np.float32),
        "stability_mean": np.array([float(S.mean())], dtype=np.float32),
        "stability_run": Irun,
        "form_invariance": base,
    }


# ------------------------------------------------------------------ view 4
def pitch_track(x, sr, n_fft=2048, hop=512, fmin=60.0, fmax=1200.0):
    """Normalised-autocorrelation F0 track on STFT-aligned frames."""
    x = np.asarray(x, dtype=np.float32)
    # frame grid identical to common.stft (padded by n_fft on both sides)
    T = (len(x) + n_fft) // hop + 1
    if T <= 0 or len(x) < 2:
        return np.zeros(1, np.float32), np.zeros(1, np.float32)
    idx = np.arange(T)[:, None] * hop - n_fft + np.arange(n_fft)[None, :]
    idx = np.clip(idx, 0, len(x) - 1)
    fr = x[idx].astype(np.float32)
    fr -= fr.mean(axis=1, keepdims=True)
    F = np.fft.rfft(fr, n=2 * n_fft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=2 * n_fft, axis=1)[:, :n_fft]
    ac /= (np.abs(ac[:, :1]) + EPS)
    lag_lo = int(max(2, np.floor(sr / fmax)))
    lag_hi = int(min(n_fft - 2, np.ceil(sr / fmin)))
    if lag_hi <= lag_lo + 1:
        return np.zeros(T, np.float32), np.zeros(T, np.float32)
    seg = ac[:, lag_lo:lag_hi]
    k = np.argmax(seg, axis=1)
    voicing = np.clip(seg[np.arange(T), k], 0.0, 1.0).astype(np.float32)
    f0 = (sr / (lag_lo + k).astype(np.float64)).astype(np.float32)
    f0 = np.where(voicing > 0.30, f0, 0.0).astype(np.float32)
    return f0, voicing


def _harmonic_comb(M, freqs, f0, voicing, n_harm=12, tol_semitone=0.45):
    F, T = M.shape
    comb = np.zeros((F, T), dtype=np.float32)
    tol = tol_semitone / 12.0
    for t in range(T):
        if f0[t] <= 0 or voicing[t] < 0.3:
            continue
        v = float(voicing[t])
        for k in range(1, n_harm + 1):
            fk = float(f0[t]) * k
            if fk >= freqs[-1]:
                break
            m = (freqs >= fk * 2.0 ** (-tol)) & (freqs <= fk * 2.0 ** (tol))
            if np.any(m):
                comb[m, t] = np.maximum(comb[m, t], v)
    return comb


def _region_weights(S, nr=16, bands=8):
    """Soft cross-region topology weight (self-contained copy of multi_stat's)."""
    F, T = S.shape
    R = int(np.ceil(T / float(nr)))
    pad = np.pad(S, ((0, 0), (0, R * nr - T)), mode="edge")
    blocks = pad.reshape(F, R, nr)
    edges = np.linspace(0, F, bands + 1).astype(int)
    feat = np.zeros((R, bands), dtype=np.float32)
    for b in range(bands):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        feat[:, b] = blocks[lo:hi].mean(axis=(0, 2))
    feat -= feat.mean(axis=1, keepdims=True)
    feat /= (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    deg = np.zeros(R, dtype=np.float32)
    for r in range(R):
        if r > 0:
            deg[r] += max(0.0, float(feat[r] @ feat[r - 1]))
        if r < R - 1:
            deg[r] += max(0.0, float(feat[r] @ feat[r + 1]))
    w = deg / (deg.max() + 1e-8)
    return np.repeat(w, nr)[:T].astype(np.float32)


def struct_view(x, M, S, freqs, sr, n_fft=2048, hop=512):
    """结构视角：谐波梳 + 跨区拓扑 + 瞬态起音。"""
    F, T = M.shape
    f0, voicing = pitch_track(x, sr, n_fft, hop)
    if len(f0) != T:
        f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
        voicing = np.pad(voicing, (0, max(0, T - len(voicing))))[:T]
    comb = _harmonic_comb(M, freqs, f0, voicing)

    rw = _region_weights(S) if T > 4 else np.zeros(T, np.float32)
    topo = np.tile((0.55 * rw)[None, :], (F, 1)).astype(np.float32)

    flux = np.zeros(T, dtype=np.float32)
    if T > 1:
        flux[1:] = np.maximum(0.0, M[:, 1:] - M[:, :-1]).sum(axis=0)
    onset = np.zeros((F, T), dtype=np.float32)
    if T > 8:
        med = float(np.median(flux))
        mad = float(np.median(np.abs(flux - med))) + EPS
        on = flux > med + 2.5 * 1.4826 * mad
        for shift in (-2, -1, 0, 1, 2):
            src = np.clip(np.arange(T) - shift, 0, T - 1)
            onset = np.maximum(onset, on[src][None, :].astype(np.float32) * 0.7)

    # harmonic recurrence: the same comb returns after 1-4 frames
    rec = np.zeros_like(comb)
    for lag in (1, 2, 3, 4):
        if T > lag:
            rec[:, lag:] = np.maximum(rec[:, lag:],
                                      np.minimum(comb[:, lag:], comb[:, :-lag]))
    I = np.maximum(np.maximum(np.maximum(comb, topo), onset), rec)
    return np.clip(I, 0.0, 1.0).astype(np.float32), {
        "f0_hz": f0,
        "voicing": voicing,
        "harmonic_comb": comb,
        "harmonic_recurrence": rec,
        "onset_mask": onset,
        "region_topo": topo,
    }


# ------------------------------------------------------------------ fusion
def fuse(views, weights):
    """Weighted consensus of the four angles.

    A noisy-OR saturates: with four maps that each sit near 0.6 in noise, the
    OR is already 0.97, and the dose loses all authority.  The weighted mean
    keeps a real 0/1 range -- a cell is protected when most angles agree, not
    when any single one happens to fire.
    """
    first = next(iter(views.values()))
    acc = np.zeros_like(first, dtype=np.float32)
    wsum = 0.0
    for k, v in views.items():
        w = float(weights.get(k, 0.8))
        acc += w * np.asarray(v, dtype=np.float32)
        wsum += w
    return np.clip(acc / max(wsum, 1e-6), 0.0, 1.0).astype(np.float32)


def smooth_protect(I, passes=1):
    out = np.asarray(I, dtype=np.float32)
    for _ in range(int(passes)):
        out = (0.5 * out + 0.5 * _median3(out)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ discovery
def _auc(score, label):
    """Rank-based AUC of `score` separating label==1 from label==0."""
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 < 8 or n0 < 8:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0 + EPS))


def _feature_stability(base, pert):
    """1 - normalised mean deviation between base and perturbed feature."""
    a = np.asarray(base, dtype=np.float64).ravel()
    b = np.asarray(pert, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return 0.5
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    scale = float(np.std(a) + np.std(b)) + EPS
    return float(np.clip(1.0 - np.mean(np.abs(a - b)) / scale, 0.0, 1.0))


def discover(x, sr, M, npow, S, clean=None, n_fft=2048, hop=512, K=4, seed=0):
    """Measure how invariant and how discriminative every named feature is.

    Returns ``(weights, table)``: ``weights`` feeds :func:`fuse`, ``table`` is
    the evidence used to justify them.
    """
    import common as C

    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr)).astype(np.float32)
    M = np.asarray(M, dtype=np.float32)
    if clean is not None and len(clean) >= len(x):
        Mc = np.abs(C.stft(np.asarray(clean, np.float32)[:len(x)], n_fft, hop))
        label = np.asarray(Mc, np.float32) > np.maximum(np.sqrt(npow) * 2.0, 1e-6)
    else:
        label = M > np.sqrt(np.maximum(npow, EPS)) * 2.0

    Ie, fe = energy_view(M, npow)
    If, ff = freq_view(M, npow, freqs)
    Im, fm = form_view(S)
    x = np.asarray(x, np.float32)
    Is, fs = struct_view(x, M, S, freqs, sr, n_fft, hop)

    rms = float(np.sqrt(np.mean(x ** 2))) + EPS
    import multi_stat as MS
    bank = MS.make_stats_bank(len(x), seed=int(seed) + 900)
    names = ["white", "student_t", "impulsive", "am", "bursty", "traffic"]
    levels = np.logspace(-1.0, 0.3, max(1, int(K)))
    pert = {"energy": [], "freq": [], "struct": []}
    for k in range(max(1, int(K))):
        nk = (bank[names[k % len(names)]].astype(np.float32)
              * rms * float(levels[k % len(levels)]))
        xk = (x + nk).astype(np.float32)
        Mk = np.abs(C.stft(xk, n_fft, hop)).astype(np.float32)
        npk = C.estimate_noise(Mk, 1.5)
        pert["energy"].append(energy_view(Mk, npk)[1])
        pert["freq"].append(freq_view(Mk, npk, freqs)[1])
        pert["struct"].append(struct_view(xk, Mk, S, freqs, sr, n_fft, hop)[1])

    def stab(view, name):
        if not pert[view]:
            return 0.5
        base = {"energy": fe, "freq": ff, "struct": fs}[view][name]
        return float(np.mean([_feature_stability(base, p[name]) for p in pert[view]]))

    band_of, _ = band_index(freqs, 24)
    band_map = _expand_bands(ff["band_snr_db"], band_of, M.shape[0])
    rows = [
        ("能量", "energy", "cell_snr_db", stab("energy", "cell_snr_db"),
         _auc(fe["cell_snr_db"], label), 1.0),
        ("能量", "energy", "frame_energy_db", stab("energy", "frame_energy_db"),
         _auc(np.broadcast_to(fe["frame_energy_db"][None, :], M.shape), label), 0.9),
        ("频率", "freq", "band_snr_db", stab("freq", "band_snr_db"),
         _auc(band_map, label), 1.0),
        ("频率", "freq", "peak_persistence", stab("freq", "peak_persistence"),
         _auc(ff["peak_persistence"], label), 0.9),
        ("频率", "freq", "spectral_flatness", stab("freq", "spectral_flatness"),
         _auc(ff["spectral_flatness"], label), 0.8),
        ("形态", "form", "stability_S", float(np.mean(fm["stability_S"])), 
         _auc(fm["stability_S"], label), 1.0),
        ("形态", "form", "pivot_density", float(fm["pivot_density"][0]),
         _auc(Im, label), 0.9),
        ("形态", "form", "stability_run", float(np.mean(fm["stability_run"])),
         _auc(fm["stability_run"], label), 0.8),
        ("结构", "struct", "harmonic_comb", stab("struct", "harmonic_comb"),
         _auc(fs["harmonic_comb"], label), 1.0),
        ("结构", "struct", "onset_protect", stab("struct", "onset_mask"),
         _auc(fs["onset_mask"], label), 0.7),
        ("结构", "struct", "harmonic_recurrence", stab("struct", "harmonic_recurrence"),
         _auc(fs["harmonic_recurrence"], label), 0.8),
    ]
    table = []
    for view_cn, key, name, st, auc, prior in rows:
        # AUC counts only above chance; stability counts twice for map features
        table.append(dict(view=view_cn, key=key, feature=name,
                          stability=float(st),
                          auc=float(auc),
                          prior=float(prior)))
    weights = {}
    for k in ("energy", "freq", "form", "struct"):
        rs = [r for r in table if r["key"] == k]
        disc = max(0.0, float(np.mean([r["auc"] for r in rs])) - 0.5) * 2.0
        stab_v = float(np.mean([r["stability"] for r in rs]))
        weights[k] = float(np.clip(0.40 + 0.55 * disc * (0.5 + 0.5 * stab_v),
                                   0.40, 0.95))
    return weights, table


def build(x, sr, M, npow, S, weights=None, clean=None, n_fft=2048, hop=512):
    """Compute the four view maps, the fused protect map and the evidence."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr)).astype(np.float32)
    Ie, fe = energy_view(M, npow)
    If, ff = freq_view(M, npow, freqs)
    Im, fm = form_view(S)
    Is, fs = struct_view(x, M, S, freqs, sr, n_fft, hop)
    views = {"energy": Ie, "freq": If, "form": Im, "struct": Is}
    table = []
    if weights is None:
        weights, table = discover(x, sr, M, npow, S, clean=clean,
                                  n_fft=n_fft, hop=hop)
    I = smooth_protect(fuse(views, weights), passes=1)
    feats = {}
    feats.update(fe)
    feats.update(ff)
    feats.update(fm)
    feats.update(fs)
    return I, views, weights, table, feats
