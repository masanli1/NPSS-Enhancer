#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标降噪（定果寻因）：把噪声底压到指定 dB，在四视角不变特征约束下反解剂量。

This is *not* a separate program: it is the optional last step of
``multi_stat.denoise_best``.  That function already builds the NPSS gate
``G0 = 0.5 * (G_single + G_gmin)``; this module either returns that gate
untouched (target off) or shapes it with a solved dose (target on), and the
same caller performs the single ISTFT.


Classical denoising fixes the *cause* (thresholds, floors, passes) and then
reports whatever effect comes out.  This module inverts the direction:

    果 (effect) : the noise floor must drop by exactly T dB (default 15).
    因 (cause)  : one scalar dose ``c`` applied to the existing NPSS gain,
                  ``G = G0 ** (1 + c * (1 - I))``,
                  where ``I`` is the four-view protect map from ``four_views``.

``c`` is solved by bracketing a measured response curve and bisecting the
bracket that straddles the target -- never by guessing a parameter.  ``I``
decides where the dose is allowed to land, so the target is met by eating noise
cells rather than signal cells.  The four views are re-scored on the output, and
a target outside the reachable range is reported (``hit=False`` plus the
closest achievable value) instead of being silently replaced by a different
number.

Measurement modes:

    ``nr_floor``  noise-floor attenuation in the file's own quiet frames.
                  Blind (no clean reference needed) -- the default, because a
                  real recording has no clean reference.
    ``dsnr``      SNR improvement against a clean reference.  Diagnostic only:
                  it saturates long before the noise floor does.
    ``snr_out``   absolute output SNR against a clean reference.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common as C
import four_views as FV

EPS = 1e-12
GATE_MIN = 1e-4


# ------------------------------------------------------------------ framing
def frames(v, n=512, h=160):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    T = 1 + (len(v) - n) // h
    idx = np.arange(T)[:, None] * h + np.arange(n)[None, :]
    return v[idx]


def frame_db(v, n=512, h=160):
    fr = frames(v, n, h).astype(np.float64)
    return 10.0 * np.log10(np.mean(fr ** 2, axis=1) + 1e-20)


def noise_frames(x, clean=None, n=512, h=160, top_db=30.0, frac=0.20):
    """Frame indices used as the noise floor.

    With a reference: frames more than ``top_db`` below the reference peak.
    Blind: the quietest ``frac`` of frames of the input itself.  Both are
    derived from the *input* only, so the same mask scores x and y.
    """
    xdb = frame_db(x, n, h)
    if clean is not None and len(clean) >= len(x):
        cdb = frame_db(np.asarray(clean, np.float32)[:len(x)], n, h)
        mask = cdb < float(np.percentile(cdb, 99)) - float(top_db)
        if mask.sum() >= 4:
            return mask, "reference"
    k = max(4, int(float(frac) * len(xdb)))
    k = min(k, len(xdb))
    mask = np.zeros(len(xdb), dtype=bool)
    mask[np.argsort(xdb)[:k]] = True
    return mask, "blind"


def nr_floor(x, y, clean=None, n=512, h=160, mask=None):
    """Noise-floor attenuation in dB, measured as a median level drop.

    The mean of the residual power is dominated by the few frames the gate
    leaks through, so a power average under-reports the attenuation and is
    unstable under mask changes.  The median frame level over the quiet frames
    is robust to that tail, monotone in the applied dose, and agrees within
    ~0.2 dB between the blind and the reference mask (measured on the demo).
    """
    if mask is None:
        mask, _ = noise_frames(x, clean, n, h)
    ex = frame_db(x, n, h)[mask]
    ey = frame_db(y, n, h)[mask]
    return float(np.clip(float(np.median(ex)) - float(np.median(ey)), -60.0, 120.0))


def snr(est, ref):
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    return 10.0 * np.log10(np.sum(ref ** 2) / (np.sum((est - ref) ** 2) + EPS))


def si_sdr(est, ref):
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    t = a * ref
    return 10.0 * np.log10(np.sum(t * t) / (np.sum((est - t) ** 2) + EPS))


# ------------------------------------------------------------------ view scores
def _corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 3:
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS
    return float(np.clip(np.sum(a * b) / d, -1.0, 1.0))


def _band_db(v, sr, n_fft=1024, nb=24):
    v = np.asarray(v, np.float32)
    n = min(len(v), n_fft)
    if n < 8:
        return np.zeros(nb)
    X = np.abs(np.fft.rfft(v[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.geomspace(max(60.0, f[1]), f[-1] * 0.98, nb + 1)
    out = np.empty(nb)
    for b in range(nb):
        m = (f >= edges[b]) & (f < edges[b + 1])
        out[b] = float(np.sum(X[m] ** 2) + EPS)
    return 10.0 * np.log10(out + EPS)


def _envelope(v, sr, smooth_ms=20.0):
    v = np.abs(np.asarray(v, np.float32))
    n = max(1, int(sr * smooth_ms / 1000.0))
    return np.convolve(v, np.ones(n, np.float32) / n, mode="same")


def view_scores(clean, y, sr):
    """Per-view preservation scores for the four angles, in [0, 1]."""
    clean = np.asarray(clean, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = min(len(clean), len(y))
    if n < 64:
        return {"scores": {k: 0.0 for k in ("energy", "freq", "form", "struct")},
                "detail": {}, "composite": 0.0}
    c, d = clean[:n], y[:n]

    # 能量：包络相关 + 增益偏差
    ec, ed = frame_db(c, 512, 160), frame_db(d, 512, 160)
    act = ec > float(np.percentile(ec, 99)) - 40.0
    if act.sum() < 4:
        act = np.ones_like(act, bool)
    r_en = _corr(ec[act], ed[act])
    drms = 10.0 * np.log10((np.mean(d ** 2) + EPS) / (np.mean(c ** 2) + EPS))
    s_energy = float(np.clip(r_en, 0, 1) * np.exp(-abs(drms) / 6.0))

    # 频率：频带形状（去掉整体增益后）相关 + 平均偏差
    bc, bd = _band_db(c, sr), _band_db(d, sr)
    keep = bc > bc.max() - 45.0
    if keep.sum() >= 3:
        bcn = bc - float(np.mean(bc[keep]))
        bdn = bd - float(np.mean(bd[keep]))
        r_f = _corr(bcn[keep], bdn[keep])
        mad = float(np.mean(np.abs(bcn[keep] - bdn[keep])))
    else:
        r_f, mad = 1.0, 0.0
    s_freq = float(np.clip(r_f, 0, 1) * np.exp(-mad / 6.0))

    # 形态：波形包络相关 + 峰均比 / 峭度漂移
    r_m = _corr(_envelope(c, sr), _envelope(d, sr))
    crest_c = float(np.max(np.abs(c)) / (np.sqrt(np.mean(c ** 2)) + EPS))
    crest_d = float(np.max(np.abs(d)) / (np.sqrt(np.mean(d ** 2)) + EPS))
    dcrest = 20.0 * np.log10((crest_d + EPS) / (crest_c + EPS))
    kc = float(np.mean(c.astype(np.float64) ** 4) / (np.mean(c.astype(np.float64) ** 2) ** 2 + EPS))
    kd = float(np.mean(d.astype(np.float64) ** 4) / (np.mean(d.astype(np.float64) ** 2) ** 2 + EPS))
    dkurt = float(np.log10((kd + EPS) / (kc + EPS)))
    s_form = float(np.clip(r_m, 0, 1) * np.exp(-abs(dcrest) / 8.0)
                   * np.exp(-abs(dkurt)))

    # 结构：基频一致率 + 谐噪比漂移
    f0c, vc = FV.pitch_track(c, sr, 2048, 512)
    f0d, vd = FV.pitch_track(d, sr, 2048, 512)
    m = min(len(f0c), len(f0d))
    f0c, vc, f0d = f0c[:m], vc[:m], f0d[:m]
    voiced = (f0c > 0) & (vc > 0.35)
    if voiced.sum() >= 3:
        hit = np.abs(np.log2((f0d[voiced] + EPS) / (f0c[voiced] + EPS))) < (1.0 / 12.0)
        pitch_acc = float(hit.mean())
    else:
        pitch_acc = 1.0
    hnr_c = float(np.mean(vc[vc > 0])) if np.any(vc > 0) else 0.0
    hnr_d = float(np.mean(vd[vd > 0])) if np.any(vd > 0) else 0.0
    s_struct = float(pitch_acc * np.exp(-abs(hnr_c - hnr_d) / 0.35))

    scores = {"energy": float(np.clip(s_energy, 0, 1)),
              "freq": float(np.clip(s_freq, 0, 1)),
              "form": float(np.clip(s_form, 0, 1)),
              "struct": float(np.clip(s_struct, 0, 1))}
    detail = {"envelope_corr": round(r_en, 4), "gain_db": round(drms, 3),
              "band_corr": round(r_f, 4), "band_mad_db": round(mad, 3),
              "env_corr": round(r_m, 4), "crest_db": round(dcrest, 3),
              "kurt_log10": round(dkurt, 4), "pitch_acc": round(pitch_acc, 4),
              "hnr_clean": round(hnr_c, 4), "hnr_out": round(hnr_d, 4)}
    geo = float(np.exp(np.mean(np.log(np.maximum(list(scores.values()), 1e-3)))))
    return {"scores": scores, "detail": detail, "composite": round(geo, 4)}


# ------------------------------------------------------------------ the solver
def apply_dose(G0, I, c):
    """G = G0 ** (1 + c (1 - I)): scalar dose, shaped by the protect map."""
    G = np.clip(np.power(np.maximum(G0, 1e-6), 1.0 + float(c) * (1.0 - I)),
                GATE_MIN, 1.0)
    return G.astype(np.float32)


def solve_dose(X, G0, I, n, x, sr, target_db, n_fft, hop, mask,
               metric="nr_floor", clean=None, tol=0.03, max_iter=24,
               verbose=False):
    """Bisect the dose until the *measured* value equals the target.

    The response is scanned on a fixed ladder of doses and the bracket that
    straddles the target is bisected, so the solver does not need to assume in
    which direction the metric moves with ``c`` (``nr_floor`` rises with the
    dose; ``dsnr`` can fall).  If the target is outside the reachable range the
    closest achievable point is returned with ``hit=False``.
    """
    cache = {}

    def evaluate(c):
        key = round(float(c), 6)
        if key not in cache:
            G = apply_dose(G0, I, c)
            y = C.istft(G * X, n, n_fft, hop)
            if metric == "dsnr" and clean is not None:
                val = snr(y, clean) - snr(x, clean)
            elif metric == "snr_out" and clean is not None:
                val = snr(y, clean)
            elif metric == "si_sdr_gain" and clean is not None:
                val = si_sdr(y, clean) - si_sdr(x, clean)
            else:
                val = nr_floor(x, y, clean, mask=mask)
            cache[key] = (float(val), y, G)
        return cache[key]

    ladder = [-0.999, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 26.0]
    vals = [evaluate(c)[0] for c in ladder]

    idx = None
    for i in range(len(ladder) - 1):
        d0, d1 = vals[i] - target_db, vals[i + 1] - target_db
        if d0 == 0.0:
            val, y, G = evaluate(ladder[i])
            return dict(dose=ladder[i], value=val, y=y, G=G, hit=True,
                        evaluations=len(cache))
        if d0 * d1 <= 0.0:
            idx = i
            break
    if idx is None:
        j = int(np.argmin(np.abs(np.asarray(vals) - target_db)))
        val, y, G = evaluate(ladder[j])
        return dict(dose=ladder[j], value=val, y=y, G=G, hit=False,
                    evaluations=len(cache),
                    note="target outside the reachable range: closest %.2f dB"
                         " at c=%.2f" % (val, ladder[j]))

    a, b = ladder[idx], ladder[idx + 1]
    va, vb = vals[idx], vals[idx + 1]
    it = 0
    while it < max_iter and abs(vb - va) > tol:
        mid = 0.5 * (a + b)
        vm = evaluate(mid)[0]
        if verbose:
            print("    c=%.3f -> %.3f" % (mid, vm))
        if (vm - target_db) * (va - target_db) <= 0.0:
            b, vb = mid, vm
        else:
            a, va = mid, vm
        it += 1
    c = 0.5 * (a + b)
    val, y, G = evaluate(c)
    return dict(dose=float(c), value=float(val), y=y, G=G, hit=True,
                iters=it, evaluations=len(cache))


# ------------------------------------------------------------------ pipeline
def protect(x, sr, L, weights=None, clean=None, n_fft=None, hop=None):
    """Build the four-view protect map for a set of NPSS loads."""
    import multi_stat as MS
    n_fft = n_fft or MS.NFFT
    hop = hop or MS.HOP
    npow = C.estimate_noise(L.M, 1.5)
    return FV.build(x, sr, L.M, npow, L.S, weights=weights,
                    clean=clean, n_fft=n_fft, hop=hop)


def shape_gate(L, x, sr, target_db=None, dose=None, weights=None, clean=None,
               verbose=False, metric="nr_floor"):
    """Fold the target dose into the NPSS gate.  Returns ``(G, info)``.

    ``dose=None`` solves for ``target_db``; a given ``dose`` is applied as-is
    (that is how every chunk after the calibration slice reuses one dose).
    The caller does the ISTFT, so the target is a modification of the existing
    gain, not a second denoiser.
    """
    import multi_stat as MS
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n_fft, hop = MS.NFFT, MS.HOP
    t0 = time.time()
    I, views, weights, table, feats = protect(x, sr, L, weights=weights,
                                              clean=clean)
    mask, mask_src = noise_frames(x, clean)
    if metric in ("dsnr", "snr_out", "si_sdr_gain") and clean is None:
        raise ValueError("metric=%s needs a clean reference" % metric)
    if dose is None:
        if target_db is None:
            raise ValueError("shape_gate needs target_db or dose")
        sol = solve_dose(L.X, L.G, I, len(x), x, sr, float(target_db),
                         n_fft, hop, mask, metric=metric,
                         clean=clean if metric != "nr_floor" else None,
                         verbose=verbose)
        target = float(target_db)
    else:
        G = apply_dose(L.G, I, float(dose))
        y0 = C.istft(G * L.X, len(x), n_fft, hop)
        sol = dict(dose=float(dose), value=nr_floor(x, y0, mask=mask),
                   G=G, hit=True)
        target = float(target_db) if target_db is not None else None

    G = sol["G"]
    y = C.istft(G * L.X, len(x), n_fft, hop)
    info = dict(target_db=target, achieved_db=round(float(sol["value"]), 3),
                dose=round(float(sol["dose"]), 4), hit=bool(sol["hit"]),
                metric=metric, mask_source=mask_src,
                noise_frames=int(mask.sum()), frames=int(len(mask)),
                weights={k: round(float(v), 3) for k, v in weights.items()},
                evidence=table,
                nr_floor_db=round(float(nr_floor(x, y, clean, mask=mask)), 3),
                seconds=round(time.time() - t0, 2))
    if "note" in sol:
        info["note"] = sol["note"]
    if clean is not None:
        n = min(len(y), len(clean))
        ref = np.asarray(clean, np.float32)[:n]
        info["snr_in_db"] = round(float(snr(x[:n], ref)), 3)
        info["snr_out_db"] = round(float(snr(y[:n], ref)), 3)
        info["dsnr_db"] = round(info["snr_out_db"] - info["snr_in_db"], 3)
        info["si_sdr_db"] = round(float(si_sdr(y[:n], ref)), 3)
        info["view_scores"] = view_scores(ref, y[:n], sr)
        info["views"] = views
    return G, info


def calibrate(x, sr, target_db=15.0, seed=7, fast=False, low_floor=None,
              verbose=False, metric="nr_floor", clean=None):
    """One-slice calibration: ``(dose, weights, info)`` for the whole file."""
    import multi_stat as MS
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    L = MS.loads(x, sr, seed=seed, fast=fast, low_floor=low_floor)
    G, info = shape_gate(L, x, sr, target_db=target_db, verbose=verbose,
                         metric=metric, clean=clean)
    return float(info["dose"]), dict(info["weights"]), info


# ------------------------------------------------------------------ CLI
def main(argv=None):
    """Single-file CLI: same path the GUI/CLI use, so numbers stay comparable."""
    import argparse
    import json
    import engine

    ap = argparse.ArgumentParser(
        description="目标降噪（定果寻因）：把噪声底压到指定 dB")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--target", type=float, default=15.0, help="目标降噪量 dB")
    ap.add_argument("--clean", default=None, help="干净参考（可选，仅用于评估）")
    ap.add_argument("--metric", default="nr_floor",
                    choices=["nr_floor", "dsnr", "snr_out", "si_sdr_gain"],
                    help="目标量：底噪降幅 / ΔSNR / 输出 SNR / ΔSI-SDR")
    ap.add_argument("--json", default=None, help="把 info 写成 json")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)

    x, sr = C.read_wav(a.input)
    info = {}
    if a.metric != "nr_floor":
        if not a.clean:
            raise SystemExit("--metric %s 需要 --clean 干净参考" % a.metric)
        clean, _ = C.read_wav(a.clean)
        import multi_stat as MS
        L = MS.loads(x, sr, seed=7, fast=a.fast)
        G, info = shape_gate(L, x, sr, target_db=float(a.target),
                             clean=clean, metric=a.metric)
        y = C.istft(G * L.X, len(x), MS.NFFT, MS.HOP)
    else:
        y = engine.denoise_array(x, sr, content_mode="music", multi_stat=True,
                                 workers=1, chunk_sec=20, fast=a.fast,
                                 target_nr_db=float(a.target), info_out=info)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.99
    C.write_wav(a.output, y, sr)

    out = {k: v for k, v in info.items() if k not in ("views", "calibration")}
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    if "achieved_db" in info:
        print("target %.2f dB -> achieved %.2f dB (dose c=%.3f, %s)" % (
            float(info.get("target_db") or 0.0), float(info["achieved_db"]),
            float(info.get("dose") or 0.0), "hit" if info.get("hit") else "MISS"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
