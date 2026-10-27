#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 定果寻因 against the fixed-threshold baseline.

For every clean/noisy pair it runs three methods on the *same* NPSS loads:

    baseline   : stock denoise_best (fixed thresholds) -- whatever dB it gives
    target15   : dose solved to 15 dB, attenuation shaped by the four views
    uniform15  : dose solved to 15 dB, attenuation applied uniformly (no views)

and reports the measured noise-floor drop, the SNR change, and the four view
preservation scores.  ``uniform15`` is the ablation: same target, no invariants.

    python scripts/eval_target15.py --target 15
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "GUI"))

import common as C
import four_views as FV
import multi_stat as MS
import target_nr as T

CASES = ["harmonic_white_+05", "harmonic_white_+00", "harmonic_pink_+05",
         "harmonic_traffic_+10", "music_white_+05", "music_pink_+00",
         "formant_white_+05", "formant_traffic_+05"]


def metrics(noisy, clean, y):
    n = min(len(y), len(clean))
    ref = np.asarray(clean, np.float32)[:n]
    out = np.asarray(y, np.float32)[:n]
    mref, _ = T.noise_frames(noisy, ref)
    vs = T.view_scores(ref, out, 16000)
    return dict(nr_blind=round(T.nr_floor(noisy, y), 3),
                nr_ref=round(T.nr_floor(noisy, y, ref, mask=mref), 3),
                dsnr=round(T.snr(out, ref) - T.snr(noisy[:n], ref), 3),
                si_sdr=round(T.si_sdr(out, ref), 3),
                energy=round(vs["scores"]["energy"], 3),
                freq=round(vs["scores"]["freq"], 3),
                form=round(vs["scores"]["form"], 3),
                struct=round(vs["scores"]["struct"], 3),
                composite=round(vs["composite"], 3))


def run_case(tag, target, outdir, save=False):
    noisy, sr = C.read_wav(os.path.join(ROOT, "samples", "target15", tag + "_noisy.wav"))
    clean, _ = C.read_wav(os.path.join(ROOT, "samples", "target15", tag + "_clean.wav"))
    rows = []
    t0 = time.time()
    L = MS.loads(noisy, sr)
    t_load = time.time() - t0

    y_base = C.istft(L.G * L.X, len(noisy), MS.NFFT, MS.HOP)
    rows.append(dict(case=tag, method="baseline", dose=0.0,
                     seconds=round(t_load, 2), **metrics(noisy, clean, y_base)))

    npow = C.estimate_noise(L.M, 1.5)
    I, views, weights, table, feats = FV.build(noisy, sr, L.M, npow, L.S,
                                               clean=clean, n_fft=MS.NFFT, hop=MS.HOP)
    mask, _ = T.noise_frames(noisy)
    t1 = time.time()
    sol = T.solve_dose(L.X, L.G, I, len(noisy), noisy, sr, target,
                       MS.NFFT, MS.HOP, mask)
    dt1 = time.time() - t1
    rows.append(dict(case=tag, method="target", dose=round(sol["dose"], 3),
                     seconds=round(t_load + dt1, 2),
                     **metrics(noisy, clean, sol["y"])))
    t2 = time.time()
    solu = T.solve_dose(L.X, L.G, np.zeros_like(I), len(noisy), noisy, sr, target,
                        MS.NFFT, MS.HOP, mask)
    dt2 = time.time() - t2
    rows.append(dict(case=tag, method="uniform", dose=round(solu["dose"], 3),
                     seconds=round(t_load + dt2, 2),
                     **metrics(noisy, clean, solu["y"])))

    if save and outdir:
        os.makedirs(outdir, exist_ok=True)
        C.write_wav(os.path.join(outdir, tag + "_baseline.wav"), y_base, sr)
        C.write_wav(os.path.join(outdir, tag + "_target.wav"), sol["y"], sr)
    meta = dict(case=tag, weights=weights, evidence=table,
                hit=sol["hit"], achieved=round(sol["value"], 3),
                dose=sol["dose"], load_seconds=round(t_load, 2))
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=15.0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "samples", "target15", "out"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "target15_metrics.csv"))
    ap.add_argument("--json", default=os.path.join(ROOT, "target15_evidence.json"))
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--cases", nargs="*", default=CASES)
    a = ap.parse_args()

    all_rows, metas = [], []
    for tag in a.cases:
        t0 = time.time()
        rows, meta = run_case(tag, a.target, a.outdir, save=a.save)
        all_rows += rows
        metas.append(meta)
        print("%-24s %5.1fs  " % (tag, time.time() - t0) +
              "  ".join("%s:%s/%s" % (r["method"][:5], r["nr_blind"], r["composite"])
                        for r in rows))

    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False, default=float)
    print("wrote", a.csv, "and", a.json)

    # summary
    for m in ("baseline", "target", "uniform"):
        rs = [r for r in all_rows if r["method"] == m]
        print("%-9s NR=%.2f±%.2f dSNR=%.2f comp=%.3f" % (
            m,
            np.mean([r["nr_blind"] for r in rs]), np.std([r["nr_blind"] for r in rs]),
            np.mean([r["dsnr"] for r in rs]),
            np.mean([r["composite"] for r in rs])))


if __name__ == "__main__":
    main()
