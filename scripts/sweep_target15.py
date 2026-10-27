#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise-reduction / quality frontier: sweep the target and measure both sides.

For every clean/noisy pair this solves the dose for a ladder of targets and
records what each target actually buys:

    nr      : measured noise-floor drop (dB, blind)
    dsnr    : SNR change against the reference
    si_sdr  : SI-SDR against the reference

The result is a Pareto table: how much signal you pay per dB of noise removed,
and where 15 dB sits on that curve.

    python scripts/sweep_target15.py
"""

import argparse
import csv
import json
import os
import sys

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
TARGETS = [None, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]


def score(noisy, clean, y):
    n = min(len(y), len(clean))
    ref = np.asarray(clean, np.float32)[:n]
    out = np.asarray(y, np.float32)[:n]
    return dict(nr=round(T.nr_floor(noisy, y), 3),
                dsnr=round(T.snr(out, ref) - T.snr(noisy[:n], ref), 3),
                si_sdr=round(T.si_sdr(out, ref), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", action="store_true",
                    help="dose applied uniformly (I=0): the ablation frontier")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for tag in CASES:
        noisy, sr = C.read_wav(os.path.join(ROOT, "samples", "target15", tag + "_noisy.wav"))
        clean, _ = C.read_wav(os.path.join(ROOT, "samples", "target15", tag + "_clean.wav"))
        L = MS.loads(noisy, sr)
        npow = C.estimate_noise(L.M, 1.5)
        I, _, _, _, _ = FV.build(noisy, sr, L.M, npow, L.S, clean=clean,
                                 n_fft=MS.NFFT, hop=MS.HOP)
        if a.uniform:
            I = np.zeros_like(I)
        mask, _ = T.noise_frames(noisy)
        base = score(noisy, clean, C.istft(L.G * L.X, len(noisy), MS.NFFT, MS.HOP))
        rows.append(dict(case=tag, target="baseline", dose=0.0, hit=1, **base))
        for t in TARGETS:
            if t is None:
                continue
            sol = T.solve_dose(L.X, L.G, I, len(noisy), noisy, sr, t,
                               MS.NFFT, MS.HOP, mask)
            rows.append(dict(case=tag, target=t, dose=round(sol["dose"], 3),
                             hit=int(sol["hit"]), **score(noisy, clean, sol["y"])))
        print("%-24s done" % tag)

    out = a.out or os.path.join(
        ROOT, "target15_sweep_uniform.csv" if a.uniform else "target15_sweep.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # best target per case, by SI-SDR
    best = {}
    for tag in CASES:
        rs = [r for r in rows if r["case"] == tag and r["target"] != "baseline"]
        b = max(rs, key=lambda r: r["si_sdr"])
        best[tag] = dict(target=b["target"], si_sdr=b["si_sdr"], nr=b["nr"])
    with open(out.replace(".csv", "_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("%-24s %8s %8s %8s   %s" % ("case", "NR@15", "SI@15", "SI@best", "best target"))
    for tag in CASES:
        r15 = [r for r in rows if r["case"] == tag and r["target"] == 15.0][0]
        print("%-24s %8.2f %8.2f %8.2f   %.0f dB (NR %.1f)" % (
            tag, r15["nr"], r15["si_sdr"], best[tag]["si_sdr"],
            best[tag]["target"], best[tag]["nr"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
