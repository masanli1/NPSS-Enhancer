#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""PESQ/STOI for classical baselines on a clean/noisy test set."""
import os, sys, csv, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "GUI"))
import common as C
from metrics import pesq_stoi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", default="data/val16/clean")
    p.add_argument("--noisy", default="data/val16/noisy")
    p.add_argument("--out", default="results_pesq_stoi_baselines.csv")
    a = p.parse_args()
    clean_dir = os.path.abspath(a.clean)
    noisy_dir = os.path.abspath(a.noisy)
    rows = []
    for f in sorted(x for x in os.listdir(clean_dir) if x.endswith(".wav")):
        clean, _ = C.read_wav(os.path.join(clean_dir, f))
        noisy, _ = C.read_wav(os.path.join(noisy_dir, f))
        n = min(len(clean), len(noisy)); clean, noisy = clean[:n], noisy[:n]
        X, M, gb, Y0, R, se = C.prepare(noisy, 512, 128, 1.5, 0.05, "wiener")
        npow = C.estimate_noise(M, 1.5)
        outs = {"noisy": noisy, "wiener": C.istft(gb * X, n, 512, 128)}
        for mode in ["specsub", "mmse_lsa", "omlsa"]:
            g = C.backend_gain(M, mode, 0.05, 1.5, npow)
            outs[mode] = C.istft(g * X, n, 512, 128)
        row = {"file": f}
        for k, y in outs.items():
            m = min(len(y), n)
            row[k + "_pesq"], row[k + "_stoi"] = pesq_stoi(clean, y[:m], 16000)
        rows.append(row)
        print(f, "ok", flush=True)
    with open(os.path.abspath(a.out), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nmeans:")
    for k in ["noisy", "wiener", "specsub", "mmse_lsa", "omlsa"]:
        print("  %-9s PESQ %.3f  STOI %.4f" %
              (k, np.mean([r[k + "_pesq"] for r in rows]),
               np.mean([r[k + "_stoi"] for r in rows])))


if __name__ == "__main__":
    main()
