#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PESQ/STOI for classical baselines on the 30 held-out files."""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/Users/ma/Downloads/降噪"
CLEAN = os.path.join(BASE, "归档/旧测试/官方测试/val16/clean")
NOISY = os.path.join(BASE, "归档/旧测试/官方测试/val16/noisy")
sys.path.insert(0, os.path.join(HERE, "GUI"))
sys.path.insert(0, HERE)
import common as C
from metrics import pesq_stoi

if __name__ == "__main__":
    rows = []
    for f in sorted(x for x in os.listdir(CLEAN) if x.endswith(".wav")):
        clean, _ = C.read_wav(os.path.join(CLEAN, f))
        noisy, _ = C.read_wav(os.path.join(NOISY, f))
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
    print("\n30 means:")
    for k in ["noisy", "wiener", "specsub", "mmse_lsa", "omlsa"]:
        print("  %-9s PESQ %.3f  STOI %.4f" %
              (k, np.mean([r[k + "_pesq"] for r in rows]),
               np.mean([r[k + "_stoi"] for r in rows])))
