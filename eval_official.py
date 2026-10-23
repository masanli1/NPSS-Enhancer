#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate PESQ/STOI on the 30 official held-out files."""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/Users/ma/Downloads/降噪"
CLEAN = os.path.join(BASE, "归档/旧测试/官方测试/val16/clean")
NOISY = os.path.join(BASE, "归档/旧测试/官方测试/val16/noisy")
sys.path.insert(0, os.path.join(HERE, "GUI"))
sys.path.insert(0, HERE)
import common as C
import engine
from metrics import pesq_stoi


def specsub(x):
    X, M, gb, Y0, R, se = C.prepare(x, 512, 128, 1.5, 0.05, "specsub")
    return C.istft(gb * X, len(x), 512, 128)


if __name__ == "__main__":
    rows = []
    for f in sorted(x for x in os.listdir(CLEAN) if x.endswith(".wav")):
        clean, _ = C.read_wav(os.path.join(CLEAN, f))
        noisy, _ = C.read_wav(os.path.join(NOISY, f))
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        outs = {
            "noisy": noisy,
            "specsub": specsub(noisy),
            "npss": engine.denoise_array(noisy, 16000, content_mode="music",
                                         complement=False, multi_stat=False,
                                         workers=1, chunk_sec=20),
            "comp_k2": engine.denoise_array(noisy, 16000, content_mode="music",
                                            complement=True, multi_stat=False,
                                            loops=2, workers=1, chunk_sec=20),
            "comp_k5": engine.denoise_array(noisy, 16000, content_mode="music",
                                            complement=True, multi_stat=False,
                                            loops=5, workers=1, chunk_sec=20),
            "voice_k2": engine.denoise_array(noisy, 16000, content_mode="voice",
                                             complement=True, multi_stat=False,
                                             loops=2, workers=1, chunk_sec=20),
            "voice_k5": engine.denoise_array(noisy, 16000, content_mode="voice",
                                             complement=True, multi_stat=False,
                                             loops=5, workers=1, chunk_sec=20),
            "multi": engine.denoise_array(noisy, 16000, content_mode="music",
                                          complement=True, multi_stat=True,
                                          workers=1, chunk_sec=20),
            "multi_voice": engine.denoise_array(noisy, 16000, content_mode="voice",
                                                complement=True, multi_stat=True,
                                                workers=1, chunk_sec=20),
        }
        row = {"file": f}
        for k, y in outs.items():
            m = min(len(y), n)
            row[k + "_pesq"], row[k + "_stoi"] = pesq_stoi(clean, y[:m], 16000)
        rows.append(row)
        print(f, "ok", flush=True)
    with open(os.path.join(HERE, "results_pesq_stoi.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n30 means:")
    for k in ["noisy", "specsub", "npss", "comp_k2", "comp_k5", "voice_k2", "voice_k5",
              "multi", "multi_voice"]:
        print("  %-9s PESQ %.3f  STOI %.4f" %
              (k, np.mean([r[k + "_pesq"] for r in rows]),
               np.mean([r[k + "_stoi"] for r in rows])))
