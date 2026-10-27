#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-driven four-view ablation/tuning on VoiceBank-DEMAND 16k.

No hard target: the objective is quality (SI-SDR/STOI/PESQ), with the
four-view maps used as soft protectors/modulators of the existing loads.
"""
import os, sys, argparse, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GUI"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C
import multi_stat as MS
import four_views as FV
from metrics import pesq_stoi


def apply_views(G, I, beta=0.35):
    # high I -> exponent < 1 (protect); low I -> exponent > 1 (suppress)
    e = np.clip(1.0 + beta * (0.5 - I), 0.25, 2.0)
    return np.clip(np.power(np.maximum(G, 1e-6), e), 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/val16/clean")
    ap.add_argument("--noisy", default="data/val16/noisy")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="four_view_tune.csv")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.clean) if f.endswith(".wav"))[:max(1, a.limit)]
    rows = []
    for fi, f in enumerate(files):
        c, _ = C.read_wav(os.path.join(a.clean, f))
        x, _ = C.read_wav(os.path.join(a.noisy, f))
        n = min(len(c), len(x)); c, x = c[:n], x[:n]
        L = MS.loads(x, 16000, fast=False, view_profile="off")
        base = C.istft(L.G * L.X, n, MS.NFFT, MS.HOP)
        npow = C.estimate_noise(L.M, 1.5)
        w = {"energy": 0.25, "freq": 0.25, "form": 0.25, "struct": 0.25}
        I, views, weights, table, feats = FV.build(x, 16000, L.M, npow, L.S,
                                                   weights=w)
        I = FV.smooth_protect(I, 1)
        variants = {"base": None, "all": I}
        for k in ("energy", "freq", "form", "struct"):
            variants[k] = FV.smooth_protect(views[k], 1)
        variants["ef"] = FV.smooth_protect(FV.fuse({"energy": views["energy"], "freq": views["freq"]}, {"energy": 0.5, "freq": 0.5}), 1)
        variants["fs"] = FV.smooth_protect(FV.fuse({"form": views["form"], "struct": views["struct"]}, {"form": 0.5, "struct": 0.5}), 1)
        row = {"file": f}
        for name, Im in variants.items():
            for beta in ([0.0] if Im is None else [0.25, 0.5]):
                tag = "%s_b%.2f" % (name, beta)
                y = base if Im is None else C.istft(apply_views(L.G, Im, beta) * L.X, n, MS.NFFT, MS.HOP)
                row[tag + "_si"] = C.si_sdr(y, c)
                p, s = pesq_stoi(c, y); row[tag + "_pesq"] = p; row[tag + "_stoi"] = s
        rows.append(row)
        print(fi + 1, f, flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    keys = [k for k in rows[0] if k.endswith("_si")]
    print("\nmeans (N=%d):" % len(rows))
    for k in keys:
        tag = k[:-3]
        si = np.mean([r[k] for r in rows]); pe = np.mean([r[tag + "_pesq"] for r in rows]); st = np.mean([r[tag + "_stoi"] for r in rows])
        print("  %-14s SI %6.2f  PESQ %.3f  STOI %.4f" % (tag, si, pe, st))


if __name__ == "__main__":
    main()
