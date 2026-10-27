#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run the v2.7 protocol, unchanged, with the target mode added as rows.

v2.7 measured denoising with SI-SDR / PESQ / STOI on clean-noisy pairs
(`eval_official.py`, 30 VoiceBank-DEMAND pairs, 16 kHz).  This script keeps that
protocol exactly -- same input layout, same methods, same metrics, same CSV
columns -- and adds three rows for the 定果寻因 mode.  Nothing here re-defines
"denoising effect": the controlled variable (noise-floor drop) is *not* scored
as quality.

    # on the real corpus (produces numbers comparable to v2.7's table)
    PYTHONPATH=... python3 scripts/eval_protocol_v27.py \
        --clean data/val16/clean --noisy data/val16/noisy --out results_v28.csv

    # on a local stand-in set
    python3 scripts/eval_protocol_v27.py \
        --clean data/local16/clean --noisy data/local16/noisy

PESQ needs the `pesq` C extension.  If it cannot be imported (hardened-runtime
Python, missing wheel), the run continues with SI-SDR + STOI and the PESQ
column is left empty -- it never silently substitutes a different metric.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "GUI"))
sys.path.insert(0, ROOT)

import common as C
import engine

try:                       # same metric module v2.7 used
    from metrics import pesq_stoi as _pesq_stoi
    HAVE_PESQ = True
except Exception as _e:    # pragma: no cover - environment dependent
    HAVE_PESQ = False
    _PESQ_ERR = str(_e)

try:
    from pystoi import stoi as _stoi
    HAVE_STOI = True
except Exception:
    HAVE_STOI = False

try:
    from pesq import pesq as _pesq
    HAVE_PESQ_C = True
except Exception:
    HAVE_PESQ_C = False


def si_sdr(est, ref):
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref
    return 10.0 * np.log10(np.sum(t * t) / (np.sum((est - t) ** 2) + 1e-12))


def score(clean, y, sr=16000):
    n = min(len(clean), len(y))
    c = np.asarray(clean, np.float32)[:n]
    d = np.asarray(y, np.float32)[:n]
    out = {"si_sdr": round(float(si_sdr(d, c)), 3)}
    if HAVE_STOI:
        out["stoi"] = round(float(_stoi(c, d, 16000, extended=False)), 4)
    else:
        out["stoi"] = ""
    if HAVE_PESQ_C:
        try:
            out["pesq"] = round(float(_pesq(16000, c, d, "wb")), 3)
        except Exception:
            out["pesq"] = ""
    else:
        out["pesq"] = ""
    return out


def specsub(x):
    X, M, gb, Y0, R, se = C.prepare(x, 512, 128, 1.5, 0.05, "specsub")
    return C.istft(gb * X, len(x), 512, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=os.path.join(ROOT, "data", "local16", "clean"))
    ap.add_argument("--noisy", default=os.path.join(ROOT, "data", "local16", "noisy"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v28_protocol.csv"))
    ap.add_argument("--targets", nargs="*", type=float, default=[15.0, 18.0])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.clean) if f.endswith(".wav"))
    if a.limit:
        files = files[: a.limit]
    print("pairs: %d   PESQ: %s   STOI: %s" % (
        len(files), "yes" if HAVE_PESQ_C else "no", "yes" if HAVE_STOI else "no"))

    rows = []
    for fi, f in enumerate(files):
        clean, _ = C.read_wav(os.path.join(a.clean, f))
        noisy, _ = C.read_wav(os.path.join(a.noisy, f))
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]

        outs = {
            "noisy": noisy,
            "specsub": specsub(noisy),
            "npss": engine.denoise_array(noisy, 16000, content_mode="music",
                                         complement=False, multi_stat=False,
                                         workers=1, chunk_sec=20),
            "multi": engine.denoise_array(noisy, 16000, content_mode="music",
                                          multi_stat=True, workers=1, chunk_sec=20),
            "multi_voice": engine.denoise_array(noisy, 16000, content_mode="voice",
                                                multi_stat=True, workers=1, chunk_sec=20),
        }
        for t in a.targets:
            outs["target%g" % t] = engine.denoise_array(
                noisy, 16000, content_mode="music", multi_stat=True, workers=1,
                chunk_sec=20, target_nr_db=t)

        for method, y in outs.items():
            row = dict(file=f, method=method)
            row.update(score(clean, y))
            rows.append(row)
        print("[%d/%d] %-32s %s" % (fi + 1, len(files), f,
                                    " ".join("%s=%s" % (m, rows[-len(outs) + i]["si_sdr"])
                                             for i, m in enumerate(outs))))

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "method", "si_sdr", "pesq", "stoi"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)

    methods = sorted({r["method"] for r in rows})
    print("\n%-14s %10s %8s %8s" % ("method", "SI-SDR", "PESQ", "STOI"))
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        si = np.mean([float(r["si_sdr"]) for r in rs])
        pe = [float(r["pesq"]) for r in rs if r["pesq"] != ""]
        st = [float(r["stoi"]) for r in rs if r["stoi"] != ""]
        print("%-14s %10.2f %8s %8s" % (
            m, si,
            ("%.3f" % np.mean(pe)) if pe else "-",
            ("%.4f" % np.mean(st)) if st else "-"))


if __name__ == "__main__":
    main()
