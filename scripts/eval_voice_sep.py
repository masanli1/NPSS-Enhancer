#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""人声特征分离的复现脚本（v3.3）。

两种测量：

1. 语音质量（需要 clean/noisy 成对数据）：
   noisy / 多统计 music / 多统计 voice / 人声提取 四种方法的
   SI-SDR、PESQ-WB、STOI。
2. 非人声抑制（合成、可复现）：
   纯音乐、谐波 drone、共振峰音、白噪、traffic 输入时的输出衰减（dB），
   以及干净语音单独输入时的保真度（SI-SDR / PESQ）。

    python scripts/eval_voice_sep.py --clean ../vbd_test/clean_testset_wav \
                                     --noisy ../vbd_test/noisy_testset_wav --limit 24
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "GUI"))
sys.path.insert(0, ROOT)

import common as C
import engine
import voice_sep as V

try:
    from pesq import pesq
    from pystoi import stoi
except Exception:  # metrics are optional
    pesq = None
    stoi = None


def _load(path):
    x, sr = C.read_wav(path)
    return V.resample(x, sr, V.SR) if sr != V.SR else x


def speech_metrics(clean, den):
    n = min(len(clean), len(den))
    clean, den = clean[:n], den[:n]
    out = {"si_sdr": float(C.si_sdr(den, clean))}
    if pesq is not None:
        out["pesq"] = float(pesq(16000, clean.astype(np.float64),
                                 den.astype(np.float64), "wb"))
        out["stoi"] = float(stoi(clean, den, 16000, extended=False))
    return out


def attenuation(x, y):
    n = min(len(x), len(y))
    a = float(np.sqrt(np.mean(np.asarray(x[:n], np.float64) ** 2)) + 1e-12)
    b = float(np.sqrt(np.mean(np.asarray(y[:n], np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(b / a)


def synth_cases(seconds=8.0, seed=1):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_target_set as M
    rng = np.random.default_rng(seed)
    n = int(V.SR * seconds)
    return {
        "music": (M.sig_music(seconds) * 0.5).astype(np.float32),
        "harmonic_drone": (M.sig_harmonic(seconds) * 0.5).astype(np.float32),
        "formant_vowel": (M.sig_formant(seconds) * 0.5).astype(np.float32),
        "white": (rng.standard_normal(n) * 0.05).astype(np.float32),
        "traffic": M.noise_traffic(n, rng).astype(np.float32),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="", help="16k/48k clean wav dir")
    ap.add_argument("--noisy", default="", help="matching noisy wav dir")
    ap.add_argument("--limit", type=int, default=24,
                    help="pairs picked evenly across speakers")
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--out", default="results_voice_sep.json")
    a = ap.parse_args(argv)

    report = {"strength": a.strength}

    if a.clean and a.noisy and os.path.isdir(a.clean) and os.path.isdir(a.noisy):
        files = sorted(f for f in os.listdir(a.noisy) if f.endswith(".wav"))
        step = max(1, len(files) // max(1, int(a.limit)))
        picked = files[::step][:int(a.limit)]
        rows = []
        for f in picked:
            t0 = time.time()
            clean = _load(os.path.join(a.clean, f))
            noisy = _load(os.path.join(a.noisy, f))[:len(clean)]
            out = {"file": f}
            out["noisy"] = speech_metrics(clean, noisy)
            out["npss_music"] = speech_metrics(clean, engine.denoise_array(
                noisy, V.SR, content_mode="music", multi_stat=True, workers=1))
            out["npss_voice"] = speech_metrics(clean, engine.denoise_array(
                noisy, V.SR, content_mode="voice", multi_stat=True, workers=1))
            ys, _ = V.separate(noisy, V.SR, strength=a.strength)
            out["voice_sep"] = speech_metrics(clean, ys)
            rows.append(out)
            print("%-16s noisy %.2f | music %.2f | voice %.2f | sep %.2f  (%.0fs)"
                  % (f, out["noisy"]["si_sdr"], out["npss_music"]["si_sdr"],
                     out["npss_voice"]["si_sdr"], out["voice_sep"]["si_sdr"],
                     time.time() - t0), flush=True)
        keys = ["noisy", "npss_music", "npss_voice", "voice_sep"]
        mean = {}
        for k in keys:
            mean[k] = {m: float(np.mean([r[k][m] for r in rows]))
                       for m in rows[0][k]}
        report["pairs"] = {"files": rows, "mean": mean}
        print("\n%-12s %8s %8s %8s" % ("method", "SI-SDR", "PESQ", "STOI"))
        for k in keys:
            m = mean[k]
            print("%-12s %8.2f %8.3f %8.4f"
                  % (k, m["si_sdr"], m.get("pesq", float("nan")),
                     m.get("stoi", float("nan"))))

    # ---------------------------------------------------------- rejection
    reject = {}
    for name, x in synth_cases().items():
        y, F = V.separate(x, V.SR, strength=a.strength)
        reject[name] = {"attenuation_db": attenuation(x, y),
                        "speech_ratio": float((F["vad"] > 0.5).mean())}
        print("reject %-15s %7.1f dB  (voice segments %.0f%%)"
              % (name, reject[name]["attenuation_db"],
                 100 * reject[name]["speech_ratio"]))
    report["rejection"] = reject

    # ---------------------------------------------------------- retention
    if a.clean and os.path.isdir(a.clean):
        f = sorted(os.listdir(a.clean))[0]
        clean = _load(os.path.join(a.clean, f))
        y, _ = V.separate(clean, V.SR, strength=a.strength)
        report["retention"] = speech_metrics(clean, y)
        print("retention %-12s SI-SDR %.2f  PESQ %.3f"
              % (f, report["retention"]["si_sdr"],
                 report["retention"].get("pesq", float("nan"))))

    with open(os.path.abspath(a.out), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print("\nwrote", os.path.abspath(a.out))
    return report


if __name__ == "__main__":
    main()
