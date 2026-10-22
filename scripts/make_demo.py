#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Generate a synthetic clean/noisy demo pair (no third-party audio)."""
import os, wave
import numpy as np

SR = 16000
DUR = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "samples")


def write_wav(path, x):
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    w.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    t = np.arange(int(SR * DUR)) / SR
    f0 = 115.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for h in range(1, 12):
        sig += (0.9 ** h) * np.sin(h * phase)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 1.8 * t) ** 2
    # pauses, so the demo has both speech-like and silence regions
    env *= (np.sin(2 * np.pi * 0.23 * t) > -0.35).astype(float)
    clean = 0.18 * sig * env
    clean = clean / (np.max(np.abs(clean)) + 1e-9) * 0.7
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t))
    snr_db = 5.0
    scale = np.sqrt(np.mean(clean ** 2)) / (
        np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20))
    noisy = np.clip(clean + scale * noise, -1, 1)
    write_wav(os.path.join(OUT, "demo_clean.wav"), clean)
    write_wav(os.path.join(OUT, "demo_noisy.wav"), noisy)
    print("wrote samples/demo_clean.wav and samples/demo_noisy.wav (16 kHz, 5 dB)")


if __name__ == "__main__":
    main()
