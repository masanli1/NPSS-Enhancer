# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GUI"))
import engine


def test_denoise_smoke():
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.2 * np.sin(2 * np.pi * 180 * t) +
         0.01 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
    y = engine.denoise_array(x, sr, content_mode="music", complement=True,
                             loops=2, workers=1, chunk_sec=20)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) <= 1.01


def test_voice_features_and_separation():
    import voice_sep as V
    sr = 16000
    t = np.arange(sr * 2) / sr
    f0 = 140.0
    sig = sum((0.6 ** k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) ** 2
    x = (0.25 * sig * env).astype(np.float32)
    x += 0.01 * np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    y, F = V.separate(x, sr, use_base=False, strength=0.8)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    v = F["voicing"] > 0.3
    assert v.mean() > 0.3
    f0_est = float(np.median(F["f0"][v]))
    assert abs(f0_est - f0) / f0 < 0.1


def test_voice_sep_length_48k():
    """48k -> 16k -> 48k rounding must not lose a sample (v3.3.1 fix)."""
    import voice_sep as V
    sr = 48000
    x = np.random.default_rng(1).normal(0, 0.02, sr * 2 + 1).astype(np.float32)
    y, _ = V.separate(x, sr, use_base=False)
    assert len(y) == len(x)
