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
