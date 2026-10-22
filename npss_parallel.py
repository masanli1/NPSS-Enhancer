#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Best music-mode CLI: NPSS + complementary pivots (loops=5, floor=0.3)."""
import os, sys, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "GUI"))
import common as C
import engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--chunk", type=float, default=20.0)
    a = p.parse_args()
    x, sr = C.read_wav(a.input)
    t0 = time.time()
    y = engine.denoise_array(x, sr, content_mode="music", complement=True,
                             loops=max(2, min(8, a.loops)),
                             workers=a.workers, chunk_sec=a.chunk)
    dt = time.time() - t0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    C.write_wav(a.output, y, sr)
    print("in :", a.input, "sr", sr, "dur %.1f s" % (len(x) / sr))
    print("out:", a.output, "time %.2f s  RTF %.3f" % (dt, dt / max(1e-9, len(x) / sr)))
    print("mode: music 16k comp loops=%d floor=0.3" % a.loops)


if __name__ == "__main__":
    main()
