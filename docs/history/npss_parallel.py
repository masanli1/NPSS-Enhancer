#!/usr/bin/env python3
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
    p.add_argument("--single", action="store_true",
                   help="disable multi-stat Gmin (use old complementary mode)")
    a = p.parse_args()
    x, sr = C.read_wav_stereo(a.input)
    if x.ndim == 1:
        x = x[:, None]
    t0 = time.time()
    if x.shape[1] == 1:
        y0 = engine.denoise_array(x[:, 0], sr, content_mode="music", complement=True,
                                  loops=max(2, min(8, a.loops)),
                                  multi_stat=not a.single,
                                  workers=a.workers, chunk_sec=a.chunk)
        y = np.repeat(y0[:, None], 2, axis=1)
    elif np.allclose(x[:, 0], x[:, 1]):
        y0 = engine.denoise_array(x[:, 0], sr, content_mode="music", complement=True,
                                  loops=max(2, min(8, a.loops)),
                                  multi_stat=not a.single,
                                  workers=a.workers, chunk_sec=a.chunk)
        y = np.repeat(y0[:, None], 2, axis=1)
    else:
        ys = [engine.denoise_array(x[:, ci], sr, content_mode="music", complement=True,
                                   loops=max(2, min(8, a.loops)),
                                   multi_stat=not a.single,
                                   workers=a.workers, chunk_sec=a.chunk)
              for ci in range(2)]
        nn = min(len(ys[0]), len(ys[1]), len(x))
        y = np.stack([ys[0][:nn], ys[1][:nn]], axis=1)
    dt = time.time() - t0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    C.write_wav_stereo(a.output, y, sr)
    print("in :", a.input, "sr", sr, "ch", x.shape[1], "dur %.1f s" % (len(x) / sr))
    print("out:", a.output, "time %.2f s  RTF %.3f" % (dt, dt / max(1e-9, len(x) / sr)))
    if a.single:
        mode = "single/legacy"
    elif sr > 16000:
        mode = "48k full-band (16k gate + 8k+ preserved)"
    else:
        mode = "16k multi-stat Gmin + blend"
    print("mode: music %s" % mode)


if __name__ == "__main__":
    main()
