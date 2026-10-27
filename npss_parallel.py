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
    p.add_argument("--fast", action="store_true",
                   help="fast mode: K=6, 2 passes")
    p.add_argument("--view-profile", default="freq",
                   choices=["freq", "blend", "energy", "struct", "off"],
                   help="four-view quality profile (default freq)")
    p.add_argument("--single", action="store_true",
                   help="disable multi-stat Gmin (use old complementary mode)")
    p.add_argument("--target-nr", type=float, default=None, metavar="DB",
                   help="定果寻因：把噪声底压到正好 DB 分贝（如 15）")
    a = p.parse_args()
    x, sr = C.read_wav_stereo(a.input)
    if x.ndim == 1:
        x = x[:, None]
    info = {}
    t0 = time.time()
    if x.shape[1] == 1:
        y0 = engine.denoise_array(x[:, 0], sr, content_mode="music", complement=True,
                                  loops=max(2, min(8, a.loops)),
                                  multi_stat=not a.single, fast=a.fast,
                                  workers=a.workers, chunk_sec=a.chunk,
                                  target_nr_db=a.target_nr, info_out=info,
                                  view_profile=a.view_profile)
        y = np.repeat(y0[:, None], 2, axis=1)
    elif np.allclose(x[:, 0], x[:, 1]):
        y0 = engine.denoise_array(x[:, 0], sr, content_mode="music", complement=True,
                                  loops=max(2, min(8, a.loops)),
                                  multi_stat=not a.single, fast=a.fast,
                                  workers=a.workers, chunk_sec=a.chunk,
                                  target_nr_db=a.target_nr, info_out=info,
                                  view_profile=a.view_profile)
        y = np.repeat(y0[:, None], 2, axis=1)
    else:
        ys = [engine.denoise_array(x[:, ci], sr, content_mode="music", complement=True,
                                   loops=max(2, min(8, a.loops)),
                                   multi_stat=not a.single, fast=a.fast,
                                   workers=a.workers, chunk_sec=a.chunk,
                                   target_nr_db=a.target_nr, info_out=info,
                                   view_profile=a.view_profile)
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
    if a.target_nr is not None:
        if "achieved_db" in info:
            print("定果寻因: target %.2f dB -> measured %.2f dB (dose c=%.3f, %s)" % (
                float(a.target_nr), float(info["achieved_db"]),
                float(info.get("dose", 0.0)),
                "hit" if info.get("hit") else "MISS"))
        else:
            print("定果寻因: 未完成标定（可能已跳过 multi_stat）")


if __name__ == "__main__":
    main()
