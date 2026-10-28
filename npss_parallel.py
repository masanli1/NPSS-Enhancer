#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI: any audio/video container supported by ffmpeg in, any supported out."""
import os, sys, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "GUI"))
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
    p.add_argument("--single", action="store_true",
                   help="disable multi-stat Gmin (use old complementary mode)")
    p.add_argument("--content", default="music", choices=["music", "voice"],
                   help="music or voice (voice adds 40%% NMF at 16k)")
    p.add_argument("--proc-sr", default="auto", choices=["auto", "16000", "48000"],
                   help="processing sample rate (auto: 48k for hi-res/video)")
    p.add_argument("--view-profile", default="freq",
                   choices=["freq", "blend", "energy", "struct", "off"],
                   help="four-view quality profile (default freq)")
    p.add_argument("--target-nr", type=float, default=None, metavar="DB",
                   help="定果寻因：把噪声底压到正好 DB 分贝（如 15）")
    a = p.parse_args()

    if a.proc_sr == "auto":
        sr0 = engine.probe_audio_sr(a.input)
        proc_sr = 48000 if (sr0 and sr0 >= 32000) or engine.is_video(a.input) else 16000
    else:
        proc_sr = int(a.proc_sr)

    info = {}
    t0 = time.time()
    engine.process_file(
        a.input, a.output, mode="auto", proc_sr=proc_sr, out_sr=proc_sr,
        content_mode=a.content, complement=True,
        multi_stat=not a.single, fast=a.fast, workers=a.workers,
        chunk_sec=a.chunk, loops=max(2, min(8, a.loops)),
        target_nr_db=a.target_nr, view_profile=a.view_profile,
        info_out=info, log=lambda s: None)
    dt = time.time() - t0
    print("in :", a.input)
    print("out:", a.output, "proc_sr", proc_sr, "time %.2f s" % dt)
    if a.target_nr is not None and "achieved_db" in info:
        print("target %.1f dB -> %.2f dB (%s)" % (
            float(a.target_nr), float(info["achieved_db"]),
            "hit" if info.get("hit") else "miss"))


if __name__ == "__main__":
    main()
