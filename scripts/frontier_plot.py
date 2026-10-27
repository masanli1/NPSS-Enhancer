#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the noise-reduction / quality frontier from target15_sweep*.csv.

    python scripts/frontier_plot.py
"""

import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "GUI"))
sys.path.insert(0, HERE)

from report_target15 import font, axes, plot_xy  # drawing helpers

CASES = ["harmonic_white_+05", "harmonic_white_+00", "harmonic_pink_+05",
         "harmonic_traffic_+10", "music_white_+05", "music_pink_+00",
         "formant_white_+05", "formant_traffic_+05"]
SHORT = ["harm/white/5", "harm/white/0", "harm/pink/5", "harm/traffic/10",
         "music/white/5", "music/pink/0", "formant/white/5", "formant/traffic/5"]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load(os.path.join(ROOT, "target15_sweep.csv"))
    uni = load(os.path.join(ROOT, "target15_sweep_uniform.csv"))
    if not rows:
        print("先运行 scripts/sweep_target15.py")
        return

    W, H = 1400, 640
    img = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(img)
    d.text((28, 14), "降噪量 / 音质前沿：固定目标买到什么，花掉什么", fill=(15, 15, 15), font=font(22))
    d.text((28, 46), "每条细线是一个素材；粗线是 8 个素材的均值。竖线 = 现在的 15 dB 目标。",
           fill=(80, 80, 80), font=font(13))

    tg = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
    ax = (95, 110, 640, 540)
    ylo, yhi = 1.5, 21.0
    axes(d, ax, "target (dB)", "SI-SDR (dB)",
         yticks=[2, 5, 8, 11, 14, 17, 20], ylim=(ylo, yhi))
    x15 = ax[0] + (15 - 6) / 18.0 * (ax[2] - ax[0])
    d.line([x15, ax[1], x15, ax[3]], fill=(210, 120, 120), width=1)
    for t in tg:
        px = ax[0] + (t - 6) / 18.0 * (ax[2] - ax[0])
        d.line([px, ax[3], px, ax[3] + 4], fill=(140, 140, 140))
        d.text((px - 10, ax[3] + 5), "%d" % t, fill=(120, 120, 120), font=font(10))

    for tag in CASES:
        ys = [float([r for r in rows if r["case"] == tag and r["target"] == str(t)][0]["si_sdr"])
              for t in tg]
        plot_xy(d, ax, tg, ys, (6, 24), (ylo, yhi), color=(214, 214, 218), width=1)

    mean_si = [np.mean([float([r for r in rows if r["case"] == c and r["target"] == str(t)][0]["si_sdr"])
                        for c in CASES]) for t in tg]
    plot_xy(d, ax, tg, mean_si, (6, 24), (ylo, yhi), color=(200, 40, 40), width=4)

    if uni:
        mu = [np.mean([float([r for r in uni if r["case"] == c and r["target"] == str(t)][0]["si_sdr"])
                       for c in CASES]) for t in tg]
        plot_xy(d, ax, tg, mu, (6, 24), (ylo, yhi), color=(210, 130, 60), width=3)
        d.text((ax[2] - 190, ax[1] + 10), "— 四视角", fill=(200, 40, 40), font=font(12))
        d.text((ax[2] - 120, ax[1] + 10), "— 均匀施加", fill=(210, 130, 60), font=font(12))
    bi = int(np.argmax(mean_si))
    i15 = tg.index(15.0)
    y15 = ax[3] - (mean_si[i15] - ylo) / (yhi - ylo) * (ax[3] - ax[1])
    d.ellipse([x15 - 4, y15 - 4, x15 + 4, y15 + 4], fill=(40, 90, 200))
    d.text((x15 - 78, y15 + 8), "15 dB: %.2f" % mean_si[i15], fill=(40, 90, 200), font=font(12))
    xb = ax[0] + (tg[bi] - 6) / 18.0 * (ax[2] - ax[0])
    d.ellipse([xb - 5, ax[3] - (mean_si[bi] - ylo) / (yhi - ylo) * (ax[3] - ax[1]) - 5,
               xb + 5, ax[3] - (mean_si[bi] - ylo) / (yhi - ylo) * (ax[3] - ax[1]) + 5],
              fill=(30, 140, 90))
    d.text((xb - 40, ax[3] - (mean_si[bi] - ylo) / (yhi - ylo) * (ax[3] - ax[1]) - 24),
           "均值最优 %.0f dB: %.2f" % (tg[bi], mean_si[bi]), fill=(30, 120, 80), font=font(12))

    # right: table-ish summary of per-case optimum
    x0 = 690
    d.text((x0, 100), "每个素材自己的最优点", fill=(20, 20, 20), font=font(15))
    d.text((x0, 126), "素材", fill=(90, 90, 90), font=font(12))
    d.text((x0 + 150, 126), "SI@15", fill=(90, 90, 90), font=font(12))
    d.text((x0 + 230, 126), "SI@最优", fill=(90, 90, 90), font=font(12))
    d.text((x0 + 330, 126), "最优目标", fill=(90, 90, 90), font=font(12))
    for i, (tag, nm) in enumerate(zip(CASES, SHORT)):
        rs = [r for r in rows if r["case"] == tag and r["target"] != "baseline"]
        b = max(rs, key=lambda r: float(r["si_sdr"]))
        r15 = [r for r in rows if r["case"] == tag and r["target"] == "15.0"][0]
        yy = 150 + i * 26
        d.text((x0, yy), nm, fill=(40, 40, 40), font=font(12))
        d.text((x0 + 150, yy), "%.2f" % float(r15["si_sdr"]), fill=(40, 40, 40), font=font(12))
        d.text((x0 + 230, yy), "%.2f" % float(b["si_sdr"]), fill=(40, 40, 40), font=font(12))
        d.text((x0 + 330, yy), "%s dB" % str(int(float(b["target"]))).rjust(2),
               fill=(40, 40, 40), font=font(12))
        gap = float(b["si_sdr"]) - float(r15["si_sdr"])
        d.text((x0 + 420, yy), ("+%.2f" % gap) if gap > 0.005 else "—",
               fill=(150, 80, 80) if gap > 0.005 else (150, 150, 150), font=font(12))

    d.text((28, H - 46),
           "均值在 %d dB 附近见顶（%.2f），15 dB 只低 %.2f dB；但单素材最优分散在 9–24 dB。"
           % (tg[bi], mean_si[bi], mean_si[bi] - mean_si[i15]), fill=(90, 90, 90), font=font(12))
    d.text((28, H - 24),
           "结论：15 dB 是“可复现的约定”，不是“普遍最优”；固定目标的价值在可控，不在更高的平均分。",
           fill=(90, 90, 90), font=font(12))
    out = os.path.join(ROOT, "target15_frontier.png")
    img.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
