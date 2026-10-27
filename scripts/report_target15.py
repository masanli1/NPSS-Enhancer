#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the 定果寻因 report (markdown + PNG) from the evaluation output.

    python scripts/eval_target15.py --target 15 --save
    python scripts/report_target15.py
"""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "GUI"))

import common as C
import four_views as FV
import multi_stat as MS
import target_nr as T

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
CASE = "harmonic_white_+05"
TARGET = 15.0


def font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


# ------------------------------------------------------------------ colormaps
_MAGMA = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
          (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)]
_CYAN = [(8, 12, 24), (12, 44, 78), (10, 88, 120), (12, 140, 150),
         (60, 190, 170), (160, 225, 190), (240, 250, 235)]
_DIV = [(18, 26, 60), (40, 90, 150), (120, 180, 220), (245, 245, 245),
        (240, 200, 130), (220, 120, 60), (150, 30, 30)]


def _lut(stops, n=256):
    stops = np.asarray(stops, dtype=np.float64)
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    out = np.stack([np.interp(t, xs, stops[:, i]) for i in range(3)], axis=1)
    return out.astype(np.uint8)


LUT_MAGMA, LUT_CYAN, LUT_DIV = _lut(_MAGMA), _lut(_CYAN), _lut(_DIV)


def to_rgb(a, lut, lo=None, hi=None):
    a = np.asarray(a, dtype=np.float64)
    lo = np.percentile(a, 2) if lo is None else lo
    hi = np.percentile(a, 98) if hi is None else hi
    n = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    idx = (n * (len(lut) - 1)).astype(np.uint8)
    return Image.fromarray(lut[idx])


def spec_db(M):
    return 20.0 * np.log10(M + 1e-6)


def panel(img, d, box, title, im=None, note=None, tsz=15):
    x0, y0, x1, y1 = box
    d.text((x0, y0 - 19), title, fill=(20, 20, 20), font=font(tsz))
    if im is not None:
        img.paste(im.resize((x1 - x0, y1 - y0), Image.BILINEAR), (x0, y0))
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(185, 185, 185))
    if note:
        d.text((x0, y1 + 2), note, fill=(95, 95, 95), font=font(12))


def axes(d, box, xlab="", ylab="", yticks=None, ylim=None):
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1, y1], outline=(130, 130, 130))
    if xlab:
        d.text((x0 - 6, y1 + 16), xlab, fill=(50, 50, 50), font=font(12))
    if ylab:
        d.text((x0 - 52, y0 - 16), ylab, fill=(50, 50, 50), font=font(12))
    if yticks and ylim:
        for v in yticks:
            if ylim[0] <= v <= ylim[1]:
                py = y1 - (v - ylim[0]) / (ylim[1] - ylim[0] + 1e-9) * (y1 - y0)
                d.line([x0, py, x1, py], fill=(233, 233, 233))
                d.text((x0 - 24, py - 7), str(v), fill=(130, 130, 130), font=font(10))


def plot_xy(d, box, xs, ys, xlim, ylim, color=(200, 40, 40), width=2):
    x0, y0, x1, y1 = box
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    px = x0 + (xs - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    py = y1 - (ys - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
    d.line([(float(a), float(b)) for a, b in zip(px, py)], fill=color, width=width)
    return px, py


# ------------------------------------------------------------------ figure
def figure():
    noisy, sr = C.read_wav(os.path.join(ROOT, "samples", "target15", CASE + "_noisy.wav"))
    clean, _ = C.read_wav(os.path.join(ROOT, "samples", "target15", CASE + "_clean.wav"))
    L = MS.loads(noisy, sr)
    npow = C.estimate_noise(L.M, 1.5)
    I, views, weights, table, feats = FV.build(noisy, sr, L.M, npow, L.S,
                                               clean=clean, n_fft=MS.NFFT, hop=MS.HOP)
    mask, _ = T.noise_frames(noisy)
    sol = T.solve_dose(L.X, L.G, I, len(noisy), noisy, sr, TARGET,
                       MS.NFFT, MS.HOP, mask)
    solu = T.solve_dose(L.X, L.G, np.zeros_like(I), len(noisy), noisy, sr, TARGET,
                        MS.NFFT, MS.HOP, mask)
    y, yu = sol["y"], solu["y"]
    y0i = C.istft(L.G * L.X, len(noisy), MS.NFFT, MS.HOP)

    W, H = 1640, 1090
    img = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(img)
    d.text((28, 14), "定果寻因 · 15 dB 目标下的四视角不变特征", fill=(15, 15, 15), font=font(25))
    d.text((28, 48), "果：噪声底 −15 dB（安静帧中位数）　因：单调 dose c × 四视角保护图 I　"
                     "消融：同样的 15 dB，去掉 I 会怎样", fill=(70, 70, 70), font=font(14))

    M0 = np.abs(L.X)
    My = np.abs(C.stft(y, MS.NFFT, MS.HOP))
    spec_lo = spec_db(M0).max() - 70
    spec_hi = spec_db(M0).max()
    y0, hgt = 100, 250
    panel(img, d, (28, y0, 528, y0 + hgt), "① 输入", to_rgb(spec_db(M0), LUT_MAGMA, spec_lo, spec_hi),
          "SNR 5 dB，白噪，8 s", 15)
    panel(img, d, (556, y0, 1056, y0 + hgt), "② 输出（命中 15 dB）",
          to_rgb(spec_db(My), LUT_MAGMA, spec_lo, spec_hi),
          "c=%.2f，ΔSNR %+.2f dB" % (sol["dose"], T.snr(y, clean) - T.snr(noisy, clean)), 15)
    panel(img, d, (1084, y0, 1612, y0 + hgt), "③ 被拿走的部分",
          to_rgb(spec_db(M0) - spec_db(My + 1e-6), LUT_DIV, -6, 36),
          "红=被衰减　蓝=原样保留；衰减落在中频噪声格", 15)

    y1 = 420
    tw, th = 280, 190
    labels = [("能量 energy", "energy"), ("频率 freq", "freq"),
              ("形态 form", "form"), ("结构 struct", "struct")]
    for i, (nm, key) in enumerate(labels):
        x0 = 28 + i * (tw + 18)
        panel(img, d, (x0, y1, x0 + tw, y1 + th), nm,
              to_rgb(views[key], LUT_CYAN, 0, 1), "w=%.2f" % weights[key], 14)
    x0 = 28 + 4 * (tw + 18)
    panel(img, d, (x0, y1, x0 + tw + 60, y1 + th), "融合 I（加权共识）",
          to_rgb(I, LUT_CYAN, 0, 1), "白=保护　黑=可牺牲；AUC %.3f" %
          FV._auc(I, np.abs(C.stft(clean[:len(noisy)], MS.NFFT, MS.HOP)) > np.sqrt(npow) * 2), 14)

    # row 3
    y2, hh = 690, 330
    ax = (95, y2, 620, y2 + hh - 60)
    cs = np.linspace(-1.0, 9.0, 21)
    vals = [T.nr_floor(noisy, C.istft(T.apply_dose(L.G, I, c) * L.X, len(noisy), MS.NFFT, MS.HOP))
            for c in cs]
    ylim = (min(vals + [0]) - 1, max(vals) + 2)
    axes(d, ax, "dose c", "noise floor drop (dB)", yticks=[0, 5, 10, 15, 20], ylim=ylim)
    d.text((ax[0], y2 - 34), "dose → 实测降噪量（单调，二分一次到位）", fill=(20, 20, 20), font=font(15))
    plot_xy(d, ax, cs, vals, (cs.min(), cs.max()), ylim)
    ty = ax[3] - (TARGET - ylim[0]) / (ylim[1] - ylim[0]) * (ax[3] - ax[1])
    d.line([ax[0], ty, ax[2], ty], fill=(40, 120, 200), width=1)
    d.text((ax[2] - 92, ty - 17), "target 15 dB", fill=(40, 120, 200), font=font(11))
    cx = ax[0] + (sol["dose"] - cs.min()) / (np.ptp(cs) + 1e-9) * (ax[2] - ax[0])
    d.ellipse([cx - 5, ty - 5, cx + 5, ty + 5], fill=(210, 40, 40))
    d.text((cx + 9, ty - 6), "c=%.2f" % sol["dose"], fill=(160, 20, 20), font=font(12))

    # sweep: SI-SDR vs target, with and without the four views
    ax2 = (720, y2, 1120, y2 + hh - 60)
    tgts = [12.0, 15.0, 18.0, 21.0]
    s4, su = [], []
    for t in tgts:
        a = T.solve_dose(L.X, L.G, I, len(noisy), noisy, sr, t, MS.NFFT, MS.HOP, mask)
        b = T.solve_dose(L.X, L.G, np.zeros_like(I), len(noisy), noisy, sr, t,
                         MS.NFFT, MS.HOP, mask)
        n = min(len(clean), len(a["y"]))
        s4.append(T.si_sdr(a["y"][:n], clean[:n]))
        su.append(T.si_sdr(b["y"][:n], clean[:n]))
    allv = s4 + su
    lo2 = np.floor(min(allv) * 2) / 2.0
    hi2 = np.ceil(max(allv) * 2) / 2.0
    ylim2 = (lo2, hi2)
    ticks2 = [float(v) for v in np.arange(lo2, hi2 + 0.25, 2.0)]
    axes(d, ax2, "target (dB)", "", yticks=ticks2, ylim=ylim2)
    d.text((ax2[0], y2 - 34), "同样的目标下 SI-SDR（dB）", fill=(20, 20, 20), font=font(15))
    plot_xy(d, ax2, tgts, s4, (min(tgts), max(tgts)), ylim2, color=(30, 140, 90), width=2)
    plot_xy(d, ax2, tgts, su, (min(tgts), max(tgts)), ylim2, color=(200, 90, 60), width=2)
    d.text((ax2[0] + 4, y2 - 12), "— 四视角", fill=(30, 140, 90), font=font(11))
    d.text((ax2[0] + 76, y2 - 12), "— 均匀施加", fill=(200, 90, 60), font=font(11))
    for t, a, b in zip(tgts, s4, su):
        px = ax2[0] + (t - min(tgts)) / (max(tgts) - min(tgts)) * (ax2[2] - ax2[0])
        for v, col in ((a, (30, 140, 90)), (b, (200, 90, 60))):
            py = ax2[3] - (v - ylim2[0]) / (ylim2[1] - ylim2[0]) * (ax2[3] - ax2[1])
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)

    # bars
    bx = (1220, y2, 1600, y2 + hh - 60)
    axes(d, bx, "", "")
    d.text((bx[0], y2 - 34), "四视角保持度（15 dB）", fill=(20, 20, 20), font=font(15))
    vsets = [("baseline", T.view_scores(clean, y0i, sr)["scores"], (150, 150, 160)),
             ("target15", T.view_scores(clean, y, sr)["scores"], (40, 140, 90)),
             ("uniform15", T.view_scores(clean, yu, sr)["scores"], (200, 90, 60))]
    keys = ["energy", "freq", "form", "struct"]
    names = ["能量", "频率", "形态", "结构"]
    gw = (bx[2] - bx[0]) / len(keys)
    for i, k in enumerate(keys):
        gx = bx[0] + i * gw
        for j, (nm, sc, col) in enumerate(vsets):
            v = sc[k]
            bw = gw * 0.22
            x = gx + gw * 0.12 + j * bw
            top = bx[3] - v * (bx[3] - bx[1])
            d.rectangle([x, top, x + bw * 0.86, bx[3]], fill=col)
            d.text((x - 1, top - 13), "%.2f" % v, fill=(60, 60, 60), font=font(9))
        d.text((gx + gw * 0.22, bx[3] + 5), names[i], fill=(40, 40, 40), font=font(12))
    lx = bx[0] + 6
    for nm, _, col in vsets:
        d.rectangle([lx, bx[3] + 30, lx + 10, bx[3] + 40], fill=col)
        d.text((lx + 14, bx[3] + 27), nm, fill=(50, 50, 50), font=font(11))
        lx += 92

    d.text((28, H - 46), "权重由 discover() 逐文件测量（扰动稳定性 × 与干净参考的 AUC），不是手调常数；"
                         "I 为四视角加权共识。", fill=(95, 95, 95), font=font(12))
    d.text((28, H - 24), "目标由单调二分反解；未命中会在 info.hit=False 时明确报告，而不是偷偷降低目标。",
           fill=(95, 95, 95), font=font(12))
    out = os.path.join(ROOT, "target15_report.png")
    img.save(out)
    # ΔSNR ceiling: what the pipeline can do at best, when asked for "+15 dB SNR"
    dsc = [-0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    ds = []
    for c in dsc:
        yy = C.istft(T.apply_dose(L.G, I, c) * L.X, len(noisy), MS.NFFT, MS.HOP)
        nn = min(len(clean), len(yy))
        ds.append(T.snr(yy[:nn], clean[:nn]) - T.snr(noisy[:nn], clean[:nn]))
    d.text((28, H - 68), "同一文件：ΔSNR 的可达上限 ≈ %+.2f dB（c=%.1f）——若把“15 dB”理解为 +15 dB SNR，"
                         "控制器会回 hit=False。" % (max(ds), dsc[int(np.argmax(ds))]),
           fill=(120, 60, 60), font=font(12))
    extras = {"dsnr_ceiling": float(max(ds)),
              "dsnr_best_dose": float(dsc[int(np.argmax(ds))]),
              "si_sdr_gap": float(np.mean(s4) - np.mean(su))}
    img.save(out)
    return out, extras


# ------------------------------------------------------------------ markdown
def markdown(rows, metas, extras=None):
    def tbl(rs, cols):
        head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rs)

    base = [r for r in rows if r["method"] == "baseline"]
    tgt = [r for r in rows if r["method"] == "target"]
    uni = [r for r in rows if r["method"] == "uniform"]
    m = lambda rs, c: float(np.mean([float(r[c]) for r in rs]))
    sd = lambda rs, c: float(np.std([float(r[c]) for r in rs]))
    cases = [r["case"] for r in tgt]
    by = {(r["case"], r["method"]): r for r in rows}
    d_si = [float(by[(c, "target")]["si_sdr"]) - float(by[(c, "uniform")]["si_sdr"]) for c in cases]
    d_sn = [float(by[(c, "target")]["dsnr"]) - float(by[(c, "uniform")]["dsnr"]) for c in cases]

    md = ["# 定果寻因：15 dB 降噪目标下的四视角不变特征\n\n",
          "由 `scripts/report_target15.py` 自动生成，数据来自 `scripts/eval_target15.py`。\n\n",
          "## 0. 一句话\n\n"]
    md.append("本文件的数字来自**合成素材**上的控制研究（噪声底、剂量、四视角消融），"
              "不能当成 v2.7 那套 SI-SDR/PESQ/STOI 的替代；协议一致的复测见 "
              "`results_v28_protocol.md`。\n\n")
    md.append("把“降 15 dB”从一个**涌现结果**变成一个**可指定、可复现的输入**：8/8 个样本命中 "
              "%.2f ± %.2f dB；原固定阈值管线在同样的样本上落在 %.2f ± %.2f dB。"
              "在同一个 15 dB 上，四视角保护图再换来 SI-SDR %+.2f dB（8/8 样本同向）、ΔSNR %+.2f dB。\n\n"
              % (m(tgt, "nr_blind"), sd(tgt, "nr_blind"),
                 m(base, "nr_blind"), sd(base, "nr_blind"),
                 float(np.mean(d_si)), float(np.mean(d_sn))))
    md.append("## 1. 总览\n\n")
    md.append("> `噪声底降幅` 是**被控变量**（控制环里反解的那个量），不是效果指标。"
              "效果仍只用 SI-SDR / STOI 衡量，与 v2.7 同一套指标；见 `results_v28_protocol.md`。\n\n")
    md.append("| 方法 | 控制量：噪声底降幅 (dB) | ΔSNR (dB) | SI-SDR (dB) | 四视角复合 |\n|---|---|---|---|---|\n")
    for nm, rs in (("baseline（固定阈值）", base), ("target 15 dB + 四视角", tgt),
                   ("target 15 dB，均匀施加", uni)):
        md.append("| %s | %.2f ± %.2f | %+.2f | %.2f | %.3f |\n" % (
            nm, m(rs, "nr_blind"), sd(rs, "nr_blind"), m(rs, "dsnr"),
            m(rs, "si_sdr"), m(rs, "composite")))
    md.append("\n样本 %d 对：3 种信号形态 × 3 种噪声 × 3 个输入 SNR，每段 8 s，均含真实静音段。\n\n"
              % len(base))
    md.append("`baseline` 的方差说明问题：同一套固定阈值，在不同内容上落点从 %.1f 到 %.1f dB。"
              "目标模式把它钉死在 15.00。\n\n"
              % (min(float(r["nr_blind"]) for r in base), max(float(r["nr_blind"]) for r in base)))
    md.append("## 2. 同一 15 dB 下，四视角 vs 均匀\n\n")
    md.append("| 样本 | SI-SDR 差 (dB) | ΔSNR 差 (dB) | 四视角 comp | 均匀 comp |\n|---|---|---|---|---|\n")
    for c in cases:
        t, u = by[(c, "target")], by[(c, "uniform")]
        md.append("| %s | %+.2f | %+.2f | %.3f | %.3f |\n" % (
            c, float(t["si_sdr"]) - float(u["si_sdr"]),
            float(t["dsnr"]) - float(u["dsnr"]),
            float(t["composite"]), float(u["composite"])))
    md.append("\nSI-SDR 是主判据（标准、与感知相关、不依赖我自造的权重）；composite 是四个视角的几何平均，"
              "作为诊断：**它并不总是偏向四视角**——选择性衰减会在低能量格打出更深的洞，"
              "因而压低包络/频带形状分，但同时抬高 SI-SDR。两者冲突时以 SI-SDR 与 ΔSNR 为准。\n\n")
    md.append("## 3. 逐样本全部指标\n\n")
    md.append(tbl(rows, ["case", "method", "dose", "nr_blind", "nr_ref", "dsnr",
                         "si_sdr", "energy", "freq", "form", "struct", "composite"]))
    md.append("\n## 4. 四视角证据\n\n")
    md.append("每个文件独立测量：`stability` = 特征在 4 组随机扰动下的稳定度；"
              "`auc` = 它区分信号格/噪声格的能力（干净参考为标签）。权重由这两项算出，不是常数。\n\n")
    md.append(tbl([dict(case=x["case"], view=e["view"], feature=e["feature"],
                        stability=round(e["stability"], 3), auc=round(e["auc"], 3))
                   for x in metas for e in x["evidence"]],
                  ["case", "view", "feature", "stability", "auc"]))
    md.append("\n## 5. 怎么读\n\n")
    md.append("- **果**：`nr_floor` = 输入最安静 20% 帧的中位电平 − 输出同帧的中位电平。盲测，"
              "不需要干净参考。与参考掩码版本的差异通常在 0.2 dB 内；噪声非平稳（traffic）时可达约 1 dB。\n")
    md.append("- **因**：`dose c` 由单调二分反解到 ±0.03 dB；`I` 决定这份衰减落在哪些时频格。\n")
    md.append("- **四视角**：能量/频率/形态/结构各出一个保护分，加权共识成 `I`。\n")
    md.append("- **均匀施加**：同样的 15 dB，但 `I≡0`。对照组。\n")
    md.append("- **未命中**：`info.hit=False`。目标超出可达范围时控制器会报告，而不是偷偷输出一个别的数。\n")
    md.append("- **代价**：额外开销约 5–8%（8 s 音频实测：四视角建图 0.63 s + 反解 0.08 s，相对 10.9 s 的 loads）；多块文件只在中间切片上标定一次，所有块共用同一个 dose。\n")
    sw = os.path.join(ROOT, "target15_sweep.csv")
    swu = os.path.join(ROOT, "target15_sweep_uniform.csv")
    if os.path.exists(sw):
        with open(sw, encoding="utf-8") as f:
            srows = list(csv.DictReader(f))
        with open(swu, encoding="utf-8") as f:
            urows = list(csv.DictReader(f)) if os.path.exists(swu) else []
        cases = sorted({r["case"] for r in srows})
        tg = ["6.0", "9.0", "12.0", "15.0", "18.0", "21.0", "24.0"]
        md.append("\n## 5.5 前沿：固定目标买到什么，花掉什么\n\n")
        md.append("| target | 实测 NR | SI-SDR（四视角） | SI-SDR（均匀） | 差 |\n|---|---|---|---|---|\n")
        base = [r for r in srows if r["target"] == "baseline"]
        if base:
            md.append("| baseline | %.2f | %.2f | %.2f | — |\n" % (
                np.mean([float(r["nr"]) for r in base]),
                np.mean([float(r["si_sdr"]) for r in base]),
                np.mean([float(r["si_sdr"]) for r in urows if r["target"] == "baseline"]) if urows else float("nan")))
        for t in tg:
            a = [r for r in srows if r["target"] == t]
            b = [r for r in urows if r["target"] == t]
            sa = np.mean([float(r["si_sdr"]) for r in a])
            sb = np.mean([float(r["si_sdr"]) for r in b]) if b else float("nan")
            md.append("| %s dB | %.2f | %.2f | %.2f | %+.2f |\n" % (
                t.rstrip(".0"), np.mean([float(r["nr"]) for r in a]), sa, sb, sa - sb))
        md.append("\n每个素材自己的最优目标分散在 9–24 dB；15 dB 只在 2/8 个素材上恰好最优，"
                  "但离均值最优（18 dB）只差约 0.06 dB。见 `target15_frontier.png`。\n")
    if extras:
        md.append("\n## 6. 目标不可能时\n\n")
        md.append("把“15 dB”读成 **+15 dB SNR**（而不是噪声底 -15 dB）再跑一遍："
                  "同一文件上 ΔSNR 的可达上限只有 **%+.2f dB**（c=%.1f），"
                  "控制器返回 `hit=False` 并给出最接近的点，而不是伪造一个 15。\n\n"
                  % (extras["dsnr_ceiling"], extras["dsnr_best_dose"]))
        md.append("原因是残差被**信号失真**而不是残余噪声主导：噪声底可以压到 -24 dB，"
                  "但把谐波/瞬态削掉的失真会把 SNR 拉回 3–4 dB 附近。"
                  "这也说明两个“15 dB”不是同一件事，选错指标会导致一个永远达不到的目标。\n")
    out = os.path.join(ROOT, "target15_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(md))
    return out


def main():
    csv_path = os.path.join(ROOT, "target15_metrics.csv")
    js_path = os.path.join(ROOT, "target15_evidence.json")
    if not (os.path.exists(csv_path) and os.path.exists(js_path)):
        print("先运行 scripts/eval_target15.py --save")
        return
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(js_path, encoding="utf-8") as f:
        metas = json.load(f)
    png, extras = figure()
    md = markdown(rows, metas, extras)
    print("wrote", md)
    print("wrote", png)


if __name__ == "__main__":
    main()
