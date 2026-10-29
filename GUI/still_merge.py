#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频取帧：用一段连续帧融合出一张干净照片（iPhone 夜景/Deep Fusion 的思路）。

    若干帧 → 分块仿射配准 → 鲁棒加权融合 → 按时间一致性混合（运动区回退单帧）

纯 numpy + ffmpeg。
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

import npss_image as M


def _ffmpeg():
    import engine as E
    p = E.find_ffmpeg()
    if p:
        return p
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def _pc_sub(a, b):
    """相位相关，返回把 b 对齐到 a 的亚像素位移 (dy, dx)。"""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfft2(R, s=a.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def sub(axis):
        if axis == 0:
            y0, y1, y2 = c[(iy - 1) % c.shape[0], ix], c[iy, ix], c[(iy + 1) % c.shape[0], ix]
        else:
            y0, y1, y2 = c[iy, (ix - 1) % c.shape[1]], c[iy, ix], c[iy, (ix + 1) % c.shape[1]]
        return (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)

    dy, dx = iy + sub(0), ix + sub(1)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dy), float(dx)


def _fit_affine(ref, mov, grid=4, win=(640, 480)):
    H, W = ref.shape
    pts, sh = [], []
    for iy in range(grid):
        for ix in range(grid):
            cy = int((iy + 0.5) * H / grid)
            cx = int((ix + 0.5) * W / grid)
            y0 = min(max(0, cy - win[0] // 2), H - win[0])
            x0 = min(max(0, cx - win[1] // 2), W - win[1])
            dy, dx = _pc_sub(ref[y0:y0 + win[0], x0:x0 + win[1]],
                             mov[y0:y0 + win[0], x0:x0 + win[1]])
            pts.append((cx, cy))
            sh.append((dx, dy))
    P = np.array(pts, float)
    S = np.array(sh, float)
    A = np.stack([np.ones(len(P)), (P[:, 0] - W / 2) / W, (P[:, 1] - H / 2) / H], 1)
    cx = np.linalg.lstsq(A, S[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, S[:, 1], rcond=None)[0]
    return cx, cy


def _warp(img, cx, cy):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = -(cx[0] + cx[1] * (xx - W / 2) / W + cx[2] * (yy - H / 2) / H)
    sy = -(cy[0] + cy[1] * (xx - W / 2) / W + cy[2] * (yy - H / 2) / H)
    gx = np.clip(xx + sx, 0, W - 1.001)
    gy = np.clip(yy + sy, 0, H - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    fx = (gx - x0)[..., None]
    fy = (gy - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def read_frames(src, t_sec, count, log=print):
    """读取 t_sec 前后 count 帧（uint8 HxWx3）。"""
    ff = _ffmpeg()
    r = subprocess.run([ff, "-hide_banner", "-i", src], capture_output=True, text=True)
    import re
    m = re.search(r"(\d{2,5})x(\d{2,5})", r.stderr)
    if not m:
        raise RuntimeError("无法解析视频尺寸")
    W, H = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", r.stderr)
    fps = float(m.group(1)) if m else 25.0
    start = max(0.0, t_sec - (count // 2) / fps)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % start, "-i", src,
           "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (H * W * 3)
    fr = np.frombuffer(raw[:n * H * W * 3], np.uint8).reshape(n, H, W, 3)
    return fr.astype(np.float32) / 255.0, fps


def still(src, t_sec=None, count=12, sigma=None, log=print):
    """从视频融合出一张干净照片，返回 (image float32, info)。"""
    info = {}
    raw, fps = read_frames(src, t_sec if t_sec else 1.0, count, log)
    if t_sec is None:
        # 用中点作为参考帧，避免边缘
        raw = read_frames(src, 1.0, count)[0]
    n = len(raw)
    if n < 2:
        raise RuntimeError("帧数不足")
    ref = raw[n // 2]
    if sigma is None:
        y = M.luma(ref)
        sigma = float(np.median(np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)
    log("融合 %d 帧，参考帧噪声 σ≈%.1f/255" % (n, sigma * 255))

    ref_l = M.luma(ref)
    aligned = []
    for i in range(n):
        cx, cy = _fit_affine(ref_l, M.luma(raw[i]))
        aligned.append(_warp(raw[i], cx, cy))
    aligned = np.stack(aligned)
    med = np.median(aligned, 0)
    res = np.abs(aligned - med)
    w = np.exp(-(res / (2 * sigma)) ** 2)
    merged = (aligned * w).sum(0) / (w.sum(0) + 1e-6)

    single, _ = M.denoise(ref, sigma=sigma)
    # 时间一致性：跨帧残差小的位置用融合结果，大的回退单帧
    var = ((aligned - merged) ** 2).mean(0).mean(-1)
    thr = np.percentile(var, 60)
    mask = (var < thr).astype(np.float32)
    mask = M.gaussian_blur(mask, 2.0)[..., None]
    out = np.clip(mask * merged + (1 - mask) * single, 0, 1).astype(np.float32)

    info["frames"] = n
    info["sigma"] = sigma
    info["merged_psnr_gain_hint"] = float(mask.mean())
    return out, info
