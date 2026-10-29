#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沿运动轨迹的多帧累积降噪（连贯性繁衍）。

每一帧都沿自己的运动轨迹与相邻帧对齐，然后加权合并；不做前景/背景分割，
运动主体和静止背景一样参与累积。
"""
from __future__ import annotations

import numpy as np

import common as C
import npss_image as M


def flow_tiles(a, b, tile=64, stride=32, max_shift=None):
    """分块相位相关得到稠密光流（把 b 对齐到 a 的位移场）。"""
    H, W = a.shape[:2]
    ga, gb = M.luma(a), M.luma(b)
    U = np.zeros((H, W), np.float32)
    V = np.zeros((H, W), np.float32)
    Wt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - tile) + 1, stride))
    xs = list(range(0, max(1, W - tile) + 1, stride))
    for y0 in ys:
        for x0 in xs:
            ta = ga[y0:y0 + tile, x0:x0 + tile]
            tb = gb[y0:y0 + tile, x0:x0 + tile]
            dy, dx = _pc(ta, tb)
            yy = slice(y0, min(y0 + tile, H))
            xx = slice(x0, min(x0 + tile, W))
            U[yy, xx] += dx
            V[yy, xx] += dy
            Wt[yy, xx] += 1.0
    Wt = np.maximum(Wt, 1.0)
    U /= Wt
    V /= Wt
    U = M.gaussian_blur(U, tile / 2.0)
    V = M.gaussian_blur(V, tile / 2.0)
    return U, V


def _pc(a, b):
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


def warp(img, u, v):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = np.clip(xx + u, 0, W - 1.001)
    gy = np.clip(yy + v, 0, H - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    fx = (gx - x0)[..., None]
    fy = (gy - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def accumulate(frames, flows, idx, radius=3):
    """以 idx 为中心，沿轨迹累积 ±radius 帧。"""
    n = len(frames)
    acc = frames[idx].astype(np.float64)
    cnt = np.ones(frames[idx].shape[:2])
    # 向前
    cur = frames[idx]
    u_sum = np.zeros(frames[idx].shape[:2], np.float32)
    v_sum = np.zeros_like(u_sum)
    for k in range(1, radius + 1):
        j = idx + k
        if j >= n:
            break
        u, v = flows[j - 1]
        u_sum += u
        v_sum += v
        cur = warp(frames[j], u_sum, v_sum)
        acc += cur
        cnt += 1
    # 向后
    u_sum = np.zeros(frames[idx].shape[:2], np.float32)
    v_sum = np.zeros_like(u_sum)
    for k in range(1, radius + 1):
        j = idx - k
        if j < 0:
            break
        u, v = flows[j]        # flow[j]: 帧 j -> j+1；反向取负
        u_sum -= u
        v_sum -= v
        cur = warp(frames[j], u_sum, v_sum)
        acc += cur
        cnt += 1
    return np.clip(acc / cnt[..., None], 0, 1).astype(np.float32)


def bm_flow(a, b, B=8, R=6):
    """向量化块匹配光流（把 b 对齐到 a 的整数位移场，再平滑）。"""
    ga, gb = M.luma(a), M.luma(b)
    H, W = ga.shape
    ny, nx = H // B, W // B
    A = ga[:ny * B, :nx * B].reshape(ny, B, nx, B)
    best = np.full((ny, nx), 1e9, np.float32)
    U = np.zeros((ny, nx), np.float32)
    V = np.zeros_like(U)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            bb = np.roll(gb, (dy, dx), (0, 1))[:ny * B, :nx * B].reshape(ny, B, nx, B)
            e = ((A - bb) ** 2).sum((1, 3))
            m = e < best
            best[m] = e[m]
            U[m] = dx
            V[m] = dy
    Uf = np.kron(U, np.ones((B, B)))[:H, :W]
    Vf = np.kron(V, np.ones((B, B)))[:H, :W]
    return M.gaussian_blur(Uf, 3.0), M.gaussian_blur(Vf, 3.0)


def accumulate_weighted(frames, flows, idx, radius=3, sigma=0.04):
    """沿轨迹对齐 ±radius 帧，残差加权合并。"""
    H, W = frames[idx].shape[:2]
    st = [frames[idx]]
    us = np.zeros((H, W), np.float32)
    vs = np.zeros_like(us)
    for k in range(1, radius + 1):
        j = idx + k
        if j >= len(frames):
            break
        u, v = flows[j - 1]
        us = us + u
        vs = vs + v
        st.append(warp(frames[j], us, vs))
    us = np.zeros((H, W), np.float32)
    vs = np.zeros_like(us)
    for k in range(1, radius + 1):
        j = idx - k
        if j < 0:
            break
        u, v = flows[j]
        us = us - u
        vs = vs - v
        st.append(warp(frames[j], us, vs))
    S_ = np.stack(st)
    med = np.median(S_, 0)
    res = np.abs(S_ - med).mean(-1)
    w = np.exp(-(res / (2 * sigma)) ** 2)
    return np.clip((S_ * w[..., None]).sum(0) / (w.sum(0)[..., None] + 1e-6), 0, 1).astype(np.float32)
