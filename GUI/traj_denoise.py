#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沿运动轨迹的多帧累积降噪（连贯性繁衍）。

每帧沿自己的运动轨迹与相邻帧对齐后加权合并；不做前景/背景分割，
运动主体与静止背景一样参与累积。
"""
from __future__ import annotations

import numpy as np

import npss_image as M

_GRID = {}


def _grid(H, W):
    if (H, W) not in _GRID:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _GRID[(H, W)] = (yy, xx)
    return _GRID[(H, W)]


def _box_down(a, f):
    if f <= 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def bm_flow(a, b, B=8, R=6, scale=4):
    """块匹配光流：在 1/scale 分辨率上搜索，再放大回全分辨率。"""
    ga = _box_down(M.luma(a), scale)
    gb = _box_down(M.luma(b), scale)
    H, W = ga.shape
    ny, nx = H // B, W // B
    A = ga[:ny * B, :nx * B].reshape(ny, B, nx, B)
    best = np.full((ny, nx), 1e9, np.float32)
    U = np.zeros((ny, nx), np.float32)
    V = np.zeros_like(U)
    canvas = np.zeros((ny * B, nx * B), np.float32)
    for dy in range(-R, R + 1):
        y0, y1 = max(0, dy), min(H, H + dy)
        for dx in range(-R, R + 1):
            x0, x1 = max(0, dx), min(W, W + dx)
            sub = gb[y0:y1, x0:x1]
            m0, m1 = -min(0, dy), -min(0, dx)
            canvas[:] = 0.0
            hh, ww = sub.shape
            canvas[m0:m0 + hh, m1:m1 + ww] = sub[:min(hh, ny * B - m0), :min(ww, nx * B - m1)]
            e = ((A - canvas.reshape(ny, B, nx, B)) ** 2).sum((1, 3))
            m = e < best
            best[m] = e[m]
            U[m] = dx
            V[m] = dy
    U = np.repeat(np.repeat(U, B, 0), B, 1)[:H, :W]
    V = np.repeat(np.repeat(V, B, 0), B, 1)[:H, :W]
    U = M.gaussian_blur(U, 2.0) * scale
    V = M.gaussian_blur(V, 2.0) * scale
    fu = np.repeat(np.repeat(U, scale, 0), scale, 1)
    fv = np.repeat(np.repeat(V, scale, 0), scale, 1)
    fu = np.pad(fu, ((0, max(0, a.shape[0] - fu.shape[0])), (0, max(0, a.shape[1] - fu.shape[1]))), mode="edge")
    fv = np.pad(fv, ((0, max(0, a.shape[0] - fv.shape[0])), (0, max(0, a.shape[1] - fv.shape[1]))), mode="edge")
    return M.gaussian_blur(fu, 2.0), M.gaussian_blur(fv, 2.0)


def warp(img, u, v):
    """双线性重采样（flat take）。"""
    H, W = img.shape[:2]
    yy, xx = _grid(H, W)
    gx = np.clip(xx + u, 0, W - 1.001)
    gy = np.clip(yy + v, 0, H - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    fx = gx - x0
    fy = gy - y0
    flat = img.reshape(-1, img.shape[2])
    idx = (y0.ravel() * W + x0.ravel())
    w00 = ((1 - fx) * (1 - fy)).ravel()[:, None]
    w01 = (fx * (1 - fy)).ravel()[:, None]
    w10 = ((1 - fx) * fy).ravel()[:, None]
    w11 = (fx * fy).ravel()[:, None]
    out = (flat.take(idx, axis=0) * w00
           + flat.take(idx + 1, axis=0) * w01
           + flat.take(idx + W, axis=0) * w10
           + flat.take(idx + W + 1, axis=0) * w11)
    return out.reshape(H, W, img.shape[2])


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
    S_ = np.stack(st).astype(np.float32)
    med = np.median(S_, 0)
    res = np.abs(S_ - med).mean(-1)
    w = np.exp(-(res / (2 * sigma)) ** 2)
    return np.clip((S_ * w[..., None]).sum(0) / (w.sum(0)[..., None] + 1e-6), 0, 1).astype(np.float32)
