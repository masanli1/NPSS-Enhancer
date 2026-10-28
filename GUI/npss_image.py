# -*- coding: utf-8 -*-
"""NPSS-Image —— 最佳效果算法（单文件、无第三方依赖，只需 numpy + Pillow）。

算法：8x8 块 DCT，3σ 硬阈值（DC 保护），16 相位 cycle-spin 平均。
可选：量化稳定度层（灰度 AWGN +0.15 dB，传感器噪声 -0.46 dB，默认关）。

实测（5 张 Kodak 灰度，AWGN）：
    sigma=25 : 27.90 dB / SSIM 0.7785   (高斯 25.71 / 中值 24.70)
    sigma=50 : 24.08 dB / SSIM 0.6392   (高斯 22.65 / 中值 20.72)
传感器噪声 31.09（高斯 28.50）；JPEG q20 32.45（输入 31.82）。
"""
from __future__ import annotations

import os as _os
# 单线程 BLAS：小矩阵/小批量下反而更快（Accelerate 多线程开销大）
for _k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import numpy as np

EPS = 1e-12
BLK = 8


def to_float(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def to_uint8(a):
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def luma(a):
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.shape[-1] == 1:
        return a[..., 0].astype(np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def gaussian_blur(a, sigma=1.0, truncate=3.0):
    if sigma <= 0:
        return a.astype(np.float32)
    r = max(1, int(round(truncate * float(sigma))))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()

    def blur2(m):
        m = np.pad(m, ((r, r), (r, r)), mode="reflect")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, m)
        return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, out)

    a = a.astype(np.float32)
    if a.ndim == 2:
        return blur2(a)
    return np.stack([blur2(a[..., c]) for c in range(a.shape[2])], axis=-1)


def extremum_position(mag):
    h, w = mag.shape
    mp = np.pad(mag, 2, mode="edge")
    b = np.lib.stride_tricks.sliding_window_view(mp, (2, 2))[2:2 + h, 2:2 + w]
    c = np.lib.stride_tricks.sliding_window_view(mp, (4, 4))[1:1 + h, 1:1 + w]
    m2 = b.mean(axis=(-2, -1))
    m4 = c.mean(axis=(-2, -1))
    flat = np.ascontiguousarray(b).reshape(h, w, 4)
    return np.where(m2 > m4, flat.argmax(axis=2), flat.argmin(axis=2)).astype(np.int8)


def quant_stability(a, sigma=0.1, K=6, detail_sigma=1.2):
    d = a - gaussian_blur(a, detail_sigma)
    p0 = extremum_position(np.abs(d))
    levels = np.logspace(-0.3, 1.0, int(K))
    acc = np.zeros(a.shape, dtype=np.float32)
    for lv in levels:
        step = max(1, int(round(1.0 + 12.0 * float(lv) * float(sigma))))
        q = np.round(np.clip(a, 0, 1) * 255.0 / step) * step / 255.0
        dq = q - gaussian_blur(q, detail_sigma)
        acc += (extremum_position(np.abs(dq)) == p0).astype(np.float32)
    return acc / float(len(levels))


def _dct_mat(n=BLK):
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * m + 1) * k / (2.0 * n))
    d[0] /= np.sqrt(2.0)
    return d.astype(np.float64)


_D = _dct_mat(BLK)


def _split_blocks(a):
    h, w = a.shape
    ny, nx = h // BLK, w // BLK
    return a[:ny * BLK, :nx * BLK].reshape(ny, BLK, nx, BLK).transpose(0, 2, 1, 3)


def _merge_blocks(b, shape):
    ny, nx = b.shape[:2]
    a = b.transpose(0, 2, 1, 3).reshape(ny * BLK, nx * BLK)
    out = np.zeros(shape, dtype=np.float64)
    out[:a.shape[0], :a.shape[1]] = a
    return out


# 2-D DCT as one 64x64 matrix: vec(D P D^T) = kron(D, D) vec(P)
_K = np.kron(_D, _D)
_KT = np.ascontiguousarray(_K.T)


def _dct2(p):
    sh = p.shape[:-2]
    return (p.reshape(-1, 64) @ _KT).reshape(sh + (8, 8))


def _idct2(c):
    sh = c.shape[:-2]
    return (c.reshape(-1, 64) @ _K).reshape(sh + (8, 8))


def _estimate_sigma(a):
    c = _dct2(_split_blocks(a))
    return float(np.median(np.abs(c[..., 4:, 4:]).ravel()) / 0.6745)


def denoise(img, sigma=None, k0=3.0, stability=0, spins=4):
    """降噪。

    img       : HxW 或 HxWxC，uint8 或 [0,1] float
    sigma     : 噪声 std（[0,1]）；None 时自动估计
    k0        : 阈值倍数（3.0 = 经典 3σ）
    stability : 0 = 纯主干（默认）；"quant" = 量化稳定度层
    spins     : 4 = 16 相位；1 = 单网格
    """
    a = to_float(img)
    single = a.ndim == 2
    if single:
        a = a[..., None]
    a = a.astype(np.float32)
    h, w, ch = a.shape

    if sigma is None:
        sig_c = [max(_estimate_sigma(a[..., c]), 1e-4) for c in range(ch)]
        sigma = float(np.mean(sig_c))
    else:
        sig_c = [max(float(sigma), 1e-4)] * ch
    sigma = max(float(sigma), 1e-4)

    if spins >= 8:
        shifts = [(y, x) for y in range(0, BLK) for x in range(0, BLK)]
    elif spins >= 4:
        shifts = [(y, x) for y in range(0, BLK, 2) for x in range(0, BLK, 2)]
    elif spins >= 2:
        shifts = [(0, 0), (0, BLK // 2), (BLK // 2, 0), (BLK // 2, BLK // 2)]
    else:
        shifts = [(0, 0)]

    smap = None
    if stability == "quant":
        sq = quant_stability(luma(a), sigma)
        lo, hi = np.percentile(sq, 20), np.percentile(sq, 80)
        smap = np.clip((sq - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    acc = np.zeros((h, w, ch), dtype=np.float64)
    for sy, sx in shifts:
        p = np.pad(a, ((sy, sy + BLK), (sx, sx + BLK), (0, 0)), mode="symmetric")
        ph, pw = p.shape[:2]
        ny, nx = ph // BLK, pw // BLK
        if smap is not None:
            sp = np.pad(smap, ((sy, sy + BLK), (sx, sx + BLK)), mode="symmetric")
            sb = sp[:ny * BLK, :nx * BLK].reshape(ny, BLK, nx, BLK).mean(axis=(1, 3))
            kmap = np.clip(k0 * (1.0 + 0.8 * (0.6 - sb)), 0.6 * k0, 1.7 * k0) * sigma
        else:
            kmap = np.full((ny, nx), k0 * sigma)
        for c in range(ch):
            sc = sig_c[c]
            c2 = _dct2(_split_blocks(p[..., c].astype(np.float64)))
            g = (np.abs(c2) > (kmap * (sc / sigma))[..., None, None]).astype(np.float64)
            g[..., 0, 0] = 1.0
            acc[..., c] += _merge_blocks(_idct2(g * c2), p.shape[:2])[sy:sy + h, sx:sx + w]

    out = np.clip(acc / float(len(shifts)), 0.0, 1.0).astype(np.float32)
    return out[..., 0] if single else out


# ------------------------------------------------------------- 对照与指标
def gaussian_denoise(img, sigma):
    a = to_float(img)
    return np.clip(gaussian_blur(a, float(np.clip(float(sigma) * 255.0 / 30.0, 0.8, 2.0))), 0, 1)


def median_denoise(img, size=3):
    from PIL import Image, ImageFilter
    im = Image.fromarray(to_uint8(to_float(img))).filter(ImageFilter.MedianFilter(size))
    return np.asarray(im).astype(np.float32) / 255.0


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


def ssim(a, b, sigma=1.5, k=(0.01, 0.03)):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.ndim == 3:
        a = luma(a.astype(np.float32)).astype(np.float64)
        b = luma(b.astype(np.float32)).astype(np.float64)
    c1, c2 = k[0] ** 2, k[1] ** 2
    mu1 = gaussian_blur(a.astype(np.float32), sigma).astype(np.float64)
    mu2 = gaussian_blur(b.astype(np.float32), sigma).astype(np.float64)
    s11 = gaussian_blur((a * a).astype(np.float32), sigma).astype(np.float64) - mu1 * mu1
    s22 = gaussian_blur((b * b).astype(np.float32), sigma).astype(np.float64) - mu2 * mu2
    s12 = gaussian_blur((a * b).astype(np.float32), sigma).astype(np.float64) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2)
    return float(np.mean(num / den))


# ------------------------------------------------------------------ CLI
def _main():
    import argparse
    import os
    from PIL import Image

    ap = argparse.ArgumentParser(description="NPSS-Image: 8x8 DCT 3σ + 16 相位 cycle-spin")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--sigma", type=float, default=25.0, help="噪声 std（0-255）")
    ap.add_argument("--estimate", action="store_true", help="自动估计噪声")
    ap.add_argument("--stability", choices=["off", "quant"], default="off")
    ap.add_argument("--spins", type=int, default=4)
    args = ap.parse_args()

    img = to_float(Image.open(args.input).convert("RGB"))
    if args.estimate:
        s = float(np.median(np.abs(luma(img) - gaussian_blur(luma(img), 1.2))) / 0.6745)
        print("estimated sigma %.1f/255" % (s * 255))
        sig = None
    else:
        sig = args.sigma / 255.0
    out = denoise(img, sigma=sig, spins=args.spins,
                  stability="quant" if args.stability == "quant" else 0)
    dst = args.output or os.path.splitext(args.input)[0] + "_npss.png"
    Image.fromarray(to_uint8(out)).save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    _main()
