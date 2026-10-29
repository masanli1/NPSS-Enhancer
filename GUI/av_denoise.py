#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频降噪：一个文件同时降画面和声音。

    输入视频
      ├─ 音轨 → ffmpeg 抽成 wav → NPSS 音频降噪（engine.denoise_array）
      └─ 画面 → ffmpeg 解成 rgb24 帧 → NPSS-Image 逐帧降噪
    两路再合成输出视频

用法：
    python3 av_denoise.py in.mp4 out.mp4                 # 自动
    python3 av_denoise.py in.mp4 out.mp4 --sigma 25 --audio-target 12
    python3 av_denoise.py in.mp4 out.mp4 --fast          # 单相位，快 4 倍

依赖：numpy、imageio_ffmpeg（自带 ffmpeg）。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import os as _os
# 单线程 BLAS：小矩阵/小批量下反而更快（Accelerate 多线程开销大）
for _k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C
import engine as E
import npss_image as IMG


def _ffmpeg():
    p = E.find_ffmpeg()
    if p:
        return p
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def probe(path):
    """Parse ffmpeg -i output -> dict(width, height, fps, has_audio, sr)."""
    ff = _ffmpeg()
    r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True)
    txt = r.stderr
    info = dict(width=None, height=None, fps=None, has_audio=False, sr=48000)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Audio:.*?(\d{4,6})\s*Hz", txt)
    if m:
        info["has_audio"] = True
        info["sr"] = int(m.group(1))
    else:
        info["has_audio"] = "Audio:" in txt
    return info


def _extract_audio(src, dst):
    ff = _ffmpeg()
    subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", dst],
                   check=True)


def _denoise_audio(src_wav, dst_wav, target_db=None, fast=False, content="voice", log=print):
    x, sr = C.read_wav_stereo(src_wav)
    chans = []
    for c in range(x.shape[1]):
        y = E.denoise_array(x[:, c], sr, multi_stat=True, fast=fast,
                            target_nr_db=target_db, content_mode=content)
        chans.append(y)
    y = np.stack(chans, axis=1)
    C.write_wav_stereo(dst_wav, y, sr)
    return y, sr


def _denoise_chunk(args):
    """Worker: 解码后的若干帧（uint8 bytes）-> 降噪后的 bytes。"""
    bufs, H, W, sigma, blocks, phases = args
    import numpy as np
    import npss_image as M
    out = []
    for b in bufs:
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        y, _ = M.denoise(f, sigma=sigma, blocks=blocks, phases=phases)
        out.append((np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    return out


def denoise_video(src, dst, sigma=None, audio_target=None, fast=False,
                  blocks=None, phases=None, max_frames=None, crf=18, content="voice",
                  workers=None, chunk=8, log=print):
    """同时降画面与声音。

    sigma       : 画面噪声 std（0-255 标度）；None = 自动估计
    audio_target: 音频目标降噪 dB（定果寻因）；None = 默认
    fast        : True 时画面用单相位（快约 4 倍）
    spins       : 画面相位密度（1/2/4），默认 fast ? 1 : 2
    """
    ff = _ffmpeg()
    info = probe(src)
    W, H, fps = info["width"], info["height"], info["fps"] or 25.0
    if not W or not H:
        raise RuntimeError("无法解析视频尺寸: %s" % src)
    if blocks is None:
        blocks = (8,)
    if phases is None:
        phases = 16        # 视频默认 16 相位：全滑动的 64 相位对长片太慢
    log("输入 %dx%d @ %.2f fps  音频=%s" % (W, H, fps, info["has_audio"]))

    tmp = tempfile.mkdtemp(prefix="npss_av_")
    awav = os.path.join(tmp, "in.wav")
    bwav = os.path.join(tmp, "out.wav")
    if info["has_audio"]:
        _extract_audio(src, awav)
        log("音频降噪中 ...")
        _denoise_audio(awav, bwav, target_db=audio_target, fast=fast,
                       content=content, log=log)
        log("音频完成")

    # 解码 -> 逐帧降噪 -> 编码
    dec = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-i", src,
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", "%.6f" % fps, "-i", "-"]
    if info["has_audio"]:
        cmd += ["-i", bwav]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if info["has_audio"]:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [dst]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    if workers is None:
        workers = max(1, min(8, os.cpu_count() or 4))
    sig = None if sigma is None else sigma / 255.0
    n = 0
    fsz = W * H * 3
    pool = None
    if workers > 1:
        import multiprocessing as _mp
        try:
            ctx = _mp.get_context("fork")     # 快，且不受 __main__ 重导入影响
        except ValueError:
            ctx = _mp.get_context()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    pending = []
    done_frames = 0

    def flush(oldest=False):
        nonlocal done_frames
        while pending and (oldest or len(pending) >= workers * 2):
            fut = pending.pop(0)
            for b in fut.result():
                enc.stdin.write(b)
                done_frames += 1
            if done_frames % 24 == 0 or oldest:
                log("  已处理 %d 帧" % done_frames)

    while True:
        bufs = []
        for _ in range(int(chunk)):
            buf = dec.stdout.read(fsz)
            if len(buf) < fsz:
                break
            bufs.append(buf)
            n += 1
            if max_frames and n >= max_frames:
                break
        if not bufs:
            break
        if pool is None:
            for b in _denoise_chunk((bufs, H, W, sig, blocks, phases)):
                enc.stdin.write(b)
                done_frames += 1
            if done_frames % 24 == 0:
                log("  已处理 %d 帧" % done_frames)
        else:
            pending.append(pool.submit(_denoise_chunk, (bufs, H, W, sig, blocks, phases)))
            flush()
        if max_frames and n >= max_frames:
            break
    flush(oldest=True)
    if pool is not None:
        pool.shutdown(wait=True)
    try:
        enc.stdin.close()
    except BrokenPipeError:
        pass
    if max_frames and n >= max_frames:
        dec.kill()
    else:
        dec.stdout.close()
    enc.wait()
    dec.wait()
    if info["has_audio"]:
        try:
            os.remove(awav); os.remove(bwav)
        except OSError:
            pass
    log("输出 %s（%d 帧）" % (dst, n))
    return dst


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sigma", type=float, default=None, help="画面噪声 std（0-255）")
    ap.add_argument("--audio-target", type=float, default=None, help="音频目标降噪 dB")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--quality", action="store_true",
                    help="画面用 4/8/16 多尺度融合（最慢，画质最高）")
    ap.add_argument("--spins", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--content", choices=["voice", "music"], default="voice",
                    help="音轨内容类型（默认 voice）")
    a = ap.parse_args()
    denoise_video(a.input, a.output, sigma=a.sigma, audio_target=a.audio_target,
                  fast=a.fast, blocks=(4, 8, 16) if a.quality else None,
                  phases=16 if a.fast else "full",
                  max_frames=a.max_frames,
                  content=a.content, log=lambda m: print(m, flush=True))
