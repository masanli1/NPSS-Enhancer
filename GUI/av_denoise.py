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
    info = dict(width=None, height=None, fps=None, has_audio=False, sr=48000,
                duration=None)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
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



def auto_phases(info, workers, budget=600.0):
    """按片长/分辨率/预算自动挑相位密度（越大越无接缝、越慢）。"""
    dur = info.get("duration") or 0.0
    fps = info.get("fps") or 25.0
    w = info.get("width") or 1280
    h = info.get("height") or 720
    frames = max(1.0, dur * fps)
    mp = (w * h) / (768.0 * 512.0)          # 相对 768x512 的像素倍数
    # 768x512 彩色 64 相位约 0.9 s/帧（单进程），并行效率按 0.8 计
    per_frame_64 = 0.9 * mp
    eff = max(1.0, workers * 0.8)
    for ph in (64, 32, 16, 8, 4):
        est = frames * per_frame_64 * (ph / 64.0) / eff
        if est <= budget:
            return ph, est
    return 4, frames * per_frame_64 * (4 / 64.0) / eff


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
    if phases is None or phases == "auto":
        phases, est = auto_phases(info, max(1, workers or 8))
        log("画面相位：自动选 %d（预估 %.0f 秒完成画面）" % (phases, est))
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
    ap.add_argument("--still", type=float, default=None,
                    help="视频取帧：在该时间点(秒)融合连续帧，输出一张图片")
    ap.add_argument("--still-frames", type=int, default=0,
                    help="融合帧数（0=按帧间运动自动，3~24）")
    a = ap.parse_args()
    if a.still is not None:
        from PIL import Image
        import still_merge as SM
        img, info = SM.still(a.input, t_sec=float(a.still),
                             count=(a.still_frames or None),
                             sigma=None if a.sigma is None else a.sigma / 255.0,
                             log=lambda m: print(m, flush=True))
        Image.fromarray(IMG.to_uint8(img)).save(a.output)
        print("已保存图片", a.output)
        raise SystemExit(0)
    denoise_video(a.input, a.output, sigma=a.sigma, audio_target=a.audio_target,
                  fast=a.fast, blocks=(4, 8, 16) if a.quality else None,
                  phases=("full" if a.quality else (16 if a.fast else None)),
                  max_frames=a.max_frames,
                  content=a.content, log=lambda m: print(m, flush=True))


def denoise_video_traj(src, dst, sigma=None, audio_target=None, radius=3,
                       content="voice", fast=False, max_frames=None,
                       crf=18, log=print):
    """沿轨迹累积的视频降噪（流式分块，内存只保留 2r+1 帧）。

    解码 → 逐帧算光流 → 环形缓冲 → 输出 i-radius 帧 → 编码。
    不做前景/背景分割：运动主体与背景同样参与累积。
    """
    import traj_denoise as T
    from collections import deque

    info = probe(src)
    W, H = info["width"], info["height"]
    fps = info["fps"] or 25.0
    ff = _ffmpeg()
    nbytes = H * W * 3

    def decode_cmd(limit):
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", src,
               "-f", "rawvideo", "-pix_fmt", "rgb24"]
        if limit:
            cmd += ["-frames:v", str(int(limit))]
        cmd += ["-"]
        return cmd

    # 自动估噪：先解 8 帧，取静态像素的时域标准差
    if sigma is None:
        probe_raw = subprocess.run(decode_cmd(8), capture_output=True).stdout
        m = len(probe_raw) // nbytes
        fs = [np.frombuffer(probe_raw[i * nbytes:(i + 1) * nbytes], np.uint8)
              .reshape(H, W, 3).astype(np.float32) / 255.0 for i in range(m)]
        sigma = max(0.02, float(np.percentile(np.stack(fs).std(0).mean(-1), 10))) if m > 1 else 0.02
        log("自动估噪 σ≈%.1f/255" % (sigma * 255))

    tmp = tempfile.mkdtemp(prefix="npss_traj_")
    silent = os.path.join(tmp, "v.mp4")
    enc = subprocess.Popen(
        [ff, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
         "-r", "%.3f" % fps, "-i", "-", "-c:v", "libx264", "-crf", str(crf),
         "-pix_fmt", "yuv420p", silent],
        stdin=subprocess.PIPE)

    dec = subprocess.Popen(decode_cmd(max_frames), stdout=subprocess.PIPE)
    buf_f = deque(maxlen=2 * radius + 1)
    buf_g = deque(maxlen=2 * radius)          # 光流：帧 i -> i+1
    written = 0
    i = 0
    while True:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        fr = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        if buf_f:
            buf_g.append(T.bm_flow(buf_f[-1], fr))
        buf_f.append(fr)
        if len(buf_f) == 2 * radius + 1:
            out = T.accumulate_weighted(list(buf_f), list(buf_g), radius, radius, sigma)
            enc.stdin.write((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
            written += 1
            if written % 50 == 0:
                log("  已输出 %d 帧" % written)
        i += 1
    dec.wait()

    # 尾部：剩余的 r 帧用缓冲里已有的邻域尽量累积
    base = i - len(buf_f)
    for idx in range(written, i):
        local = idx - base
        out = T.accumulate_weighted(list(buf_f), list(buf_g), local, radius, sigma)
        enc.stdin.write((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        written += 1

    enc.stdin.close()
    enc.wait()

    src_wav = os.path.join(tmp, "a.wav")
    if _extract_audio(src, src_wav):
        out_wav = os.path.join(tmp, "a_out.wav")
        _denoise_audio(src_wav, out_wav, audio_target, fast, content, log)
        mux = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", silent,
               "-i", out_wav, "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", dst]
    else:
        mux = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", silent,
               "-c:v", "copy", dst]
    subprocess.run(mux, check=False)
    log("输出 %s（%d 帧）" % (dst, written))
    return dst
