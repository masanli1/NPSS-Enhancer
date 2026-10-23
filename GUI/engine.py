#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio/video denoising engine built on the best NPSS config (06)."""
import os
import sys
import shutil
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor
import nmf_blend

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import common_win as W
import multi_stat as MS

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".ts"}

_FFMPEG = None


def find_ffmpeg():
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    p = shutil.which("ffmpeg")
    if p:
        _FFMPEG = p
        return p
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        pass
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.exists(c):
            _FFMPEG = c
            return c
    return None


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_audio(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def run_ffmpeg(args, log=None):
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError(
            "找不到 ffmpeg。请安装 imageio-ffmpeg：\n"
            "  pip install imageio-ffmpeg\n"
            "或安装系统 ffmpeg。")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"] + list(args)
    if log:
        log("ffmpeg " + " ".join(args))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 失败：\n" + (r.stderr or r.stdout)[-1200:])
    return r


def extract_audio(src, dst, sr=16000, log=None):
    run_ffmpeg(["-i", src, "-vn", "-ac", "1", "-ar", str(sr),
                "-c:a", "pcm_s16le", dst], log)


def resample(src, dst, sr, log=None):
    run_ffmpeg(["-i", src, "-ac", "1", "-ar", str(sr),
                "-c:a", "pcm_s16le", dst], log)


def mux_video(video, audio, out, log=None):
    # copy video stream when possible; fall back to H.264 re-encode
    try:
        run_ffmpeg(["-i", video, "-i", audio,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", out], log)
    except RuntimeError:
        log("视频流复制失败，改为 H.264 重编码……")
        run_ffmpeg(["-i", video, "-i", audio,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", out], log)


def _process_chunk(args):
    (seg, sr, family, backend, n_fft, hop, passes, theta, support,
     iterations, inject, auto, content_mode, complement, loops, multi_stat) = args
    if multi_stat:
        y = MS.denoise_best(seg, sr, voice=(content_mode == "voice"))
        if len(y) != len(seg):
            y = (y[:len(seg)] if len(y) > len(seg)
                 else np.pad(y, (0, len(seg) - len(y))))
        return y.astype(np.float32)
    y, snr_in, S1, _ = W.cascade_contrast_win(
        seg, sr, family, wb=2, wc=4, passes=passes, theta=theta,
        support=support, backend=backend, n_fft=n_fft, hop=hop)
    if complement and sr == 16000:
        Xp, Mp, gb, _, _, _ = C.prepare(seg, n_fft, hop, 1.5, 0.05, backend)
        S_list = [S1]
        g = C.contrast_gate(S1, 30, 90, 0.1)
        y = C.istft(gb * g * Xp, len(seg), n_fft, hop)
        for _ in range(2, max(2, int(loops)) + 1):
            y2, _, S_new, _ = W.cascade_contrast_win(
                y, sr, family, wb=2, wc=4, passes=passes, theta=theta,
                support=support, backend=backend, n_fft=n_fft, hop=hop)
            S_list.append(S_new)
            S = np.mean(S_list, axis=0)
            g = C.contrast_gate(S, 30, 90, 0.3)
            y = C.istft(gb * g * Xp, len(seg), n_fft, hop)
    else:
        if not (auto and snr_in > 12.0):
            for _ in range(max(0, iterations - 1)):
                y, _, _, _ = W.cascade_contrast_win(
                    y, sr, family, wb=2, wc=4, passes=passes, theta=theta,
                    support=support, backend=backend, n_fft=n_fft, hop=hop)
            if inject > 0:
                y = np.clip((1.0 - inject) * y + inject * seg,
                            -1.0, 1.0).astype(np.float32)
                y, _, _, _ = W.cascade_contrast_win(
                    y, sr, family, wb=2, wc=4, passes=passes, theta=theta,
                    support=support, backend=backend, n_fft=n_fft, hop=hop)
    if content_mode == "voice" and sr == 16000:
        try:
            yn = nmf_blend.enhance(seg)
            m = min(len(y), len(yn))
            y = np.clip(0.6 * y[:m] + 0.4 * yn[:m], -1.0, 1.0).astype(np.float32)
        except Exception:
            pass
    if len(y) != len(seg):
        y = (y[:len(seg)] if len(y) > len(seg)
             else np.pad(y, (0, len(seg) - len(y))))
    return y.astype(np.float32)


def denoise_array(x, sr, family="correlated", passes=3, wb=2, wc=4,
                  theta=0.95, support=0.60, backend="specsub",
                  n_fft=2048, hop=512, chunk_sec=20.0, overlap_sec=1.0,
                  iterations=1, inject=0.0, auto=True, workers=0,
                  content_mode="music", complement=True, loops=2,
                  multi_stat=True, progress=None):
    """Chunked denoise with process-level parallelism for long files."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = len(x)
    if n == 0:
        return x
    iterations = max(1, min(5, int(iterations)))
    chunk = int(max(5.0, float(chunk_sec)) * sr)
    args0 = (x, sr, family, backend, n_fft, hop, passes, theta, support,
             iterations, inject, auto, content_mode, complement, loops, multi_stat)
    if n <= chunk:
        y = _process_chunk(args0)
        if progress:
            progress(1, 1)
        return y.astype(np.float32)

    overlap = int(max(0.3, min(float(overlap_sec), float(chunk_sec) * 0.5)) * sr)
    step = max(1, chunk - overlap)
    starts = list(range(0, n, step))
    total = len(starts)
    args_list = []
    for s in starts:
        e = min(n, s + chunk)
        args_list.append((x[s:e], sr, family, backend, n_fft, hop, passes,
                          theta, support, iterations, inject, auto, content_mode,
                          complement, loops, multi_stat))
    if workers <= 0:
        workers = min(os.cpu_count() or 4, len(args_list), 8)
    workers = max(1, min(int(workers), len(args_list), 16))

    results = []
    if workers == 1:
        for i, a in enumerate(args_list):
            results.append(_process_chunk(a))
            if progress:
                progress(i + 1, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, ys in enumerate(ex.map(_process_chunk, args_list)):
                results.append(ys)
                if progress:
                    progress(i + 1, total)

    y = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for s, ys in zip(starts, results):
        e = min(n, s + chunk)
        w = np.ones(e - s, dtype=np.float64)
        ov = min(overlap, (e - s) // 2)
        if s > 0 and ov > 0:
            w[:ov] = np.linspace(0.0, 1.0, ov)
        if e < n and ov > 0:
            w[-ov:] = np.linspace(1.0, 0.0, ov)
        y[s:e] += ys * w
        wsum[s:e] += w
    y /= np.maximum(wsum, 1e-8)
    return y.astype(np.float32)


def process_file(src, dst, mode="auto", family="correlated", passes=3,
                 chunk_sec=20.0, proc_sr=16000, out_sr=None,
                 theta=0.95, support=0.60, backend="specsub",
                 iterations=1, inject=0.0, auto=True, workers=0,
                 content_mode="music", complement=True, loops=2,
                 multi_stat=True, progress=None, log=None):
    """Process an audio or video file. mode: auto/audio/video."""
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if not find_ffmpeg():
        raise RuntimeError("找不到 ffmpeg；请先 pip install imageio-ffmpeg")
    proc_sr = int(proc_sr) if proc_sr else 16000
    video = (mode == "video") or (mode == "auto" and is_video(src))
    tmp = tempfile.mkdtemp(prefix="npss_gui_")
    try:
        in16 = os.path.join(tmp, "in16.wav")
        out16 = os.path.join(tmp, "out16.wav")
        if log:
            log("提取音频 → %d Hz 单声道" % proc_sr)
        extract_audio(src, in16, proc_sr, log)
        x, sr = C.read_wav(in16)
        if log:
            if multi_stat:
                log("开始降噪：%.1f s，多统计 6 families + single，3-pass/2-loop" % (len(x) / sr))
            else:
                log("开始降噪：%.1f s，%s，%d 次 pass" % (len(x) / sr, family, passes))
        n_fft = 2048 if proc_sr <= 16000 else 4096
        hop = n_fft // 4
        y = denoise_array(x, sr, family=family, passes=passes,
                          n_fft=n_fft, hop=hop,
                          chunk_sec=chunk_sec, iterations=iterations,
                          inject=inject, auto=auto, theta=theta,
                          support=support, backend=backend,
                          workers=workers, content_mode=content_mode,
                          complement=complement, loops=loops,
                          multi_stat=multi_stat, progress=progress)
        C.write_wav(out16, y, proc_sr)
        if video:
            if proc_sr == 48000:
                out48 = out16
            else:
                out48 = os.path.join(tmp, "out48.wav")
                if log:
                    log("重采样到 48 kHz 并封装视频")
                resample(out16, out48, 48000, log)
            mux_video(src, out48, dst, log)
        else:
            target = int(out_sr) if out_sr else proc_sr
            if target != proc_sr:
                resample(out16, dst, target, log)
            else:
                shutil.copyfile(out16, dst)
        if log:
            log("完成：" + dst)
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
