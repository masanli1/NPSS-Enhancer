#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio/video denoising engine built on the best NPSS config (06)."""
import os
import re
import sys
import shutil
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import nmf_blend

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import common_win as W
import multi_stat as MS

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".m4b", ".aac", ".ogg", ".oga",
    ".opus", ".aiff", ".aif", ".aifc", ".wma", ".mka", ".ac3", ".eac3",
    ".mp2", ".caf", ".au", ".snd", ".ape", ".wv",
}
VIDEO_EXTS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".mpe", ".m2v", ".3gp", ".3g2",
    ".ogv", ".vob", ".wmv", ".asf", ".f4v", ".mxf", ".dv", ".rm", ".rmvb",
}
AUDIO_OUT_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a", ".m4b", ".aac", ".ogg", ".oga",
    ".opus", ".aiff", ".aif", ".aifc", ".wma", ".mka", ".ac3", ".eac3",
    ".mp2", ".caf", ".au", ".snd",
}
VIDEO_OUT_EXTS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts",
    ".m2ts", ".mpg", ".mpeg", ".3gp", ".3g2", ".ogv", ".wmv", ".asf",
}

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


_AUDIO_ENCODERS = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".flac": ["-c:a", "flac"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    ".m4a": ["-c:a", "aac", "-b:a", "256k"],
    ".m4b": ["-c:a", "aac", "-b:a", "256k"],
    ".aac": ["-c:a", "aac", "-b:a", "256k", "-f", "adts"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
    ".oga": ["-c:a", "libvorbis", "-q:a", "6"],
    ".opus": ["-c:a", "libopus", "-b:a", "160k"],
    ".aiff": ["-c:a", "pcm_s16be"],
    ".aif": ["-c:a", "pcm_s16be"],
    ".aifc": ["-c:a", "pcm_s16be"],
    ".wma": ["-c:a", "wmav2", "-b:a", "192k"],
    ".mka": ["-c:a", "flac"],
    ".ac3": ["-c:a", "ac3", "-b:a", "192k"],
    ".eac3": ["-c:a", "eac3", "-b:a", "192k"],
    ".mp2": ["-c:a", "mp2", "-b:a", "192k"],
    ".caf": ["-c:a", "pcm_s16le"],
    ".au": ["-c:a", "pcm_s16be"],
    ".snd": ["-c:a", "pcm_s16be"],
}


def probe_audio_sr(src):
    """Best-effort sample rate from ffmpeg's stream info (no ffprobe needed)."""
    ff = find_ffmpeg()
    if not ff:
        return None
    r = subprocess.run([ff, "-hide_banner", "-i", src], capture_output=True, text=True)
    m = re.search(r"Audio:.*?(\d+) Hz", r.stderr or "")
    return int(m.group(1)) if m else None


def encode_audio(src_wav, dst, sr, log=None):
    """Encode a processed WAV into the container implied by dst's extension."""
    ext = os.path.splitext(dst)[1].lower()
    if not ext:
        dst += ".wav"
        ext = ".wav"
    enc = _AUDIO_ENCODERS.get(ext)
    if enc is None:
        raise RuntimeError("不支持的音频输出格式：%s（支持 %s）" % (
            ext, " ".join(sorted(AUDIO_OUT_EXTS))))
    run_ffmpeg(["-i", src_wav, "-ar", str(int(sr)), "-ac", "2"] + enc + [dst], log)
    return dst


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


def extract_audio(src, dst, sr=16000, log=None, channels=1):
    run_ffmpeg(["-i", src, "-vn", "-ac", str(int(channels)), "-ar", str(sr),
                "-c:a", "pcm_s16le", dst], log)


def resample(src, dst, sr, log=None, channels=2):
    run_ffmpeg(["-i", src, "-ac", str(int(channels)), "-ar", str(sr),
                "-c:a", "pcm_s16le", dst], log)


def mux_video(video, audio, out, log=None):
    """Mux the new audio into a video container, choosing compatible codecs."""
    ext = os.path.splitext(out)[1].lower() or ".mp4"
    if ext not in VIDEO_OUT_EXTS:
        raise RuntimeError("不支持的视频输出格式：%s（支持 %s）" % (
            ext, " ".join(sorted(VIDEO_OUT_EXTS))))
    acodec = {
        ".webm": ["-c:a", "libopus", "-b:a", "160k"],
        ".avi": ["-c:a", "libmp3lame", "-b:a", "192k"],
        ".mpg": ["-c:a", "mp2", "-b:a", "192k"],
        ".mpeg": ["-c:a", "mp2", "-b:a", "192k"],
        ".ogv": ["-c:a", "libvorbis", "-q:a", "6"],
        ".wmv": ["-c:a", "wmav2", "-b:a", "192k"],
        ".asf": ["-c:a", "wmav2", "-b:a", "192k"],
    }.get(ext, ["-c:a", "aac", "-b:a", "192k"])
    vcodec = {
        ".webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1"],
        ".avi": ["-c:v", "mpeg4", "-q:v", "4"],
        ".mpg": ["-c:v", "mpeg2video", "-q:v", "4"],
        ".mpeg": ["-c:v", "mpeg2video", "-q:v", "4"],
        ".ogv": ["-c:v", "libtheora", "-q:v", "6"],
        ".wmv": ["-c:v", "wmv2", "-q:v", "4"],
        ".asf": ["-c:v", "wmv2", "-q:v", "4"],
    }.get(ext, ["-c:v", "libx264", "-crf", "18", "-preset", "medium"])
    copy_ok = ext in {".mp4", ".m4v", ".mov", ".mkv", ".flv", ".ts",
                      ".m2ts", ".3gp", ".3g2", ".f4v", ".mxf"}
    if copy_ok:
        try:
            run_ffmpeg(["-i", video, "-i", audio,
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy"] + acodec + ["-shortest", out], log)
            return out
        except RuntimeError:
            if log:
                log("视频流复制失败，改为重编码 %s……" % ext)
    run_ffmpeg(["-i", video, "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0"] + vcodec + acodec +
               ["-shortest", out], log)
    return out


def _process_chunk(args):
    (seg, sr, family, backend, n_fft, hop, passes, theta, support,
     iterations, inject, auto, content_mode, complement, loops, multi_stat,
     fast, target_nr_db, target_dose, target_weights, view_profile,
     view_beta) = args
    info = {}
    if multi_stat:
        y = MS.denoise_best(seg, sr, voice=(content_mode == "voice"), fast=fast,
                            target_nr_db=target_nr_db,
                            target_dose=target_dose,
                            target_weights=target_weights, info_out=info,
                            view_profile=view_profile, view_beta=view_beta)
        if len(y) != len(seg):
            y = (y[:len(seg)] if len(y) > len(seg)
                 else np.pad(y, (0, len(seg) - len(y))))
        return y.astype(np.float32), info
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
    return y.astype(np.float32), info


def denoise_array(x, sr, family="correlated", passes=3, wb=2, wc=4,
                  theta=0.95, support=0.60, backend="specsub",
                  n_fft=2048, hop=512, chunk_sec=20.0, overlap_sec=1.0,
                  iterations=1, inject=0.0, auto=True, workers=0,
                  content_mode="music", complement=True, loops=2,
                  multi_stat=True, fast=False, progress=None,
                  target_nr_db=None, info_out=None,
                  view_profile="freq", view_beta=0.5):
    """Chunked denoise with process-level parallelism for long files.

    ``target_nr_db`` enables 定果寻因: a one-slice calibration solves the dose,
    then every chunk is processed with that same dose, so the noise floor lands
    on the target without chunk-to-chunk pumping.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = len(x)
    if n == 0:
        return x
    if target_nr_db is not None and not multi_stat:
        print("警告：目标降噪只在多统计 Gmin 下可用，已忽略目标 %.1f dB"
              % float(target_nr_db), file=sys.stderr)
        target_nr_db = None
    iterations = max(1, min(5, int(iterations)))
    chunk = int(max(5.0, float(chunk_sec)) * sr)
    target_dose = None
    target_weights = None
    if target_nr_db is not None and multi_stat and n > chunk:
        import target_nr as TG
        cal = x[max(0, n // 2 - 4 * sr): n // 2 + 4 * sr]
        dose, wts, cinfo = TG.calibrate(cal, sr, target_db=float(target_nr_db),
                                       fast=fast)
        target_dose, target_weights = float(dose), dict(wts)
        if progress:
            progress(0, 1)
        if info_out is not None:
            info_out.clear()
            info_out.update({"calibration": cinfo, "dose": target_dose,
                             "weights": target_weights, "target_db": float(target_nr_db)})
    args0 = (x, sr, family, backend, n_fft, hop, passes, theta, support,
             iterations, inject, auto, content_mode, complement, loops, multi_stat,
             fast, target_nr_db, target_dose, target_weights,
             view_profile, view_beta)
    if n <= chunk:
        y, info = _process_chunk(args0)
        if info_out is not None:
            info_out.clear()
            info_out.update(info)
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
                          complement, loops, multi_stat, fast,
                          target_nr_db, target_dose, target_weights,
                          view_profile, view_beta))
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
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, ys in enumerate(ex.map(_process_chunk, args_list)):
                    results.append(ys)
                    if progress:
                        progress(i + 1, total)
        except Exception:
            # Frozen app / platform fallback: run serially if the pool dies.
            results = []
            for i, a in enumerate(args_list):
                results.append(_process_chunk(a))
                if progress:
                    progress(i + 1, total)

    y = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for s, (ys, _info) in zip(starts, results):
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
    if info_out is not None and "achieved_db" not in info_out:
        for _, inf in results:
            if inf:
                info_out.update({k: v for k, v in inf.items() if k != "views"})
                break
    return y.astype(np.float32)


def process_file(src, dst, mode="auto", family="correlated", passes=3,
                 chunk_sec=20.0, proc_sr=16000, out_sr=None,
                 theta=0.95, support=0.60, backend="specsub",
                 iterations=1, inject=0.0, auto=True, workers=0,
                 content_mode="music", complement=True, loops=2,
                 multi_stat=True, fast=False, progress=None, log=None,
                 target_nr_db=None, info_out=None,
                 view_profile="freq", view_beta=0.5):
    """Process an audio or video file. mode: auto/audio/video.

    ``target_nr_db`` turns on 定果寻因: instead of fixed thresholds, the 16 kHz
    noise floor is driven to that many dB of attenuation (see target_nr.py).
    """
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
            log("提取音频 → %d Hz 立体声" % proc_sr)
        extract_audio(src, in16, proc_sr, log, channels=2)
        x, sr = C.read_wav_stereo(in16)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] > 2:
            x = x[:, :2]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if log:
            if multi_stat:
                log("开始降噪：%.1f s，多统计 6 families + single，3-pass/2-loop" % (len(x) / sr))
            else:
                log("开始降噪：%.1f s，%s，%d 次 pass" % (len(x) / sr, family, passes))
        n_fft = 2048 if proc_sr <= 16000 else 4096
        hop = n_fft // 4
        if np.allclose(x[:, 0], x[:, 1]):
            y0 = denoise_array(x[:, 0], sr, family=family, passes=passes,
                               n_fft=n_fft, hop=hop,
                               chunk_sec=chunk_sec, iterations=iterations,
                               inject=inject, auto=auto, theta=theta,
                               support=support, backend=backend,
                               workers=workers, content_mode=content_mode,
                               complement=complement, loops=loops,
                               multi_stat=multi_stat, progress=progress,
                               target_nr_db=target_nr_db, info_out=info_out,
                               view_profile=view_profile, view_beta=view_beta)
            y = np.repeat(y0[:, None], 2, axis=1).astype(np.float32)
        else:
            def run_ch(ci):
                def cp(d, t, _ci=ci):
                    if progress:
                        progress(_ci + (d / max(1, t)), 2)
                return denoise_array(x[:, ci], sr, family=family, passes=passes,
                                     n_fft=n_fft, hop=hop,
                                     chunk_sec=chunk_sec, iterations=iterations,
                                     inject=inject, auto=auto, theta=theta,
                                     support=support, backend=backend,
                                     workers=max(1, workers // 2),
                                     content_mode=content_mode,
                                     complement=complement, loops=loops,
                                     multi_stat=multi_stat, fast=fast,
                                     progress=cp,
                                     target_nr_db=target_nr_db,
                                     info_out=info_out,
                                     view_profile=view_profile,
                                     view_beta=view_beta)
            with ThreadPoolExecutor(max_workers=2) as ex:
                ys = list(ex.map(run_ch, [0, 1]))
            nmin = min(len(ys[0]), len(ys[1]), len(x))
            y = np.stack([ys[0][:nmin], ys[1][:nmin]], axis=1).astype(np.float32)
        C.write_wav_stereo(out16, y, proc_sr)
        if video:
            if proc_sr == 48000:
                out48 = out16
            else:
                out48 = os.path.join(tmp, "out48.wav")
                if log:
                    log("重采样到 48 kHz 并封装视频")
                resample(out16, out48, 48000, log, channels=2)
            mux_video(src, out48, dst, log)
        else:
            target = int(out_sr) if out_sr else proc_sr
            if target != proc_sr:
                tmp_target = os.path.join(tmp, "out_target.wav")
                resample(out16, tmp_target, target, log)
                encode_audio(tmp_target, dst, target, log)
            else:
                encode_audio(out16, dst, target, log)
        if log:
            if info_out and "achieved_db" in info_out:
                log("目标 %.1f dB → 实测 %.2f dB（dose c=%.3f，%s）" % (
                    float(info_out.get("target_db") or 0.0),
                    float(info_out["achieved_db"]),
                    float(info_out.get("dose") or 0.0),
                    "命中" if info_out.get("hit") else "未命中"))
            log("完成：" + dst)
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def process_video_av(src, dst, sigma=None, audio_target=None, fast=False,
                     spins=None, max_frames=None, crf=18, content="voice",
                     log=None):
    """视频一体化降噪：画面 + 声音同时处理。

    画面用 NPSS-Image（8x8 DCT 3σ + cycle-spin），声音用本文件的
    denoise_array（NPSS 多统计 Gmin）。实现见 ``av_denoise.py``。

    sigma        : 画面噪声 std（0-255 标度）；None = 自动估计
    audio_target : 音频目标降噪 dB（定果寻因）；None = 默认
    fast         : True 时画面单相位（快约 4 倍）
    content      : 音轨类型 "voice" / "music"
    """
    import av_denoise
    return av_denoise.denoise_video(
        src, dst, sigma=sigma, audio_target=audio_target, fast=fast,
        spins=spins, max_frames=max_frames, crf=crf, content=content,
        log=log or (lambda m: print(m, flush=True)))
