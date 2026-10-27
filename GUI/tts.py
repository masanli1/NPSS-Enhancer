#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
# -*- coding: utf-8 -*-
"""Self-built Mandarin formant TTS (pypinyin for G2P only; no external TTS)."""
import re
import numpy as np
from pypinyin import pinyin, Style

SR = 48000

VOICES = {
    "女声": {"f0": 205.0, "scale": 1.00},
    "男声": {"f0": 118.0, "scale": 0.90},
    "低沉": {"f0": 100.0, "scale": 0.85},
    "明亮": {"f0": 235.0, "scale": 1.08},
}

VOWELS = {
    "a": (800.0, 1200.0, 2500.0),
    "o": (500.0, 800.0, 2500.0),
    "e": (500.0, 1500.0, 2500.0),
    "i": (300.0, 2300.0, 3000.0),
    "u": (300.0, 800.0, 2500.0),
    "v": (300.0, 2200.0, 2500.0),
    "apical": (400.0, 1300.0, 2500.0),
    "n": (300.0, 1200.0, 2500.0),
    "ng": (300.0, 900.0, 2300.0),
    "r": (500.0, 1500.0, 2000.0),
}

FINALS = {
    "a": ["a"], "o": ["o"], "e": ["e"], "i": ["i"], "u": ["u"], "v": ["v"],
    "er": ["e", "r"],
    "ai": ["a", "i"], "ei": ["e", "i"], "ao": ["a", "u"], "ou": ["o", "u"],
    "an": ["a", "n"], "en": ["e", "n"], "ang": ["a", "ng"], "eng": ["e", "ng"],
    "ong": ["o", "ng"],
    "ia": ["i", "a"], "ie": ["i", "e"], "iao": ["i", "a", "u"],
    "iu": ["i", "o", "u"], "ian": ["i", "e", "n"], "in": ["i", "n"],
    "iang": ["i", "a", "ng"], "ing": ["i", "ng"], "iong": ["i", "o", "ng"],
    "ua": ["u", "a"], "uo": ["u", "o"], "uai": ["u", "a", "i"],
    "ui": ["u", "e", "i"], "uan": ["u", "a", "n"], "un": ["u", "e", "n"],
    "uang": ["u", "a", "ng"], "ueng": ["u", "e", "ng"],
    "ve": ["v", "e"], "van": ["v", "a", "n"], "vn": ["v", "n"],
    "apical": ["apical"],
}

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

VOICED = {"m", "n", "l", "r"}
PLOSIVE = {"b", "d", "g"}
ASPIRATED = {"p", "t", "k"}
FRICATIVE = {"f", "h", "x", "s", "sh"}
AFFRICATE = {"j", "q", "zh", "ch", "z", "c"}
GLIDE = {"y", "w"}

TONES = {
    1: [1.00, 1.00],
    2: [0.88, 1.20],
    3: [1.00, 0.82, 1.12],
    4: [1.22, 0.78],
    5: [1.00, 1.00],
}


def list_voices():
    return [(name, "zh_CN", "formant") for name in VOICES]


def default_voice(voices=None):
    return "女声"


def _resonate(x, f, bw, sr):
    """Time-varying 2nd-order resonator."""
    N = len(x)
    y = np.zeros(N, dtype=np.float64)
    r = np.exp(-np.pi * float(bw) / sr)
    a1 = -2.0 * r * np.cos(2.0 * np.pi * f / sr)
    a2 = r * r
    b0 = 1.0 - a1 - a2
    y1 = y2 = 0.0
    for n in range(N):
        v = b0[n] * x[n] - a1[n] * y1 - a2 * y2
        y[n] = v
        y2 = y1
        y1 = v
    return y


def _track(points, n):
    """Linear interpolate formant targets over n samples."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return np.full(n, pts[0, 0]), np.full(n, pts[0, 1]), np.full(n, pts[0, 2])
    src = np.linspace(0, 1, len(pts))
    dst = np.linspace(0, 1, n)
    f1 = np.interp(dst, src, pts[:, 0])
    f2 = np.interp(dst, src, pts[:, 1])
    f3 = np.interp(dst, src, pts[:, 2])
    return f1, f2, f3


def _syllable(base, initial, final, tone, voice):
    prof = VOICES.get(voice, VOICES["女声"])
    scale = prof["scale"]
    f0base = prof["f0"]

    # final formant targets (with nasal endings)
    parts = FINALS.get(final, ["e"])
    pts = [VOWELS.get(p, VOWELS["e"]) for p in parts]
    nfin = int(0.20 * SR)
    # onset
    if initial in VOICED:
        onset_type, onset_dur = "voiced", 0.055
        onset_pts = [(300, 1000, 2400)]
    elif initial in PLOSIVE:
        onset_type, onset_dur = "plosive", 0.035
        onset_pts = [(600, 1500, 3000)]
    elif initial in ASPIRATED:
        onset_type, onset_dur = "aspirated", 0.070
        onset_pts = [(700, 1800, 3400)]
    elif initial in FRICATIVE:
        onset_type, onset_dur = "fricative", 0.075
        onset_pts = [(1200, 2600, 4200)] if initial in {"f", "h"} else [(3000, 4500, 6000)]
    elif initial in AFFRICATE:
        onset_type, onset_dur = "affricate", 0.070
        onset_pts = [(1800, 3200, 4800)] if initial in {"j", "q", "zh", "ch"} else [(3500, 5200, 6800)]
    elif initial in GLIDE:
        onset_type, onset_dur = "voiced", 0.045
        onset_pts = [pts[0]] if pts else [VOWELS["i"]]
    else:
        onset_type, onset_dur = "silent", 0.0
        onset_pts = [pts[0]] if pts else [VOWELS["e"]]

    non = int(onset_dur * SR)
    total = non + nfin
    # formant tracks
    if non > 0:
        o1, o2, o3 = _track(onset_pts, non)
        v1, v2, v3 = _track(pts, nfin)
        f1 = np.concatenate([o1, v1]) * scale
        f2 = np.concatenate([o2, v2]) * scale
        f3 = np.concatenate([o3, v3]) * scale
    else:
        v1, v2, v3 = _track(pts, nfin)
        f1, f2, f3 = v1 * scale, v2 * scale, v3 * scale

    # source
    src = np.zeros(total, dtype=np.float64)
    seed = (sum(map(ord, base)) * 131 + int(tone) * 17) % (2 ** 32)
    rng = np.random.default_rng(seed)
    if onset_type == "fricative":
        src[:non] = rng.normal(0, 0.25, non)
    elif onset_type in ("affricate", "plosive", "aspirated"):
        cut = int(0.018 * SR) if onset_type in ("affricate", "plosive") else int(0.012 * SR)
        src[cut:non] = rng.normal(0, 0.30, max(0, non - cut))
    # voiced part: pulse train with tone contour
    tone_curve = TONES.get(int(tone), TONES[5])
    tcur = np.interp(np.linspace(0, 1, max(1, total - non)),
                     np.linspace(0, 1, len(tone_curve)), tone_curve)
    f0 = f0base * tcur
    phase = 0.0
    for n in range(non, total):
        phase += f0[n - non] / SR
        if phase >= 1.0:
            src[n] += 1.0
            phase -= 1.0

    # formant filtering
    y = _resonate(src, f1, 90.0, SR)
    y = _resonate(y, f2, 110.0, SR)
    y = _resonate(y, f3, 140.0, SR)

    # envelope
    env = np.ones(total, dtype=np.float64)
    a = min(int(0.008 * SR), total // 4)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
        env[-a:] = np.linspace(1, 0, a)
    y *= env
    y /= (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def _parse(ch):
    try:
        py = pinyin(ch, style=Style.TONE3, strict=False)[0][0]
    except Exception:
        return None
    if not py:
        return None
    m = re.search(r"(\d)$", py)
    tone = int(m.group(1)) if m else 5
    base = py[:-1] if m else py
    base = base.replace("ü", "v")
    initial = ""
    for ini in INITIALS:
        if base.startswith(ini):
            initial = ini
            break
    final = base[len(initial):]
    if base in {"zhi", "chi", "shi", "ri", "zi", "ci", "si"}:
        final = "apical"
    if final not in FINALS:
        # try common normalisations
        if final.startswith("i") and final not in FINALS:
            final = "i"
        elif final.startswith("u") and final not in FINALS:
            final = "u"
        else:
            final = "e"
    return base, initial, final, tone


def synth(text, voice="女声", rate=170, out_wav="out.wav", sr=SR):
    """Mandarin formant TTS. rate maps roughly to 80..320."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")
    speed = max(0.55, min(1.8, 170.0 / max(80.0, float(rate))))
    chunks = []
    for ch in text:
        if ch in "。！？!?；;":
            chunks.append(np.zeros(int(0.16 * SR), dtype=np.float32))
            continue
        if ch in "，,、：:":
            chunks.append(np.zeros(int(0.08 * SR), dtype=np.float32))
            continue
        if ch.isspace():
            chunks.append(np.zeros(int(0.05 * SR), dtype=np.float32))
            continue
        parsed = _parse(ch)
        if parsed is None:
            continue
        base, ini, fin, tone = parsed
        y = _syllable(base, ini, fin, tone, voice)
        # speed: simple resample by linear interpolation
        if abs(speed - 1.0) > 1e-3:
            n = max(2, int(len(y) / speed))
            y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y).astype(np.float32)
        chunks.append(y)
        chunks.append(np.zeros(int(0.035 * SR), dtype=np.float32))
    if not chunks:
        raise ValueError("no Chinese characters found")
    out = np.concatenate(chunks).astype(np.float32)
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.85
    # write 16-bit PCM WAV
    import wave
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((np.clip(out, -1, 1) * 32767).astype("<i2").tobytes())
    return out_wav
