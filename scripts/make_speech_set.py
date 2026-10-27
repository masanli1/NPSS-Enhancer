#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a local 16 kHz speech set with the *same* pair structure as v2.7.

v2.7 was evaluated on 30 VoiceBank-DEMAND pairs (clean/noisy, 16 kHz).  That
corpus is not in this repository, so this script builds a stand-in with the same
shape -- real speech, synthetic additive noise, identical file layout -- so the
v2.7 protocol can be re-run end to end:

    data/local16/clean/uttXX.wav
    data/local16/noisy/uttXX.wav

Speech comes from the macOS `say` voices (real formant speech, not a tone).
Absolute PESQ/SI-SDR will NOT match the VoiceBank numbers -- only the deltas
between methods are meaningful, exactly as with any corpus swap.

    python scripts/make_speech_set.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "GUI"))

import common as C
import engine

SR = 16000
OUT = os.path.join(ROOT, "data", "local16")

VOICES = ["Samantha", "Daniel"]
SNRS = [0.0, 5.0, 10.0, 15.0]
TEXTS = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
]


def synth(voice, text, path):
    """say -> 16 kHz mono 16-bit wav (via ffmpeg, so the format is exact)."""
    ff = engine.find_ffmpeg()
    tmp = path + ".aiff"
    subprocess.run(["say", "-v", voice, "-o", tmp, text], check=True,
                   capture_output=True)
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", tmp, "-ar", str(SR), "-ac", "1",
                    "-c:a", "pcm_s16le", path], check=True, capture_output=True)
    os.remove(tmp)


def trim(x, thr_db=-45.0, pad=0.15):
    """Trim leading/trailing silence, keep a little air, cap at 6 s."""
    e = np.abs(x)
    thr = np.max(e) * 10 ** (thr_db / 20.0)
    idx = np.where(e > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * SR))
    b = min(len(x), idx[-1] + int(pad * SR))
    y = x[a:b]
    return y[: int(6.0 * SR)]


def noise(kind, n, rng):
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":
        w = rng.standard_normal(n)
        W = np.fft.rfft(w)
        f = np.arange(len(W), dtype=np.float64)
        f[0] = 1.0
        W /= np.sqrt(f / f[1])
        y = np.fft.irfft(W, n=n)
        return y / (np.std(y) + 1e-12)
    if kind == "traffic":
        w = rng.standard_normal(n)
        y = np.convolve(w, np.ones(120) / 120.0, "same")
        return y / (np.std(y) + 1e-12)
    raise ValueError(kind)


def main():
    os.makedirs(os.path.join(OUT, "clean"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "noisy"), exist_ok=True)
    k = 0
    for vi, voice in enumerate(VOICES):
        with tempfile.TemporaryDirectory() as td:
            parts = []
            for ti, text in enumerate(TEXTS):
                p = os.path.join(td, "p%d.wav" % ti)
                synth(voice, text, p)
                x, _ = C.read_wav(p)
                parts.append(trim(x))
                parts.append(np.zeros(int(0.12 * SR), np.float32))  # pause
            clean = np.concatenate(parts)
            clean = clean / (np.max(np.abs(clean)) + 1e-9) * 0.70
            for snr in SNRS:
                rng = np.random.default_rng(1000 + k)
                kind = ["white", "pink", "traffic"][k % 3]
                n = noise(kind, len(clean), rng)
                n *= np.sqrt(np.mean(clean ** 2)) / (
                    np.sqrt(np.mean(n ** 2)) * 10 ** (snr / 20.0))
                noisy = np.clip(clean + n, -1.0, 1.0).astype(np.float32)
                tag = "u%02d_%s_%s_%+03.0f" % (k, voice.lower(), kind, snr)
                C.write_wav(os.path.join(OUT, "clean", tag + ".wav"), clean, SR)
                C.write_wav(os.path.join(OUT, "noisy", tag + ".wav"), noisy, SR)
                print("wrote", tag)
                k += 1


if __name__ == "__main__":
    main()
