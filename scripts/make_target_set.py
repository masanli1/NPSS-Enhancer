#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a small, deterministic target-15 dB evaluation set.

Three signal morphologies (harmonic voice, music, formant vowels) crossed with
three noise colours (white, pink, traffic) at three input SNRs.  Every file has
explicit pauses so the blind noise-floor estimator has something to measure --
without pauses "15 dB noise reduction" is not even defined on a blind recording.

    python scripts/make_target_set.py            # writes samples/target15/
"""

import os
import wave

import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "samples", "target15")


def write_wav(path, x):
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


def _pause_env(t, rate=0.23, thr=-0.30):
    """Gating envelope: speech-like on/off with real silences."""
    return (np.sin(2 * np.pi * rate * t) > thr).astype(np.float64)


def _norm(x, peak=0.7):
    return x / (np.max(np.abs(x)) + 1e-9) * peak


# ---------------------------------------------------------------- signals
def sig_harmonic(dur=8.0):
    t = np.arange(int(SR * dur)) / SR
    f0 = 115.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for h in range(1, 12):
        sig += (0.9 ** h) * np.sin(h * phase)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 1.8 * t) ** 2
    env *= _pause_env(t)
    return _norm(0.18 * sig * env)


def sig_music(dur=8.0):
    t = np.arange(int(SR * dur)) / SR
    out = np.zeros_like(t)
    chords = [[261.63, 329.63, 392.00],   # C
              [392.00, 493.88, 587.33],   # G
              [440.00, 523.25, 659.25],   # Am
              [349.23, 440.00, 523.25]]   # F
    seg = dur / len(chords)
    for i, notes in enumerate(chords):
        t0 = i * seg
        m = (t >= t0) & (t < t0 + seg - 0.35)          # 0.35 s gap = pause
        tt = t[m] - t0
        env = np.exp(-tt / 1.4) * (1.0 - np.exp(-tt / 0.02))
        for f in notes:
            for h in range(1, 4):
                out[m] += 0.22 * env * np.sin(2 * np.pi * f * h * tt) / h
        out[m] += 0.30 * env * np.sin(2 * np.pi * 65.41 * tt)      # bass
        # percussive click on the chord onset (a transient to protect)
        cl = np.exp(-tt / 0.012)
        out[m] += 0.18 * cl * np.sin(2 * np.pi * 2500.0 * tt)
    return _norm(out)


FORMANTS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
            (660, 1720, 2410), (730, 1090, 2440), (270, 2290, 3010)]


def _vowel(f0, formants, n, rng):
    """Impulse train shaped by a static formant envelope (FFT domain)."""
    exc = np.zeros(n, dtype=np.float64)
    per = max(2, int(round(SR / f0)))
    exc[::per] = 1.0
    exc += 0.02 * rng.standard_normal(n)
    F = np.fft.rfft(exc)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    H = np.zeros_like(f)
    for k, (fc, bw) in enumerate(zip(formants, (90.0, 110.0, 140.0))):
        H += (0.9 ** k) * np.exp(-0.5 * ((f - fc) / bw) ** 2)
    H += 0.02
    y = np.fft.irfft(F * H, n=n)
    return y / (np.std(y) + 1e-9)


def sig_formant(dur=8.0):
    rng = np.random.default_rng(3)
    n_total = int(SR * dur)
    out = np.zeros(n_total)
    seg = n_total // len(FORMANTS)
    for i, fm in enumerate(FORMANTS):
        n = seg
        y = _vowel(118.0 + 4.0 * i, fm, n, rng)
        tt = np.arange(n) / SR
        env = (0.5 + 0.5 * np.sin(2 * np.pi * 2.1 * tt) ** 2)
        env *= _pause_env(tt + i * seg / SR, rate=1.7, thr=-0.20)
        out[i * seg:(i + 1) * seg] += y * env
    return _norm(out)


# ---------------------------------------------------------------- noises
def noise_white(n, rng):
    return rng.standard_normal(n)


def noise_pink(n, rng):
    w = rng.standard_normal(n)
    Wf = np.fft.rfft(w)
    f = np.arange(len(Wf), dtype=np.float64)
    f[0] = 1.0
    Wf /= np.sqrt(f / f[1])
    y = np.fft.irfft(Wf, n=n)
    return y / (np.std(y) + 1e-12)


def noise_traffic(n, rng):
    w = rng.standard_normal(n)
    y = np.convolve(w, np.ones(120) / 120.0, "same")
    t = np.arange(n) / SR
    hum = 0.35 * np.sin(2 * np.pi * 60.0 * t) + 0.15 * np.sin(2 * np.pi * 120.0 * t)
    y = y / (np.std(y) + 1e-12) + hum / (np.std(hum) + 1e-12)
    return y / (np.std(y) + 1e-12)


SIGNALS = {"harmonic": sig_harmonic, "music": sig_music, "formant": sig_formant}
NOISES = {"white": noise_white, "pink": noise_pink, "traffic": noise_traffic}

CASES = [
    ("harmonic", "white", 5.0),
    ("harmonic", "white", 0.0),
    ("harmonic", "pink", 5.0),
    ("harmonic", "traffic", 10.0),
    ("music", "white", 5.0),
    ("music", "pink", 0.0),
    ("formant", "white", 5.0),
    ("formant", "traffic", 5.0),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for si, (sig_name, noise_name, snr_db) in enumerate(CASES):
        clean = SIGNALS[sig_name](8.0)
        rng = np.random.default_rng(100 + si)
        noise = NOISES[noise_name](len(clean), rng)
        noise *= np.sqrt(np.mean(clean ** 2)) / (
            np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20.0))
        noisy = np.clip(clean + noise, -1.0, 1.0)
        tag = "%s_%s_%+03.0f" % (sig_name, noise_name, snr_db)
        write_wav(os.path.join(OUT, tag + "_clean.wav"), clean)
        write_wav(os.path.join(OUT, tag + "_noisy.wav"), noisy)
        print("wrote", tag)


if __name__ == "__main__":
    main()
