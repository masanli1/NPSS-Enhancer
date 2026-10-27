#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PESQ / STOI helpers for 16 kHz speech evaluation."""
import numpy as np
from pesq import pesq
from pystoi import stoi


def _resample(x, sr, target=16000):
    if sr == target:
        return x.astype(np.float32)
    t = np.arange(int(len(x) * target / sr)) / float(target)
    return np.interp(t, np.arange(len(x)) / float(sr), x).astype(np.float32)


def pesq_stoi(clean, den, sr=16000):
    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    den = np.asarray(den, dtype=np.float32).reshape(-1)
    n = min(len(clean), len(den))
    clean, den = clean[:n], den[:n]
    clean, den = _resample(clean, sr), _resample(den, sr)
    p = float(pesq(16000, clean, den, "wb"))
    s = float(stoi(clean, den, 16000, extended=False))
    return p, s
