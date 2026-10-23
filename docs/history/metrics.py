#!/usr/bin/env python3
# Copyright (c) 2026 masanli1. All rights reserved.
# Author: masanli1
# Repository: https://github.com/masanli1/NPSS-Enhancer
# License: academic/non-commercial free (LICENSE-ACADEMIC.md);
#          commercial use requires a paid license (LICENSE-COMMERCIAL.md).
# Contact: 2459551532@qq.com
# SPDX-License-Identifier: LicenseRef-NPSS-Academic-1.0
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
