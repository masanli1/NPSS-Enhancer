#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NMF speech blend used by the voice mode."""
import os
import numpy as np
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
_DICT = None


def _dicts():
    global _DICT
    if _DICT is None:
        d = np.load(os.path.join(HERE, "nmf_dict.npz"))
        _DICT = (d["Ws"], d["Wn"])
    return _DICT


def nmf_mask(V, Ws, Wn, iters=60):
    V = V.astype(np.float64)
    V = V / (float(V.max()) + 1e-12)
    W = np.hstack([Ws, Wn]).astype(np.float64)
    K, T = W.shape[1], V.shape[1]
    H = np.full((K, T), 0.1, dtype=np.float64)
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-9)
        H = np.maximum(H, 1e-9)
    Hs = H[:Ws.shape[1]]
    M = (Ws.astype(np.float64) @ Hs) / (W @ H + 1e-9)
    return np.clip(M, 0.0, 1.0).astype(np.float32)


def enhance(x):
    Ws, Wn = _dicts()
    X = C.stft(x, 512, 128)
    M = nmf_mask(np.abs(X).astype(np.float32), Ws, Wn)
    return C.istft(M * X, len(x), 512, 128)
