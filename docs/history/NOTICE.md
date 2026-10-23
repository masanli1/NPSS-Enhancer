# NOTICE

## Code

NPSS-Enhancer
Copyright (c) 2026 `masanli1`.
Dual-licensed: academic/non-commercial under `LICENSE-ACADEMIC.md`; commercial
use requires a paid license under `LICENSE-COMMERCIAL.md`.

## Dataset attribution

The NMF dictionary (`GUI/nmf_dict.npz`) was trained on clean/noisy speech from
the **VoiceBank-DEMAND** dataset:

- Valentini-Botinhao, C., Wang, X., Takaki, S., & Yamagishi, J. (2017).
  *Noisy speech database for training speech enhancement algorithms and TTS
  models.* University of Edinburgh. DOI: 10.7488/ds/2117.
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- https://datashare.ed.ac.uk/handle/10283/2791

No third-party audio from VoiceBank-DEMAND (or any other dataset) is included
in this repository. The NMF dictionary is a derived model; if you redistribute
it, keep this attribution and comply with CC BY 4.0.

## Results

The CSV results in this repository are aggregate/per-file metrics derived from
the above dataset, included for reproducibility with attribution.

## Third-party software

This project depends on (not vendored):

- NumPy — BSD-3-Clause
- imageio-ffmpeg — BSD-2-Clause
- Pillow — HPND
- SciPy — BSD-3-Clause (metrics only)
- pesq — MIT (metrics only)
- pystoi — MIT (metrics only)

Their licenses remain in effect; see the respective packages.

## No warranty

The software and model are provided "as is", without warranty. See the licenses.
