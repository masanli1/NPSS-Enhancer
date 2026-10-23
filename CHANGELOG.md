# Changelog

## 2.2.0 — 2026-10-23

- 48k full-band path: 16k multi-stat + untouched 8k+ high band.
- Fixes the old 16k-to-48k band-limiting (8–16k was −25 dB).
- 48k 12s test: 8–16k +1.22 dB; RMS matched.
- Output is still mono; 8k+ is not denoised.
- 16k results unchanged: music 13.77 / voice 13.92.

## 2.1.0 — 2026-10-23

- Multi-stat perturbation bank: white, Student-t, impulsive, AM, bursty, traffic.
- Per-family gate + Gmin intersection + 50/50 blend.
- Voice: 40% NMF on the blend.
- Music 13.77 / PESQ 2.487 / STOI 0.9071.
- Voice 13.92 / PESQ 2.480 / STOI 0.9110.
- About 6× single-perturbation cost.
- v2.0 docs archived in `docs/history/v2.0/`.

## 2.0.0 — 2026-10-22

- Best configuration: complementary pivots (5 loops) with `gate_floor=0.30`.
- Music 16k: SI-SDR 13.39, PESQ 2.514, STOI 0.9055.
- Voice 16k: SI-SDR 13.70, PESQ 2.480, STOI 0.9104 (40% NMF blend).
- 48k: legacy NPSS (13.02).
- Added GUI/CLI, parallel chunks, waveform/spectrogram view.
- Added PESQ/STOI evaluation and classical baselines.
- Dual license: academic/non-commercial free; commercial paid.

## 1.x — earlier snapshots

- v1.0: pure algorithm (loops=2, gate_floor=0.1).
- v1.1: added PESQ/STOI.
- v1.2: added classical baselines.
- v1.2 results: music 13.25 / voice 13.65.
