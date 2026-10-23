# NPSS-Enhancer

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Enhancer](https://github.com/masanli1/NPSS-Enhancer)

[中文说明](README.zh-CN.md) · [History](docs/history/)

> **No training. No GPU. +2.7 dB SI-SDR over SpecSub.**
> A training-free general denoiser that raises the floor, not the SOTA ceiling.

**Perturbation-stable pivot denoising for speech and music.** CPU-only, interpretable, GUI + CLI, with an optional NMF speech blend.

## Results (30 held-out VoiceBank-DEMAND pairs, 16 kHz)

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **Music 16k** (multi-stat + topology + pivot completion) | **13.81** | **2.497** | **0.9086** |
| **Voice 16k** (multi-stat + topology + completion + 40% NMF) | **13.95** | **2.484** | **0.9117** |
| Music 48k (full-band hybrid) | not measured | — | — |

- **Highest SI-SDR:** voice 13.95; **highest PESQ:** music 2.497.
- vs v2.0: music +0.38 dB SI (PESQ −0.027); voice +0.22 dB SI (PESQ unchanged).
- **STOI is still below noisy/classical baselines** (0.9086/0.9117 vs 0.9156/0.9168). This is a documented quality/intelligibility trade-off.
- Not SOTA: deep-learning denoisers are roughly 18–20 SI-SDR / PESQ 3+.
- 30 pairs, one dataset, PESQ-WB; numbers are dataset-dependent.

## Features

- **Multi-stat perturbation bank:** white, Student-t, impulsive, AM, bursty, traffic — *statistics*, not just noise colour; **no voice/babble probes**.
- **Gmin intersection:** a point must be stable under all families; removes family-specific false pivots.
- **50/50 blend** with the single-perturbation result; voice adds 40% NMF.
- **Soft cross-region topology:** 0.5 s regions, 8 bands, adjacent-graph degree; `w=0.8+0.2·degree`.
- **Pivot completion:** fill small time/frequency gaps from neighbouring pivots; no audio recovery.
- **Training-free general mode**; optional small NMF dictionary for voice.
- **GUI + CLI**, parallel chunks, waveform/spectrogram, PESQ/STOI evaluation.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: PESQ/STOI
pip install -r requirements-metrics.txt
```

`tkinter` is required for the GUI and is included with most Python installers (not a pip package).

## Quick start

```bash
# synthetic demo (no third-party audio)
python scripts/make_demo.py

# CLI (multi-stat default)
python npss_parallel.py samples/demo_noisy.wav samples/demo_denoised.wav --workers 4

# old single-perturbation mode
python npss_parallel.py samples/demo_noisy.wav samples/demo_old.wav --single --workers 4

# GUI
./run_gui.command
# or
python GUI/npss_gui.py
```

## Modes

| Content | 16k | 48k |
|---|---|---|
| Music / general | multi-stat Gmin + 50/50 blend | full-band + stereo: 16k gate + 8k+ preserved |
| Voice | multi-stat Gmin + blend + 40% NMF | full-band + stereo: 16k gate + 8k+ preserved |

- GUI has a **“多统计 Gmin（v2.1）”** checkbox; default on.
- 48k no longer falls back: 16k multi-stat for 0–8k + untouched 8k+ high band; stereo L/R; choose 48k for music.
- Old loop control applies only when multi-stat is off; multi-stat is fixed at 3-pass / 2-loop.
- The NMF dictionary was trained on 60 English clean utterances; Chinese/music are not validated.

## Algorithm (short)

```
single:
  S = stability(correlated noise, K=10, 3 passes, 2 loops)
  y_single = ISTFT(g_specsub * contrast_gate(S, 30, 90, 0.30) * X)

multi-stat:
  bank = {white, student_t(3), impulsive, AM(4 Hz), bursty, traffic}
  for f in bank:
      S_f = stability(f, K=10, 3 passes)
      g_f = contrast_gate(S_f, 30, 90, 0.30)
  y_gmin = ISTFT(g_specsub * min_f(g_f) * X)

music 16k: y = 0.5*y_single + 0.5*y_gmin
voice 16k: y = 0.6*(0.5*y_single + 0.5*y_gmin) + 0.4*y_nmf

topology:
  regions = 0.5 s; features = 8-band mean stability
  graph = adjacent regions; w = 0.8 + 0.2*degree/max
  S = S * w

completion:
  if pivots at t-1/t+1 and S > 0.60 -> fill
  if pivots at f-1/f+1 and S > 0.70 -> fill
  no audio recovery

48k:
  x16 = resample(x, 16000)
  y16 = multi_stat(x16)
  ylow = resample(y16, 48000)
  xh = highpass(x, 8 kHz)          # untouched
  y = (ylow + xh) * (rms_in / rms_out)

stereo: process L/R independently; if L==R, process once and duplicate.
48k music low protection: floor 0.70 below 250 Hz.
```

- `S` uses the 2×2 local extremum position under perturbation.
- `min_f(g_f)` is the intersection: stable under every statistically different probe.
- The 50/50 blend keeps single-perturbation detail and avoids over-suppression.
- **No voice/babble probes**; structured probes would protect their own structure.

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for details.

## Evaluation

```bash
python eval_official.py --clean data/val16/clean --noisy data/val16/noisy
python eval_baselines.py --clean data/val16/clean --noisy data/val16/noisy
```

- `data/val16/` is not included; bring your own 16 kHz clean/noisy pairs.
- PESQ uses `wb` at 16 kHz; STOI uses the 16 kHz standard.
- Environment note: the bundled DSH Python may reject unsigned SciPy extensions; use a normal venv/conda environment for metrics.

## Limitations

- Traditional, single-channel; far from deep-learning SOTA.
- STOI is not improved over noisy/classical baselines.
- Multi-stat costs about **6×** the single-perturbation mode; long files are slow.
- NMF voice mode is English-specific and 16k-only.
- 30 held-out pairs only; traffic probe is synthetic, not a real environment bank.
- Music PESQ is slightly lower than v2.0 (2.497 vs 2.514); SI/STOI are higher.
- 48k: 8k+ is untouched (not denoised); stereo is preserved by processing L/R separately; low band is still about 1 dB down.
- No 48k clean reference; no 48k SI/PESQ.
- True stereo costs about 2×; identical L/R is detected and processed once.

## License (dual)

This project is **dual-licensed**:

- **Academic / non-commercial:** free under [`LICENSE-ACADEMIC.md`](LICENSE-ACADEMIC.md).
- **Commercial:** paid license; see [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md) and contact `2459551532@qq.com`.

The NMF dictionary (`GUI/nmf_dict.npz`) is derived from VoiceBank-DEMAND (CC BY 4.0); see [`NOTICE.md`](NOTICE.md). Commercial users must comply with CC BY 4.0 attribution for the model in addition to the commercial code license.

Not legal advice; consult a lawyer before commercial distribution.

## Data and model attribution

- VoiceBank-DEMAND: Valentini-Botinhao et al., University of Edinburgh, **CC BY 4.0**, DOI `10.7488/ds/2117`.
- No third-party audio is included in this repository.
- The NMF dictionary is a derived model; attribution is required under CC BY 4.0.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Layout

```
NPSS-Enhancer/
├── GUI/                  # GUI, engine, common, multi_stat, NMF model
├── npss_parallel.py      # CLI (multi-stat default)
├── eval_official.py      # PESQ/STOI for main modes
├── eval_baselines.py     # PESQ/STOI for classical baselines
├── metrics.py            # PESQ/STOI helpers
├── scripts/make_demo.py  # synthetic demo generator
├── tests/test_smoke.py   # CI smoke test
├── docs/                 # algorithm, results, licensing, history/v2.0
├── requirements.txt
├── requirements-metrics.txt
├── LICENSE / LICENSE-ACADEMIC.md / LICENSE-COMMERCIAL.md
├── NOTICE.md / CITATION.cff / CHANGELOG.md
└── run_gui.command
```

## Acknowledgements

Classical baselines and the VoiceBank-DEMAND dataset are cited in [`NOTICE.md`](NOTICE.md). This repository contains original code; third-party packages retain their own licenses.
