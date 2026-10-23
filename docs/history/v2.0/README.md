# NPSS-Enhancer

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Enhancer](https://github.com/masanli1/NPSS-Enhancer)

[中文说明](README.zh-CN.md)

> **No training. No GPU. +2.3 dB SI-SDR over SpecSub.**
> A training-free general denoiser that raises the floor, not the SOTA ceiling.

**Perturbation-stable pivot denoising for speech and music.** Training-free general mode, CPU-only, interpretable, with a GUI and a CLI (optional NMF speech blend).


## Results (30 held-out VoiceBank-DEMAND pairs, 16 kHz)

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **Music 16k** (comp k5, floor 0.3) | **13.39** | **2.514** | 0.9055 |
| **Voice 16k** (comp k5, floor 0.3 + NMF) | **13.70** | 2.480 | **0.9104** |
| Music 48k (legacy NPSS) | 13.02 | 2.421 | 0.9026 |

- **Highest SI-SDR:** voice 13.70; **highest PESQ:** music 2.514.
- **STOI is still below noisy/classical baselines** (0.9055/0.9104 vs 0.9156/0.9168). This is a documented quality/intelligibility trade-off.
- Not SOTA: deep-learning denoisers are roughly 18–20 SI-SDR / PESQ 3+.
- 30 pairs, one dataset, PESQ-WB; numbers are dataset-dependent.

## Features

- **Complementary pivots:** stability maps from the noisy and denoised signals are averaged over 5 loops.
- **Gate floor 0.3:** limits maximum suppression to 10.5 dB, preserving detail/transients.
- **SpecSub backend** (default) for music/general; optional **NMF speech blend** (40%) for voice.
- **Training-free for the general mode**; the NMF voice mode uses a small speech dictionary.
- **Parallel chunk processing**, GUI for audio/video, waveform/spectrogram view, PESQ/STOI evaluation.

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

# CLI (music mode)
python npss_parallel.py samples/demo_noisy.wav samples/demo_denoised.wav --workers 4 --loops 5

# GUI
./run_gui.command
# or
python GUI/npss_gui.py
```

## Modes

| Content | Mode | Notes |
|---|---|---|
| Music / ambient / general | Music | NPSS + SpecSub, 16k or 48k; 16k uses complementary pivots |
| English speech | Voice | Complementary pivots + 40% NMF; 16k only |

- 48k skips complementary pivots (not validated at 48k) and falls back to legacy NPSS.
- The NMF dictionary was trained on 60 English clean utterances; Chinese/music are not validated.

## Algorithm (short)

```
x_k = x + a_k * sigma * n_k
S(f,t) = (1/K) * sum_k 1[P_k(f,t) == P_0(f,t)]
S_comp = mean(S_1, S_2, ..., S_5)
gate = contrast_gate(S_comp, 30, 90, 0.30)
y = ISTFT(g_specsub * gate * STFT(x))
```

- `P_k` is the 2×2 local extremum position.
- `S_1` is computed on the noisy signal; `S_{k+1}` is computed on the previous output.
- Voice mode: `y = 0.6*y_comp + 0.4*y_nmf`.

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
- NMF voice mode is English-specific and 16k-only.
- 5 loops cost roughly 5× NPSS time; parallel chunking helps.
- Only 30 held-out pairs in the reported results; PESQ/STOI are dataset-dependent.
- The general mode is training-free; the voice mode uses a small learned dictionary.

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
├── GUI/                  # Tkinter GUI, engine, common, NMF model
├── npss_parallel.py      # CLI (music mode)
├── eval_official.py      # PESQ/STOI for main modes
├── eval_baselines.py     # PESQ/STOI for classical baselines
├── metrics.py            # PESQ/STOI helpers
├── scripts/make_demo.py  # synthetic demo generator
├── tests/test_smoke.py   # CI smoke test
├── docs/                 # algorithm, results, licensing
├── requirements.txt
├── requirements-metrics.txt
├── LICENSE / LICENSE-ACADEMIC.md / LICENSE-COMMERCIAL.md
├── NOTICE.md / CITATION.cff / CHANGELOG.md
└── run_gui.command
```

## Acknowledgements

Classical baselines and the VoiceBank-DEMAND dataset are cited in [`NOTICE.md`](NOTICE.md). This repository contains original code; third-party packages retain their own licenses.