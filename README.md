# NPSS-Enhancer

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Enhancer](https://github.com/masanli1/NPSS-Enhancer)

[中文说明](README.zh-CN.md)

> **v3.3: voice extraction — F0 / harmonics / formants / syllabic modulation, keep speech and drop the rest.**
>
> **v3.1.0: video denoising accumulates along motion trajectories (+2.2 dB over per-frame, no foreground masking); images keep the full-sliding 3σ filter.**
>
> **Images:** 8x8 block DCT, 3-sigma threshold, 16-phase cycle-spinning — 27.90 / 24.08 dB on Kodak at sigma=25 / 50.
> **Video:** picture and soundtrack are denoised together (`GUI/av_denoise.py`).
>
> **No training. No GPU. +2.7 dB SI-SDR over SpecSub.**
> A training-free general denoiser that raises the floor, not the SOTA ceiling.

**Perturbation-stable pivot denoising for speech, music, and images.** CPU-only, interpretable, GUI + CLI, with an optional NMF speech blend.

## v3.0: images and video

| Input | What is denoised | Entry |
|---|---|---|
| audio | soundtrack | `engine.process_file` |
| video | soundtrack **and** every frame | `engine.process_video_av` / GUI checkbox "视频画面也降噪" |
| image | the picture | `GUI/npss_image.py` (`denoise`), GUI mode "图片 PNG" |

Image quality (5 Kodak images, grayscale, additive white noise, 0-255 scale):

| sigma | noisy | Gaussian | median 3x3 | NPSS-Image |
|---|---|---|---|---|
| 25 | 20.43 | 25.71 | 24.70 | **27.90** |
| 50 | 14.91 | 22.65 | 20.72 | **24.08** |

The image filter is the same idea as the audio one — relative to the estimated
noise floor, DC/low band protected, ensembling over perturbations (here:
16 sub-pixel grid phases) — but the structure decision is made by coefficient
magnitude rather than by pivot stability; the stability variants were measured
and did not beat it on images (see `图像降噪/NPSS-Best/README.md`).

## v3.3: voice extraction

Denoising asks "how stable is this structure". Voice extraction asks "does this
time-frequency cell look like a human voice". Pure numpy, no model, no training:

- **F0:** cepstral candidates + Viterbi, 70–360 Hz, octave-jump penalty.
- **Harmonic comb:** time-varying, with a controlled floor between the teeth.
- **Formants:** per-frame LPC-16 roots (F1/F2/F3).
- **Syllabic modulation:** share of frame-energy envelope in 2–8 Hz — speech has
  syllables, sustained chords do not.
- **Voice core:** harmonic *and* syllabic *and* periodic *and* above the noise
  floor; dilated into speech segments so unvoiced consonants ride along;
  a tonal, non-syllabic clip (BGM only) goes to the floor as a whole.

```
G = G_npss × [ floor + (1-floor) × VAD × harmonic_comb ] ^ strength
```

On 42 fixed-stride VoiceBank-DEMAND pairs: SI-SDR **13.94** (v2.7 13.67),
PESQ 2.345 (2.351), STOI 0.9203 (0.9226). Rejection on 8 s synthetic inputs:
music −32 dB, white noise −43 dB, traffic −10.4 dB; clean speech alone keeps
SI-SDR 22.25 dB / PESQ 3.88.

GUI: 模式 → 人声提取（只留人声）, 参数 → 人声强度 (0–1, default 0.8).
Details and the reproducible script: [`docs/VOICE_SEPARATION.md`](docs/VOICE_SEPARATION.md),
[`scripts/eval_voice_sep.py`](scripts/eval_voice_sep.py).

## Results (30 held-out VoiceBank-DEMAND pairs, 16 kHz)

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **Music 16k** (multi-stat + topology + roles + energy) | **13.92** | **2.505** | **0.9082** |
| **Voice 16k** (multi-stat + topology + roles + energy + 40% NMF) | **14.03** | **2.489** | **0.9116** |
| Music 48k (full-band hybrid) | not measured | — | — |

- **Highest SI-SDR:** voice 14.03; **highest PESQ:** music 2.505.
- vs v2.0: music +0.38 dB SI (PESQ −0.027); voice +0.22 dB SI (PESQ unchanged).
- **STOI is still below noisy/classical baselines** (0.9082/0.9116 vs 0.9156/0.9168). This is a documented quality/intelligibility trade-off.
- Not SOTA: deep-learning denoisers are roughly 18–20 SI-SDR / PESQ 3+.
- 30 pairs, one dataset, PESQ-WB; numbers are dataset-dependent.

## v2.9 quality profile (default `frequency`)

v2.9 keeps the v2.8 default gate and adds a soft four-view modulation:

```text
G = G_v28 ** (1 + 0.5 * (0.5 - I_freq))
```

`frequency` was the only view that raised PESQ/STOI on both speakers in the
VoiceBank four-view tune, so it is the v2.9 default. It buys perceptual quality
at a small SI-SDR cost; `view_profile="off"` reproduces v2.8,
`"energy"`/`"struct"` are the other measured profiles.

| subset | config | SI-SDR | PESQ | STOI |
|---|---|---|---|---|
| p232, 20 | v2.8 base | **13.81** | 2.471 | 0.9396 |
| p232, 20 | **v2.9 freq β0.5** | 13.71 | **2.499** | **0.9435** |
| p257, 30 | v2.8 base | **14.29** | 2.431 | 0.9372 |
| p257, 30 | **v2.9 freq β0.5** | 14.06 | **2.448** | **0.9424** |
| p232, 20 | **均衡（频率+结构）** | 13.82 | 2.491 | 0.9418 |
| p257, 30 | **均衡（频率+结构）** | 14.23 | 2.444 | 0.9402 |

- PESQ +0.028/+0.017, STOI +0.0039/+0.0052;
- SI-SDR −0.10/−0.23;
- if SI-SDR is the priority, use `view_profile="struct"` or `"off"`;
- these are 20/30-file subsets, not the full 824 testset.


## Supported media formats (v2.9.2)

Input and output go through the bundled ffmpeg, so the container is not limited
to WAV/MP4.

| Kind | Input | Output |
|---|---|---|
| Audio | WAV, FLAC, MP3, M4A/M4B, AAC, OGG/Opus, AIFF/AIFC, WMA, MKA, AC3/EAC3, MP2, CAF, AU/SND, APE, WV | WAV, FLAC, MP3, M4A/M4B, AAC, OGG, Opus, AIFF/AIFC, WMA, MKA, AC3/EAC3, MP2, CAF, AU/SND |
| Video | MP4, M4V, MOV, MKV, AVI, WEBM, FLV, TS/MTS/M2TS, MPG/MPEG, 3GP/3G2, OGV, VOB, WMV/ASF, F4V, MXF, DV, RM/RMVB | MP4, M4V, MOV, MKV, AVI, WEBM, FLV, TS/M2TS, MPG/MPEG, 3GP/3G2, OGV, WMV/ASF |

- Audio output codec follows the extension (MP3 → libmp3lame, FLAC → flac,
  M4A → AAC, OGG → Vorbis, Opus → libopus, WMA → wmav2, AIFF → PCM BE).
- Video output tries stream-copy first for compatible containers, then
  re-encodes with the matching codec (H.264/AAC, VP9/Opus, MPEG-4/MP3,
  MPEG-2/MP2, Theora/Vorbis, WMV2/WMA).
- GUI file dialogs list all supported extensions; the CLI accepts any input
  and picks the output codec from the output extension.

```bash
# any input -> FLAC
python npss_parallel.py input.mkv output.flac --fast
# MP3 -> MP4 video output is not allowed; use an audio output extension
python npss_parallel.py input.mp3 output.m4a --proc-sr 16000
```


## Features

- **Multi-stat perturbation bank:** white, Student-t, impulsive, AM, bursty, traffic — *statistics*, not just noise colour; **no voice/babble probes**.
- **Gmin intersection:** a point must be stable under all families; removes family-specific false pivots.
- **50/50 blend** with the single-perturbation result; voice adds 40% NMF.
- **Soft cross-region topology:** 0.5 s regions, 8 bands, adjacent-graph degree; `w=0.8+0.2·degree`.
- **Pivot completion with roles:** high-contribution pivots are backbone anchors; low-contribution pivots are auxiliary and fill only to 0.82. No audio recovery.
- **Energy/SNR weighting:** local SNR → sigmoid; gate × (0.8+0.2e); 16k only.
- **Four-view invariant discovery:** energy / frequency / morphology / structure stability measured across the multi-family perturbation bank (white, Student-t, impulsive, AM, bursty, traffic), not just white-noise levels.
- **Fast mode (v2.7):** K=6, 2-pass; about 2.5× faster; music 13.62 / voice 13.84.
- **Stereo parallel:** L/R processed concurrently; 12 s stereo drops from ~26 s to ~6.9 s.
- **Voice-feature extraction (v3.3):** cepstral Viterbi F0, harmonic comb, LPC formants, syllabic modulation and a voice-core VAD; `G = G_npss × G_voice`.
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
| Music / general | multi-stat Gmin + 50/50 blend | full-band + stereo: 16k gate + 8k+ soft (×0.70) |
| Voice | multi-stat Gmin + blend + 40% NMF | full-band + stereo: 16k gate + 8k+ soft (×0.70) |
| Voice extraction (v3.3) | multi-stat gate × voice gate (F0 / comb / VAD) | same path, analysed at 16 kHz |

- GUI has a **“多统计 Gmin（v2.1）”** checkbox; default on.
- 48k no longer falls back: 16k multi-stat for 0–8k + soft-gained 8k+ high band; stereo L/R; choose 48k for music.
- Old loop control applies only when multi-stat is off; multi-stat is fixed at 3-pass / 2-loop.
- The NMF dictionary was trained on 60 English clean utterances; Chinese/music are not validated.

## Target mode — 定果寻因 (fixed-effect, seek-cause)

The stock pipeline fixes the *cause* (thresholds, floors, passes) and reports
whatever noise reduction comes out: on the eight test pairs below it lands
anywhere between 11.9 and 19.2 dB. Target mode inverts the direction:

```
果 (effect) : the noise floor must drop by exactly T dB (default 15)
因 (cause)  : one scalar dose c,  G = G0 ** (1 + c * (1 - I))
              c is bisected against the *measured* drop (blind, median of the
              quietest 20% frames), not against a parameter guess
I           : four-view protect map — energy / frequency / morphology /
              structure — fused by weighted consensus, deciding where the
              dose may land
```

```bash
# CLI: noise-floor target (blind, no clean reference)
python npss_parallel.py in.wav out.wav --target-nr 15

# single block, with the evaluation JSON
python GUI/target_nr.py in.wav out.wav --target 15 --clean ref.wav --json info.json

# gain target (development/evaluation only; needs clean reference)
#   --metric si_sdr_gain : target the SI-SDR improvement
#   --metric dsnr        : target the SNR improvement
python GUI/target_nr.py in.wav out.wav --target 15 --clean ref.wav \
  --metric si_sdr_gain --json info_gain.json
```

- GUI: **no target controls**. Target mode is development/evaluation only;
  the GUI stays a plain denoiser. Use the CLI below when you need the
  fixed-effect study.
- The controlled variable (`nr_floor`) is **not** a quality metric. Denoising
  effect is still scored the v2.7 way — SI-SDR / PESQ / STOI on clean-noisy
  pairs — via `scripts/eval_protocol_v27.py`, which reproduces
  `eval_official.py`'s methods, layout and CSV columns, plus target rows.
- Stand-in speech run (8 pairs, 16 kHz, not VoiceBank): the v2.7 ordering is
  reproduced (noisy 7.59 < SpecSub 11.02 < Music 12.44 SI-SDR), and at 15 dB
  the target mode scores **12.55 SI-SDR / 0.9153 STOI** vs 12.44 / 0.9160 for
  the stock pipeline — no measurable loss on the v2.7 metric set.
- `hit=False` in the info dict means the target was outside the reachable
  range; the controller reports the miss instead of silently changing the goal.
- The GUI has **no 15 dB button**; target/gain is a development objective,
  not a user-facing control. Gain targets need a clean reference and are
  evaluation only.
- Local16 sample, target +15 dB SI-SDR: `hit=False`, closest **+6.72 dB**
  (dose 2.0, NR 18.27 dB). This is the current traditional direction; the
  controller reports the miss rather than faking the gain.
- At 48 kHz the target is enforced on the 0–8 kHz band; 8k+ stays soft-gained
  for music, so it adds a little noise back to the full-band number.

See `target15_report.md` / `target15_report.png` for the full evidence.

Latest local16 check (8 synthetic pairs, target 15 dB noise-floor):
`12.55 SI-SDR / 1.561 PESQ / 0.9153 STOI`, 8/8 hit; target 15 is
**noise-floor attenuation**, not +15 dB SI-SDR. The report's max ΔSNR is
about +3.94 dB on these pairs; a +15 dB SNR target returns `hit=False`.

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
  high = S >= 0.95 and has a neighbouring pivot -> backbone, fill to 1.0
  low  = 0.65 <= S < 0.95 -> auxiliary, fill to 0.82
  no audio recovery; gate stays continuous

energy (16k only):
  snr = 10*log10(|X|^2 / noise)
  gate *= 0.8 + 0.2*sigmoid((snr-6)/4)

fast mode:
  K=6, 2 passes; stereo L/R in parallel
  music 13.62 / voice 13.84 (quality mode: 13.83 / 13.96)

48k:
  x16 = resample(x, 16000)
  y16 = multi_stat(x16)
  ylow = resample(y16, 48000)
  xh = highpass(x, 8 kHz) * 0.70   # soft high band
  y = (ylow + xh) * (rms_in / rms_out)

stereo: process L/R independently; if L==R, process once and duplicate.
48k music low protection: floor 0.70 below 250 Hz.
```

- `S` uses the 2×2 local extremum position under perturbation.
- `min_f(g_f)` is the intersection: stable under every statistically different probe.
- The 50/50 blend keeps single-perturbation detail and avoids over-suppression.
- **No voice/babble probes**; structured probes would protect their own structure.

See [`NPSS-Enhancer-v2.7.0-完整文档.md`](NPSS-Enhancer-v2.7.0-完整文档.md) for details.

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
- Music PESQ is slightly lower than v2.0 (2.505 vs 2.514); SI/STOI are higher.
- 48k: 8k+ is softly attenuated (×0.70, about −2.15 dB) to reduce hiss; stereo is preserved; low band is still about 1 dB down.
- No 48k clean reference; no 48k SI/PESQ.
- True stereo costs about 2×; L/R now run in parallel, so wall time is close to one channel; identical L/R is processed once.
- Fast mode trades about 0.2 dB SI / 0.01 PESQ for ~2.5× speed.

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
├── NPSS-Enhancer-v2.7.0-完整文档.md
├── requirements.txt
├── requirements-metrics.txt
├── LICENSE / LICENSE-ACADEMIC.md / LICENSE-COMMERCIAL.md
├── NOTICE.md / CITATION.cff / CHANGELOG.md
└── run_gui.command
```

## Acknowledgements

Classical baselines and the VoiceBank-DEMAND dataset are cited in [`NOTICE.md`](NOTICE.md). This repository contains original code; third-party packages retain their own licenses.
