# Changelog

## 2.9.2 — 2026-10-28

- Broad media-format support: 20+ audio input extensions and 20+ video
  input extensions through the bundled ffmpeg.
- Audio output codecs follow the output extension: WAV/FLAC/MP3/M4A/AAC/
  OGG/Opus/AIFF/WMA/MKA/AC3/EAC3/MP2/CAF/AU.
- Video output: MP4/M4V/MOV/MKV/AVI/WEBM/FLV/TS/M2TS/MPG/MPEG/3GP/3G2/OGV/
  WMV/ASF; stream-copy when compatible, otherwise codec-matched re-encode
  (H.264/AAC, VP9/Opus, MPEG-4/MP3, MPEG-2/MP2, Theora/Vorbis, WMV2/WMA).
- GUI file dialogs list all supported input/output types.
- CLI now accepts any input container and selects output codec by extension.
- No algorithm change: default profile and quality numbers stay v2.9.1.

## 2.9.1 — 2026-10-26

- New GUI/CLI profile **均衡（频率+结构）** / `blend`: `0.5*(freq + struct)`.
  p232 20: 13.82/2.491/0.9418; p257 30: 14.23/2.444/0.9402.
  It keeps most of freq's PESQ/STOI and recovers part of struct's SI-SDR.
- STOI band-envelope guard tested: +0.0004/+0.0009 STOI but PESQ -0.01 and
  SI down; not enabled by default.
- Default remains **频率优先 (freq)**; all profiles are optional.
- Target mode still development-only; GUI has no target controls.

## 2.9.0 — 2026-10-26

- v2.9 default **frequency quality profile**: soft four-view modulation
  `G ** (1 + 0.5*(0.5 - I_freq))`, selected because frequency raised
  PESQ/STOI on both p232 and p257 in the concrete VoiceBank tune.
- Profiles: `freq` (default), `energy`, `struct`, `off` (v2.8 bit-for-bit).
- Trade-off: p232 13.71/2.499/0.9435 vs base 13.81/2.471/0.9396;
  p257 14.06/2.448/0.9424 vs base 14.29/2.431/0.9372. PESQ/STOI up, SI down.
- No hard target and no target controls in the GUI; target remains a
  development diagnostic only.
- GUI **四视角 profile** dropdown (freq default / energy / struct / off);
  engine and CLI thread the chosen profile to `multi_stat.denoise_best`.
- Evidence: `four_view_tune.csv`, `four_view_tune_p257.csv`.

## 2.8.4 — 2026-10-26

- VoiceBank-DEMAND testset (824 pairs) downloaded, resampled to 16 kHz, and
  used for a concrete four-view ablation/tuning (`scripts/tune_four_views.py`).
- p232 (20) vs p257 (30) results show no single fixed view wins everywhere:
  structure is the consistent SI view; frequency raises PESQ/STOI but lowers
  SI; morphology hurts; all-view fusion is neutral.
- Default v2.7 path unchanged; no hard target added to the GUI.
- Evidence CSV: `four_view_tune.csv`, `four_view_tune_p257.csv`.

## 2.8.3 — 2026-10-26

- Removed all target controls from the GUI (no 15 dB button/spinbox):
  target/gain is a development and evaluation objective, not a user feature.
- GUI stays a plain denoiser; CLI `target_nr.py` / `--target-nr` remain for
  the fixed-effect study.
- Default path unchanged.
- Four-view invariant discovery, gain-target metric and evidence stay in code.

## 2.8.2 — 2026-10-26

- Gain-target evaluation: `--metric si_sdr_gain` / `dsnr` / `snr_out` in
  `GUI/target_nr.py` (requires a clean reference).
- GUI label clarified: **目标噪声底降幅 dB**; 15 dB button is noise-floor,
  not +15 dB SI-SDR/SNR gain.
- Local16 sample target +15 dB SI-SDR: `hit=False`, closest +6.72 dB,
  NR 18.27 dB — reports the miss instead of faking it.
- No change to the blind target path or the default denoiser.

## 2.8.1 — 2026-10-26

- Four-view `discover` now measures invariance across the multi-family
  perturbation bank (white / Student-t / impulsive / AM / bursty / traffic),
  not only white-noise levels.
- GUI: one-click **15 dB** target preset; target spinbox remains 0=off.
- Local16 target 15 check: 8/8 hit, 12.55 SI-SDR / 1.561 PESQ / 0.9153 STOI;
  no regression from the discovery change.
- Protect-map gamma 1.5 and stronger/softer fusions tested: no gain; reverted.
- 15 dB is noise-floor attenuation, not +15 dB SI-SDR / SNR.

## 2.8.0 — 2026-10-26

- 定果寻因 (target mode): fix the noise-floor drop at T dB and solve the dose
  by bisection against the measured drop, instead of fixing thresholds.
- 四视角不变特征 (GUI/four_views.py): energy / frequency / morphology /
  structure protect maps, weights measured per file (stability × AUC).
- `Loads` refactor in multi_stat: gains exposed so re-gating costs one ISTFT
  instead of the full 180-STFT cascade.
- 8-pair evaluation: target 15.00 ± 0.01 dB; SI-SDR +0.31 dB over uniform dose.
- CLI `--target-nr`, GUI spinbox, info dict with hit/miss, dose, evidence.
- No change to the default path: target off reproduces v2.7.0 bit-for-bit.

## 2.7.0 — 2026-10-25

- Energy/SNR weighting: 16k quality 13.92 / 14.03.
- Fast mode: K=6, 2-pass; music 13.62 / voice 13.84; 25 s stereo fast + parallel 9.7 s.
- Long-file (>20 s) chunking fixed (`fast` in args tuple).
- 48k high band ×0.70 soft gain (about −2.15 dB) to reduce hiss.
- Stereo L/R parallel.

## 2.6.1 — 2026-10-24

- High-frequency soft gain: 8k+ ×0.70.
- 8–16k: +1.05 dB → −2.15 dB; 4–8k unchanged.
- Reduces hiss; 16k metrics unchanged.

## 2.6.0 — 2026-10-24

- High/low contribution roles in pivot completion.
- Backbone anchors fill to 1.0; auxiliary fills to 0.82 only with anchor support.
- 16k: music 13.83 / 2.498 / 0.9086; voice 13.96 / 2.484 / 0.9117.
- Hard/soft high-low gating rejected; gate remains continuous.

## 2.5.0 — 2026-10-24

- Pivot completion from time/frequency neighbours (no audio recovery).
- 16k: music 13.81 / 2.497 / 0.9086; voice 13.95 / 2.484 / 0.9117.
- Small but consistent gain over v2.4.
- Time-only completion weaker; time-frequency gaps used.

## 2.4.0 — 2026-10-24

- Soft cross-region topology: 0.5 s regions, 8 bands, adjacent-graph degree.
- 16k: music 13.79 / 2.492 / 0.9085; voice 13.94 / 2.482 / 0.9116.
- Small but consistent gain over v2.3.
- Hard topology and temporal median rejected.

## 2.3.0 — 2026-10-24

- Stereo output: L/R processed independently; identical L/R processed once.
- 48k music low protection (floor 0.70 below 250 Hz).
- 16k unchanged: music 13.77 / voice 13.92.
- 48k 20–250 Hz: −2.52 → −1.05 dB.
- Stereo side energy −0.40 dB; width preserved.
- 8k+ still untouched; true stereo ~2× cost.

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
