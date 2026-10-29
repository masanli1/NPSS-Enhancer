# Changelog

## 3.1.0 — 2026-10-29

- **Video denoising now accumulates along motion trajectories.** New
  `traj_denoise.py` computes a dense block-matching flow between consecutive
  frames, warps each neighbour along the trajectory onto the current frame, and
  merges with residual weights. Nothing is segmented out: the moving subject is
  accumulated exactly like the static background (this follows the "continuity
  propagates detail" principle).
  Measured on a clip with synthetic noise (sigma=15, 640x480, 9 frames):
  single-frame denoise 33.06 dB -> trajectory accumulation **35.27 dB**
  (+2.2 dB), with no mask halos. Cost: about 0.7 s/frame at 768x512.
- `engine.process_video_av` defaults to the trajectory path and falls back to
  per-frame denoising if it fails; `--fast` halves the radius.
- Fixed: the GUI video branch still passed the removed `spins` argument.

## 3.0.6 — 2026-10-29

- **Adaptive frame count for the video-to-still merge.** More frames are not
  always better: on handheld footage with a moving subject the registration
  error grows faster than the 1/sqrt(N) noise gain. The merge now measures the
  inter-frame motion and picks 16/12/8/4/3 frames (static -> many, moving ->
  few), overridable with `--still-frames`.
  Same clip: fixed 12 frames 34.14 dB, adaptive 4 frames **34.65 dB**,
  fixed 30 frames 33.04 dB.

## 3.0.5 — 2026-10-29

- **Video to still (multi-frame merge).** New `still_merge.py` and
  `--still SECONDS`: decode 12 consecutive frames, register them with a
  tile-based affine fit, merge with residual weights, and blend back to the
  single-frame result wherever the frames disagree (moving subjects), then
  write one image.
  Measured on a clip with synthetic noise (sigma=15): single-frame denoise
  31.53 dB -> merged still **34.14 dB** (+2.6 dB), SSIM 0.893 -> 0.952.
- GUI: selecting "图片 PNG" with a video input now runs the merge instead of
  failing, so the app can turn a clip into one clean photo.

## 3.0.4 — 2026-10-29

- **Automatic phase density for video.** The pipeline now estimates the frame
  count, resolution and a 10-minute budget, then picks the sliding density:
  short clips get all 64 offsets (no seams), long ones step down to 32/16/8/4.
  A 6 s 1440x1920 clip picks 64; a 60 s 1080p clip picks 16; a 5 min 1080p
  clip picks 4. The choice is logged before processing.
- `--fast` still forces 16, `--quality` forces full sliding plus multi-scale
  fusion.

## 3.0.3 — 2026-10-29

- **Full sliding, no fixed block grid.** The default image path now slides the
  8x8 window over *every* integer offset (64 phases) instead of 16, which
  removes the residual block boundaries: the boundary/interior contrast ratio
  drops from 1.29/1.17 to 1.03/0.94 (1.0 = no seam), PSNR 32.29 -> 32.63 dB.
- `--fast` keeps the 16-phase version; `--quality` fuses 4/8/16 with full
  sliding (32.69 dB, ~3x slower).
- Video keeps 16 phases by default (64 phases per frame is too slow for long
  clips); `--quality` enables full sliding plus multi-scale fusion.

## 3.0.2 — 2026-10-29

- **Arbitrary block partitions.** The image filter no longer assumes a fixed
  8x8 grid: `denoise(..., blocks=(4, 8, 16))` fuses several block sizes, each
  one sliding over its own sub-pixel offsets (a 4x4 grid uses all 16 offsets).
  The outputs are averaged, which costs 2-3x time and buys +0.13 to +0.30 dB.
  Measured (same noise, same metrics):
  kodim23 sigma=25 32.29 -> 32.58 dB, kodim01 25.95 -> 26.25 dB,
  sigma=50 28.43 -> 28.56 and 22.95 -> 23.22 dB.
- `--blocks 4,8,16` on the image CLI; the video CLI gained `--quality`
  (4/8/16 fusion) while `--fast` keeps the single 8x8 partition.
- Still local (no non-local grouping), still ~1 dB behind BM3D, still ~20x
  faster than it.

## 3.0.1 — 2026-10-29

- **Image/video speed-up.** The 2-D DCT is now one 64x64 Kronecker matrix
  instead of two einsum contractions (4.7x on the transform), the BLAS thread
  count is pinned to 1 (Accelerate's small-batch threading was pure overhead:
  2.2x end-to-end), and video frames are filtered in parallel fork workers
  (up to 6x on the frame stage with 8 workers).
  End-to-end on the 768x512 / 24-frame test clip: 18.6 s -> 7.4 s.
- Output is bit-identical to 3.0.0 (same PSNR/SSIM); `npss_image.denoise`
  exposes the same signature.

## 3.0.0 — 2026-10-29

- **Image denoising is now part of the app.** `GUI/npss_image.py`: 8x8 block
  DCT, 3-sigma hard threshold (DC protected), 16-phase cycle-spinning.
  Picker accepts PNG/JPG/JPEG/BMP/TIF/TIFF/WebP/PPM/PGM/JP2; new "图片 PNG"
  mode in the GUI.
  Benchmarks (5 Kodak images, grayscale, AWGN): 27.90 dB / SSIM 0.7785 at
  sigma=25 and 24.08 dB / 0.6392 at sigma=50, i.e. +2.2 / +1.4 dB over a
  Gaussian blur and +3.2 / +3.4 dB over a 3x3 median.
- **Video mode denoises the picture too.** New `GUI/av_denoise.py` decodes the
  frames through ffmpeg, filters each frame with NPSS-Image, denoises the
  soundtrack with the existing NPSS audio path, and re-muxes H.264/AAC.
  The GUI has a "视频画面也降噪" checkbox (on by default); turning it off
  reproduces the old audio-replacement behaviour.
  Test clip (768x512, 12 fps): frame PSNR 27.18 -> 30.53 dB, audio SNR
  6.83 -> 9.51 dB.
- `engine.process_video_av()` is the programmatic entry point for the
  combined video pipeline.
- The GUI keeps every previous audio option unchanged.

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
