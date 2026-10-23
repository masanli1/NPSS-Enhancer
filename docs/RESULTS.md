# Results

All numbers are on 30 held-out VoiceBank-DEMAND pairs, 16 kHz, SI-SDR / PESQ-WB /
STOI.

## v2.1 main modes

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **Music 16k** (multi-stat blend) | **13.77** | 2.487 | 0.9071 |
| **Voice 16k** (multi-stat blend + 40% NMF) | **13.92** | 2.480 | **0.9110** |
| Music 48k (legacy) | 13.02 | 2.421 | 0.9026 |

## v2.0 comparison

| Mode | v2.0 | v2.1 | Delta |
|---|---|---|---|
| Music 16k | 13.39 / 2.514 / 0.9055 | 13.77 / 2.487 / 0.9071 | SI +0.38; PESQ −0.027; STOI +0.0016 |
| Voice 16k | 13.70 / 2.480 / 0.9104 | 13.92 / 2.480 / 0.9110 | SI +0.22; PESQ 0; STOI +0.0006 |

- **New highest SI: 13.92**; PESQ 2.480; STOI 0.9110.
- Music PESQ is slightly lower than v2.0; SI/STOI are higher.
- STOI remains below noisy/classical; this is a real trade-off.

## Classical baselines

| Method | PESQ | STOI |
|---|---|---|
| noisy | 2.056 | 0.9156 |
| Wiener | 2.178 | 0.9168 |
| SpecSub | 2.353 | 0.9168 |
| MMSE-LSA | 2.176 | 0.9168 |
| OM-LSA | 2.202 | 0.9166 |

## Interpretation

- **SI-SDR 13.8/13.9**: traditional top; far from deep (~18–20).
- **PESQ 2.48–2.49**: top of the tested traditional baselines; absolute mid.
- **STOI 0.907–0.911**: below noisy/classical; not an intelligibility win.
- Not SOTA; not a universal win.

## Caveats

- 30 pairs, one dataset, English.
- Traffic probe is synthetic, not a real environment bank.
- No voice/babble probes.
- Multi-stat costs about 6× single.
- PESQ/STOI are implementation-dependent; compare within this repo.

## 48k full-band (v2.2)

No 48k clean reference, so no SI/PESQ. A 12-second music excerpt (48k) shows:

| Band | Change |
|---|---|
| 20–250 Hz | −2.53 dB |
| 250–1k | +0.45 dB |
| 1–4k | +0.37 dB |
| 4–8k | +0.47 dB |
| **8–16k** | **+1.22 dB** (old path: −25.47 dB) |

- RMS matched to input.
- 8k+ is untouched, not denoised; high-frequency noise remains.
- Mono output; stereo not fixed.

## v2.3 stereo / low protection

- 16k unchanged: music 13.77 / 2.487 / 0.9071; voice 13.92 / 2.480 / 0.9110.
- 48k music 12 s test: 20–250 Hz **−1.05 dB** (v2.2: −2.52 dB).
- Stereo side energy: **−0.40 dB**; L/R correlation 0.979 → 0.980.
- 8–16k remains +1.0 to +1.1 dB (untouched high band).
- True stereo costs about 2×.

## v2.4 soft cross-region topology

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| v2.3 music | 13.77 | 2.487 | 0.9071 |
| **v2.4 music** | **13.79** | **2.492** | **0.9085** |
| v2.3 voice | 13.92 | 2.480 | 0.9110 |
| **v2.4 voice** | **13.94** | **2.482** | **0.9116** |

- Small but consistent gain across all three metrics.
- Hard topology weights and temporal median were worse; soft 0.8+0.2·degree is used.

## v2.5 pivot completion

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| v2.4 music | 13.79 | 2.492 | 0.9085 |
| **v2.5 music** | **13.81** | **2.497** | **0.9086** |
| v2.4 voice | 13.94 | 2.482 | 0.9116 |
| **v2.5 voice** | **13.95** | **2.484** | **0.9117** |

- Small but consistent gain; completion is cheap.
- Time-only completion was weaker; time-frequency gap filling is used.

## v2.6 high/low contribution roles

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| v2.5 music | 13.81 | 2.497 | 0.9086 |
| **v2.6 music** | **13.83** | **2.498** | **0.9086** |
| v2.5 voice | 13.95 | 2.484 | 0.9117 |
| **v2.6 voice** | **13.96** | **2.484** | **0.9117** |

- Hard high/low gating was worse; soft weighting traded PESQ for SI.
- Role separation in completion gives a small stable gain.
