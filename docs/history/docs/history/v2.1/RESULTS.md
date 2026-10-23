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
