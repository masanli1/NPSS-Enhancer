# Results

All numbers are on 30 held-out VoiceBank-DEMAND pairs, 16 kHz, SI-SDR / PESQ-WB /
STOI.

## Main modes

| Mode | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **Music 16k (k5, floor 0.3)** | **13.39** | **2.514** | 0.9055 |
| **Voice 16k (k5, floor 0.3)** | **13.70** | 2.480 | **0.9104** |
| Music 48k (legacy) | 13.02 | 2.421 | 0.9026 |

## Classical baselines

| Method | PESQ | STOI |
|---|---|---|
| noisy | 2.056 | 0.9156 |
| Wiener | 2.178 | 0.9168 |
| SpecSub | 2.353 | 0.9168 |
| MMSE-LSA | 2.176 | 0.9168 |
| OM-LSA | 2.202 | 0.9166 |

## Gate floor trade-off (music, comp k2)

| floor | SI-SDR | PESQ | STOI |
|---|---|---|---|
| 0.1 | 13.25 | 2.429 | 0.9011 |
| 0.3 | 13.28 | 2.508 | 0.9060 |
| 0.5 | 13.11 | 2.466 | 0.9103 |
| 0.8 | 12.34 | 2.368 | 0.9153 |

- **0.3** is quality-first; **0.8** is intelligibility-first.
- The shipped best version uses 0.3.

## Interpretation

- **PESQ 2.51**: top of the tested traditional baselines; absolute mid.
- **STOI 0.905–0.910**: below noisy/classical. This is a real trade-off.
- **SI-SDR 13.4/13.7**: traditional top; far from deep (~18–20).
- Not SOTA; not a universal win.

## Caveats

- 30 pairs, one dataset, one language (English).
- PESQ/STOI are implementation-dependent; compare within this repo.
- No PESQ/STOI on music; only SI-SDR.
- Per-file CSV results are included.
