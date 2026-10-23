# Algorithm (v2.1)

## Core idea

Classical denoisers estimate a noise PSD and apply a gain. NPSS instead selects
structures that survive perturbation and uses them to gate a classical backend.

### 1. Perturbation and stability

```
x_k = x + a_k * sigma * p
S(f,t) = (1/K) * sum_k 1[ P(x_k)(f,t) == P(x)(f,t) ]
```

`P` is the 2×2 local extremum position on the magnitude STFT:

```
P(M)(f,t) =
  argmax_{i,j in {0,1}} M(f+i, t+j)   if mean2x2 > mean4x4
  argmin_{i,j in {0,1}} M(f+i, t+j)   otherwise
```

Stable positions are "pivots"; unstable positions are noise or fine detail.

### 2. Single-perturbation full pass

```
S = stability(correlated noise, K=10, 3 passes)
y_single = ISTFT(g_specsub * contrast_gate(S, 30, 90, 0.30) * X)
```

- 2 loops of 3 passes; the second loop starts from the first output.
- `gate_floor=0.30` limits maximum suppression to about 10.5 dB.

### 3. Multi-stat Gmin (v2.1)

Use statistically different, structureless probes — **not voice/babble**:

```
white
student_t(df=3)
impulsive (sparse)
AM(4 Hz)
bursty (20–80 ms on)
traffic (low-pass + 60 Hz + rumble)
```

For each family `f`:

```
S_f = stability(f, K=10, 3 passes)
g_f = contrast_gate(S_f, 30, 90, 0.30)
```

Intersection gate:

```
g_min = min_f g_f
y_gmin = ISTFT(g_specsub * g_min * X)
```

`min` keeps only points that are stable under **every** statistically different
probe; family-specific false pivots are removed.

### 4. Blend

```
y_music = 0.5 * y_single + 0.5 * y_gmin
y_voice = 0.6 * y_music + 0.4 * y_nmf
```

- 50/50 keeps single-perturbation detail and avoids Gmin over-suppression.
- Voice adds 40% NMF (60 English clean utterances, 16 kHz only).
- 48 kHz falls back to legacy NPSS.

## Final parameters

```
single : family=correlated, K=10, passes=3, loops=2,
         theta=0.95, support=0.60, gate_floor=0.30,
         n_fft=2048, hop=512, backend=specsub
multi  : 6 families, K=10, 3-pass, 2-loop, gate_floor=0.30
blend  : 0.5 single + 0.5 gmin
voice  : 0.6 blend + 0.4 NMF
```

## Why it works

- Perturbation stability separates structure from noise/fine detail.
- Different **statistics** (heavy tails, sparsity, non-stationarity) produce
  less-correlated stability maps than different noise colours.
- Gmin is a consensus/intersection; it removes family-specific pivots.
- The 50/50 blend restores detail lost by the conservative intersection.

## Failure modes

- Very high SNR: the gate can remove detail; legacy SpecSub may be better.
- Very low SNR: absolute quality remains limited.
- Percussion/low-frequency music: transients and bass can be treated as
  unstable; drums lose impact.
- STOI is not improved over noisy/classical baselines.
- NMF voice mode is English-specific.
- Multi-stat is about 6× the single-perturbation cost.

## v2.0

The previous version used only a single correlated perturbation family. Its
documentation is archived in `docs/history/v2.0/`.

## 48 kHz full-band path (v2.2)

```
x16  = FFT_resample(x, sr -> 16000)
y16  = multi_stat(x16)                  # 0-8k
ylow = FFT_resample(y16, 16000 -> sr)
xh   = highpass(x, 8000 Hz)             # 8k+ untouched
y    = (ylow + xh) * (rms_in / rms_out)
```

- Prevents the old “16k processed then upsampled to 48k” band-limiting.
- 0–8k is denoised; 8k+ is preserved, not denoised.
- Output is still mono; stereo is not fixed.
- Cost is close to 16k (one extra resample + high-pass), not 54×.

## Stereo + low protection (v2.3)

- **Stereo:** extract 2 channels, process L/R independently, write stereo.
  If L and R are identical (mono source), process once and duplicate.
- **48k music low protection:** gate floor is raised to 0.70 below 250 Hz;
  20–250 Hz change improves from about −2.5 dB to −1.05 dB.
- **16k:** low floor stays 0.30 (music/voice), so 16k SI/PESQ is unchanged.
- **48k voice:** low floor 0.30; 8k+ is still untouched.

## Cross-region topology (v2.4)

- Split the time axis into 0.5 s regions (16 frames at 16k/hop 512).
- Per region, take the mean stability in 8 frequency bands → feature vector.
- Build an adjacency graph between neighbouring regions; `degree = cos(feat_i, feat_{i-1}) + cos(feat_i, feat_{i+1})`.
- Soft weight: `w = 0.8 + 0.2 * degree / max_degree`; `S = S * w` before the gate.
- Uses both previous and next regions; the weight is deliberately soft to avoid over-smoothing.
- Temporal median over ±0.5 s was tested and **rejected**: it over-smooths and hurts SI/STOI.

## Pivot completion (v2.5)

After topology weighting, fill pivots that should exist but were not detected:

- Same frequency: if pivots at `t-1` and `t+1` (or `t-2`/`t+2`) and `S > 0.60` → fill.
- Adjacent frequencies: if pivots at `f-1` and `f+1` and `S > 0.70` → fill.
- Set filled `S` to `THETA`; no audio recovery.
- This is inference from pivot relationships, not interpolation/reconstruction.
- Completion is applied before topology weighting; it uses both past and future frames.
