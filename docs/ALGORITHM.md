# Algorithm

## Core idea

Classical denoisers estimate a noise PSD and apply a gain. NPSS instead selects
**structures that survive perturbation** and uses them to gate a classical
backend.

### 1. Perturbation

For `K` perturbation levels:

```
x_k = x + a_k * sigma * n_k
```

`n_k` is a noise family (white, correlated, or impulsive); `a_k` is a log-spaced
level; `sigma` is the signal RMS.

### 2. Local extremum position

On the magnitude STFT `M = |STFT(x)|`, for each bin:

```
P(M)(f,t) =
  argmax_{i,j in {0,1}} M(f+i, t+j)   if mean2x2 > mean4x4
  argmin_{i,j in {0,1}} M(f+i, t+j)   otherwise
```

### 3. Stability

```
S(f,t) = (1/K) * sum_k 1[ P_k(f,t) == P_0(f,t) ]
```

Stable positions are "pivots"; unstable positions are noise or fine detail.

### 4. Complementary pivots (best version)

```
S_1 = stability(x)
y_1 = gate(x, S_1)
S_2 = stability(y_1)
...
S_5 = stability(y_4)
S_comp = mean(S_1, ..., S_5)
```

The first pass is computed on the noisy signal; each later pass is computed on
the previous output. Averaging the stability maps over 5 loops is the best
configuration (13.39 music / 13.70 voice).

### 5. Gate

```
gate = contrast_gate(S_comp, 30, 90, 0.30)
y_comp = ISTFT(g_specsub * gate * STFT(x))
```

- `contrast_gate` is a relative percentile gate: map 30th–90th percentiles of
  `S_comp` to [floor, 1].
- `gate_floor=0.30` limits maximum suppression to about 10.5 dB. This preserves
  detail/transients and recovers part of STOI.
- `g_specsub` is the spectral-subtraction gain.

### 6. Voice mode

```
y_voice = 0.6 * y_comp + 0.4 * y_nmf
```

`y_nmf` uses a small speech dictionary (60 clean utterances, English) and a
Wiener-like NMF mask at 512/128. Voice mode is 16 kHz only.

## Final parameters

```
family=correlated
backend=specsub
n_fft=2048, hop=512
passes=3, wb=2, wc=4
theta=0.95, support=0.60
loops=5, gate_floor=0.30
iterations=1, inject=0
```

## Modes

| Content | 16 kHz | 48 kHz |
|---|---|---|
| Music / general | complementary + SpecSub | legacy NPSS |
| Voice | complementary + 40% NMF | legacy NPSS |

## Why it helps

- Pivots are stable under perturbation; noise and fine detail are not.
- The SpecSub backend supplies the spectral body; the pivot gate selects the
  stable structure.
- Complementary maps (noisy + denoised) recover part of the structure that a
  single pass misses.
- `gate_floor=0.30` is the best quality/intelligibility trade-off in the tested
  range; `0.8` improves STOI further but lowers SI/PESQ.

## Failure modes

- Very high SNR: the gate can remove detail; legacy SpecSub may be better.
- Very low SNR: absolute quality remains limited.
- Percussion/low-frequency music: transients and bass can be treated as
  unstable; drums lose impact.
- STOI is not improved over noisy/classical baselines despite SI/PESQ gains.
- NMF voice mode is English-specific.
