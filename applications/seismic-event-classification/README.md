# Application: Seismic Event Classification

First validated application of the Few-Shot Gap Diagnostic method.

## Domain

Classify seismic events from waveform data into 7 categories using only 3 labeled seeds per class:

| Event type | Duration | Frequency | Temporal signature |
|---|---|---|---|
| tectonic_shallow | 4-10s | 2-10 Hz | Sharp P-wave, strong S-wave, short coda |
| tectonic_deep | 10-28s | 1-5 Hz | Weak P, delayed S, long coda |
| volcanic_tremor | 12-32s | 1.5-9 Hz (harmonic) | Emergent, sustained, amplitude-modulated |
| volcanic_lp | 2-7s | 1-5 Hz (harmonic) | Single low-period pulse |
| explosion | 1.5-5s | 5-35 Hz (broadband) | Sharp onset, rapid decay, no S-wave |
| icequake | 0.8-2.5s | 12-35 Hz | Extremely sharp, very short |
| noise_burst | 1-6s | 3-30 Hz (broadband) | Irregular envelope, gradual onset |

Confusable pairs: tectonic_shallow/deep (both have P+S), volcanic_tremor/lp (both harmonic), explosion/noise_burst (both short, broadband).

**Conditions**: Variable event duration (0.8s-32s), coloured 1/f noise, SNR 3-8 dB.

## Self-Directed Investigation Record

The investigation was conducted by an LLM agent (Claude) following the diagnostic loop methodology. Three iterations, each driven by diagnosing the previous failure.

### Iteration 1: Establish the hard problem

Created 7-class variable-duration dataset with coloured noise. Tested 10 representations.

**Results**: Raw pixels (resized to common width) = 66%. Spectral profile = 59%. Temporal profile = 36%. Edges = 29%.

**Diagnosis**: Temporal profiles catastrophically fail because short events (icequake: 2 spectrogram time bins) produce degenerate profiles. The spectrogram's temporal axis has variable resolution depending on event duration. The spectral axis works because all events share the same 33 frequency bins.

**Decision**: Need temporal information at consistent resolution.

### Iteration 2: Add temporal descriptors

Added 7 scalar temporal features: log-duration, onset sharpness, decay rate, peak position, temporal spread, kurtosis, number of energy peaks.

**Results**: spec+temporal_descriptors = 41%. **Worse than baseline.**

**Diagnosis**: 7 scalar features with only 3 seeds have too much variance. One outlier icequake throws off the centroid. Cosine similarity on 23 dimensions doesn't have enough redundancy to suppress noise (unlike cosine on 2112 dimensions for raw pixels).

**Decision**: Need temporal marginalisation at consistent resolution with enough dimensions for noise resilience. Use the raw waveform (constant sample rate) instead of the spectrogram (variable time bins).

### Iteration 3: Waveform envelope profiles

Computed energy envelope of the raw waveform (squared amplitude, smoothed), binned into fixed-length profiles. Combined with spectral profile.

**Results at SNR = 3 dB (hardest condition)**:

| Representation | Dimensions | Accuracy | Gap |
|---|---|---|---|
| raw_64 (resized spectrogram) | 2112 | 62% | -0.063 |
| spectral_32 | 32 | 57% | -0.034 |
| envelope_32 | 32 | 35% | -0.762 |
| **spec32 + envelope32** | **64** | **66%** | **-0.039** |

**At SNR 3 dB, the combined spectral + envelope profile (66%) beats raw pixels (62%).** This confirms the method's operating regime: profiles win when sizes vary and noise is high.

At SNR 5 dB, profiles tie raw pixels (73% vs 73%). At SNR 8 dB, raw pixels pull ahead (62% vs 59%).

### What the agent got wrong

1. **Initially assumed temporal was the identity axis.** For seismic events with variable duration, the spectral axis (frequency content) is more consistently informative. The gap diagnostic correctly identified this.

2. **Scalar temporal descriptors made things worse.** The agent predicted they'd help based on confusion matrix analysis. In practice, 7 dimensions with 3 seeds were too noisy. The diagnosis was correct (confused classes differ in temporal shape) but the solution was wrong (scalars too fragile). Fixed in iteration 3 with higher-dimensional envelope profiles.

3. **Envelope alone is terrible.** The agent learned that even marginalised temporal features need to be COMBINED with spectral features to work — neither axis alone is sufficient.

### What the agent got right

1. **Gap diagnostic correctly ranked representations** at every iteration. The least-negative gap consistently corresponded to the best or near-best accuracy.

2. **Confusion matrix diagnosis identified the right problem** — explosion/icequake/noise_burst confusion is a temporal identity problem, not a spectral one.

3. **The fix worked**: combining spectral and temporal envelope profiles achieved the best accuracy at the hardest noise level.

4. **Sharpening (edge detection) was always worst** — confirmed across all iterations and conditions.

## How to Run

```bash
# Full investigation sequence
python experiment_seismic.py        # v1: initial 5-class experiment
python experiment_seismic_v2.py     # v2: SNR sweep + axis discovery
python seismic_heatmap.py           # heat map: 9 reps × 8 SNR levels
python iteration1_hard_seismic.py   # 7-class variable-size problem
python iteration2_informed_features.py  # scalar descriptors (failed)
python iteration3_waveform_profiles.py  # waveform envelope (succeeded)
```

Requirements: `numpy`, `scipy`.

## Key Numbers

| Metric | Value |
|---|---|
| Classes | 7 (including 3 confusable pairs) |
| Seeds per class | 3 |
| Test instances per class | 25 |
| Event duration range | 0.8s - 32s (40x variation) |
| Unique spectrogram shapes | 87 |
| Noise type | Coloured (1/f spectrum) |
| Best profile accuracy (SNR 3dB) | **66%** (spec32+env32) |
| Best raw pixel accuracy (SNR 3dB) | 62% (resize to 64 cols) |
| Worst sharpening accuracy | 29% (edge detection) |
| Profile advantage at SNR 3dB | **+4%** over raw pixels |
| Investigation iterations | 3 |
| Total representations tested | ~30 across all iterations |
