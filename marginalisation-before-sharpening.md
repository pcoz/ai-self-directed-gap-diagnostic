# Marginalisation Before Sharpening: A Representation Strategy for Low-Data Identity Matching

## Status
Hypothesis with experimental support in two domains (image matching, seismology).

---

## 1. The Problem

In many pattern recognition tasks the goal is to determine whether two observations are instances of **the same thing** — the same word, the same compound, the same species, the same defect. This is the identity matching problem.

Standard approaches build an intermediate representation (features, embeddings, structural descriptions) and compare in that space. This works well with abundant labeled data. It fails in the **low-data, high-noise regime**: 2-10 labeled examples, noisy observations, no pretrained model available.

The failure mode is specific and diagnosable: the intermediate representation **amplifies noise** instead of suppressing it, causing same-class instances to appear more different in feature space than they do in the raw observation.

---

## 2. The Core Observation

### 2.1 Two classes of operation

Any transformation of a raw observation is either:

- **Sharpening**: increases the resolution of fine structure. Examples: edge detection, peak detection, deconvolution, high-pass filtering. Sharpening operations amplify both signal and noise at fine scales.

- **Averaging (marginalisation)**: suppresses fine structure by aggregating over one or more dimensions. Examples: histograms, projection profiles, pooling, binning, low-pass filtering. Averaging operations suppress noise at the cost of fine detail.

### 2.2 The order matters

The standard pipeline sharpens first, then averages:

```
raw observation  -->  sharpen (extract features)  -->  average (pool/aggregate)  -->  compare
```

This works when:
- Signal-to-noise ratio is high (features capture structure, not noise)
- Enough data to learn which fine structures are signal vs noise

It fails when:
- SNR is low (sharpening amplifies noise beyond recovery)
- Too few examples to learn the distinction

The alternative inverts the order:

```
raw observation  -->  average (marginalise along identity axis)  -->  compare
```

This works when:
- There exists an axis along which identity-relevant structure is organised
- Noise is distributed across other axes (orthogonal to identity)
- The marginalisation integrates out the noise before any sharpening can amplify it

### 2.3 Why this follows logically

Sharpening is a **variance-increasing** operation on the input. It maps small differences in the input to large differences in the output (that is its purpose — to reveal fine structure). When noise is present, those small random differences get amplified too. With few examples (k=3), there is no statistical basis to distinguish amplified signal from amplified noise.

Marginalisation is a **variance-reducing** operation. It maps many input values to their average, suppressing uncorrelated variation. When noise is independent across the marginalised axis, the noise power reduces by a factor of ~1/n (where n is the number of bins being averaged). The identity signal, being correlated along the identity axis, survives the averaging.

The mathematical basis is the same as why the **sample mean** is a better estimator than any single observation: averaging reduces variance proportional to sample size. Marginalisation applies this principle spatially rather than across repeated measurements.

---

## 3. The Method

### 3.1 Identity axis identification

For a given domain, identify the axis (or axes) along which identity-relevant structure is organised. This is a **theoretical commitment**, not a learned parameter:

| Domain | Observation | Identity axis | Marginalisation produces |
|---|---|---|---|
| Speech | Spectrogram | Time (phoneme sequence) | Temporal energy envelope |
| Crystallography | Diffraction pattern | Radial (d-spacing) | Powder diffraction profile |
| Gel electrophoresis | Gel image | Vertical (molecular weight) | Band intensity profile |
| Layered tissue (histology) | Section image | Depth (layer structure) | Depth density profile |
| Seismology | Waveform / spectrogram | Time + frequency | Energy envelope + spectral profile |
| Chromatography | Chromatogram | Retention time | Already a 1D profile |

The identity axis encodes the analyst's theory of **what makes two observations the same kind of thing**. Choosing the horizontal axis for a 2D image encodes the belief that identity is defined by the horizontal distribution of features. Choosing the radial axis for crystallography encodes the belief that crystal phases are defined by their d-spacings.

### 3.2 Marginal computation

Divide the observation along the identity axis into *n* equal bins. For each bin, compute the **mean activation** (spectral energy, pixel intensity, signal amplitude, etc.) by averaging over all other dimensions. The result is an *n*-dimensional vector: the marginal distribution of the observation along the identity axis.

Optionally compute a secondary marginal along a perpendicular axis. Concatenate. When the identity axis is uncertain, combining multiple axes is the robust default.

No learned parameters. The only choices are: which axis, and how many bins.

### 3.3 Identity matching

Compare marginal profiles using **cosine similarity**. Same-identity observations have similar profiles because the structure along the identity axis is the same; noise along other axes has been integrated out.

### 3.4 Seed-and-propagate concept discovery

Given *k* labeled seed observations (2-10), compute the mean marginal profile per concept. Rank all unlabeled observations by cosine similarity to each concept's mean profile. Assign observations above a threshold. Optionally iterate: expand seed sets, recompute means, re-evaluate.

---

## 4. The Diagnostic

### 4.1 Discrimination gap

For any representation *f*, compute:

- **S_within**: cosine similarity between same-class instances in representation *f*
- **S_between**: cosine similarity between different-class instances in representation *f*
- **Gap** = min(S_within) - max(S_between)

If gap > 0: the representation preserves identity signal — every same-class pair is more similar than every cross-class pair.

If gap < 0: overlap exists — the representation confuses some classes.

The gap is a **ranking signal**: the representation with the least-negative (or most-positive) gap is the best representation. It can be computed from as few as 2 same-class and 1 different-class observation. No gradient, no training, no test set.

### 4.2 Representation quality diagnostic

Compare the gap for the raw observation vs the representation:

| Raw gap | Representation gap | Diagnosis |
|---|---|---|
| Positive | Positive (larger) | Representation adds value |
| Positive | Positive (smaller) | Representation loses some signal but still works |
| Positive | Negative | **Representation destroys signal** — revert to raw or marginalise |
| Negative | Negative | Signal not present at this level — need different abstraction |
| Negative | Positive | Representation extracts latent signal (the ideal case for learned features) |

### 4.3 Use as a self-directed learning signal

An AI agent exploring representation strategies uses the gap as follows:

1. Propose a representation (choice of axis, number of bins)
2. Compute the gap on the seed set
3. Read the confusion matrix — which classes are confused?
4. Reason about **why** (shared spectral content? similar duration? same onset shape?)
5. Propose a targeted revision (add a new axis, change resolution, combine axes)
6. Re-measure

This is **causal diagnostic reasoning**, not black-box optimisation. The agent doesn't just maximise a score — it understands why the score is low and what to change. The search space is small (which axis, how many bins, which combinations) so convergence is fast.

---

## 5. Hypotheses

### H1: Sharpening underperforms marginalisation in the low-data, high-noise regime

**Testable prediction:** For any domain where (a) same-class raw observations have positive cosine similarity and (b) fewer than 10 labeled examples are available, a marginal projection along the correct identity axis will produce a larger discrimination gap than any sharpening-based feature extraction.

**Status:** Confirmed in two domains. Sharpening (edge detection) was the worst performer in every experimental condition tested.

### H2: The discrimination gap ranks representations correctly

**Testable prediction:** The representation with the largest (least-negative) gap on a 3-example seed set will also have the highest classification accuracy on a held-out test set.

**Status:** Confirmed as a ranking signal. The gap correctly orders representations by quality. However, it does not function as a binary classifier (gap > 0 does not guarantee high accuracy; gap < 0 does not guarantee failure).

### H3: The identity axis can be discovered by search

**Testable prediction:** An agent that tries projections along each axis and selects the one with the largest discrimination gap will find the correct identity axis within O(d) trials.

**Status:** Confirmed. In the seismic domain, an agent searching 9 configurations correctly identified the optimal representation without test labels.

### H4: Combined marginalisation is the robust default

**Testable prediction:** When the identity axis is unknown, combining marginals along multiple candidate axes produces more robust accuracy than any single-axis marginal.

**Status:** Confirmed. The combined profile (spectral + temporal) was within 3% of the best single-axis choice at every noise level, while single-axis choices varied by 20%+.

---

## 6. Relationship to Existing Work

### What this builds on
- **Projection profiles** in document analysis (Rath & Manmatha, 2003; standard since 1970s)
- **Histogram-based matching** (Swain & Ballard, 1991)
- **Sufficient statistics** (Fisher, 1920s): the marginal is a low-dimensional summary that preserves identity-relevant information
- **Bayesian marginalisation** — the principle that nuisance parameters should be integrated out before inference
- **Label propagation** (Zhu & Ghahramani, 2002)
- **Prototype networks** (Snell et al., 2017) — average-then-compare in embedding space

### What this adds
- The **sharpening-vs-averaging dichotomy** as an explicit design principle for representation construction
- The argument that **order of operations** (sharpen-then-average vs average-then-compare) determines whether noise is amplified or suppressed, and that this ordering is the critical design choice in the low-data regime
- The **discrimination gap** as a model-free, label-efficient diagnostic for iterative representation selection
- The **identity axis** concept: the marginalisation axis encodes domain theory about what constitutes identity
- A **self-directed diagnostic loop** where an AI agent uses gap measurement + confusion analysis + causal reasoning to iteratively revise representations

### What this does NOT claim
- That marginalisation is universally better than learned features (it isn't — learned features win with enough data)
- That projection profiles are novel (they aren't)
- That this replaces deep learning (it doesn't — it addresses a specific regime where deep learning is data-starved)
- That the gap is a perfect diagnostic (it's a ranking signal, not a binary classifier)

---

## 7. Operating Regime

The method is advantageous when:

| Condition | Why it matters |
|---|---|
| **Few labeled examples** (2-10 per class) | Learned representations can't converge; marginalisation has no parameters to learn |
| **High noise** (low SNR) | Sharpening amplifies noise; marginalisation suppresses it |
| **Variable-size observations** | Raw pixel comparison requires normalisation that distorts; marginals are inherently size-invariant |
| **Structured noise** | Noise that isn't iid Gaussian (e.g., artefacts, interference) doesn't average out in pixel space but does in profile space |
| **No pretrained model** | No transfer learning available; must work from the raw signal |

The method loses its advantage when:
- Abundant labeled data is available (> 100 per class) — learned features are better
- Observations have fixed, uniform size — raw pixel cosine works fine
- Noise is iid Gaussian and SNR is moderate+ — high-dimensional cosine implicitly averages

---

## 8. Experimental Validation

### Seismic Event Classification

Seven event classes, variable duration (0.8-32s), coloured noise, 3 seeds per class. Tested across SNR 0-30 dB.

- Sharpening (edge detection) was worst at every SNR level (29-95%, mean 80%)
- At SNR 3 dB: combined spectral + waveform envelope profile (66%) beat raw pixels (62%)
- At SNR 5 dB: profiles tied raw pixels (73%)
- At SNR 8+ dB: raw pixels pulled ahead
- Gap diagnostic correctly ranked all representation configurations
- Identity axis discovery via gap search correctly identified the optimal axis

The crossover point — where marginalisation loses its advantage — was around SNR 8-10 dB for this domain.

See `applications/seismic-event-classification/` for full experimental data and the agent's self-directed investigation record.
