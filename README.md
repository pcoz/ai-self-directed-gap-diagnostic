# Few-Shot Gap Diagnostic

A method for self-directed representation selection in extreme few-shot classification, where an AI agent iteratively selects and revises signal representations using a worst-case discrimination gap as its diagnostic signal.

## The Problem

Classify observations into categories given only 2-3 labeled examples per class. No pretrained model. Noisy observations. Variable-size inputs. Which representation of the raw signal should you compare?

This is a common situation in scientific and field settings: a geologist labels three example tremors, an ecologist tags three bird calls, a materials scientist marks three micrograph regions. They need the system to find more of the same. There is no training corpus. There is no pretrained model for their specific domain. There are 3 examples and a question: *what else looks like this?*

Standard approaches fail here for specific, identifiable reasons:

- **AutoML / NAS** search over model architectures and hyperparameters. They require a validation set large enough to evaluate each candidate — typically hundreds of examples. With 3 examples, the search signal is pure noise.
- **Learned embeddings** (deep metric learning, contrastive learning) require enough data to learn which features matter. With 3 examples, the network memorises the instances rather than learning the class structure.
- **Handcrafted features** (domain-specific feature engineering) can work, but the standard approach — extract fine-grained features first, then compare — is a sharpening operation that amplifies noise when the observations are noisy. The features that a domain expert would design for clean data actively hurt on noisy data with 3 examples.

## Why This Should Work: The Logical Underpinning

The method rests on a simple statistical argument:

**Sharpening increases variance. Averaging decreases variance. With 3 examples, you cannot afford increased variance.**

More precisely:

1. **Any representation is a transformation of the raw observation.** That transformation either amplifies fine-scale variation (sharpening) or suppresses it (averaging/marginalisation).

2. **Identity matching requires estimating whether two observations come from the same distribution.** With k=3 samples, your estimate of the class distribution has high variance. Any representation that *increases* the variance of individual observations (sharpening) makes this estimation harder. Any representation that *decreases* the variance (marginalisation) makes it easier.

3. **Marginalisation along an axis preserves structure that is correlated along that axis and suppresses structure that is uncorrelated.** If identity-relevant features are organised along a specific axis (e.g., temporal sequence, spectral content, spatial arrangement), then marginalising along that axis preserves the identity signal. Noise that is uncorrelated with the identity axis gets averaged out — by a factor of ~1/n where n is the number of values being aggregated.

4. **The choice of marginalisation axis is a theoretical commitment.** It encodes the analyst's (or agent's) hypothesis about what makes two observations "the same thing." This is not a parameter to learn — it is a structural assumption about the domain. Choosing time as the identity axis says "things that have the same temporal pattern are the same kind of thing." This assumption can be tested empirically via the discrimination gap.

5. **The discrimination gap provides a gradient-free, label-efficient test of whether the assumption is correct.** Compute within-class and between-class cosine similarities on the 3 seed examples. If the gap is negative, the representation is wrong — either the wrong axis, the wrong resolution, or the wrong combination. If the gap improves (becomes less negative or positive), the revision helped. This is testable in milliseconds with no training.

The method is therefore a **hypothesis-test loop**: propose an axis (hypothesis about what constitutes identity), test it (gap measurement), diagnose failures (confusion analysis), revise, repeat. The agent does not optimise a loss function — it reasons about *why* the current representation fails and proposes a targeted fix.

## Related Methods and How This Differs

### Prototype Networks (Snell et al., 2017)
Compute class prototypes as the mean embedding of support examples, then classify by nearest prototype. This is "average then compare" in embedding space — structurally similar. **Difference:** Prototype networks require a learned embedding (trained on a meta-learning corpus). This method requires no learned embedding — it operates on marginals of the raw observation. It also includes a diagnostic loop that revises the representation, which prototype networks do not.

### MAML / Meta-Learning (Finn et al., 2017)
Learn an initialisation that can be fine-tuned from few examples. **Difference:** MAML requires a distribution of tasks to meta-train on. This method requires zero prior tasks — it works from the raw signal and 3 labeled examples. MAML optimises; this method diagnoses.

### AutoML / Neural Architecture Search
Search over model architectures using validation performance as signal. **Difference:** Requires a validation set large enough to distinguish candidate architectures (typically hundreds of examples). The gap diagnostic requires 3 examples. AutoML treats the search as black-box optimisation; this method uses causal reasoning about failure modes.

### CAAFE (Hollmann et al., 2023)
LLM-based automated feature engineering: an LLM proposes features based on dataset descriptions. **Difference:** CAAFE proposes features based on dataset metadata, not based on measured failure modes. It does not diagnose *why* a representation fails or propose targeted revisions. The gap diagnostic provides the feedback signal that CAAFE lacks.

### The AI Scientist (Lu et al., 2024, Sakana AI)
LLM agent that generates hypotheses, runs experiments, analyses results. The general framework is similar. **Difference:** The AI Scientist operates at the level of "this approach didn't work, try another." The gap diagnostic provides a *specific* diagnostic: which classes are confused, why, and what representational change would fix it. The granularity of diagnosis is finer.

### Bayesian Marginalisation
The principle of integrating out nuisance parameters before inference is core to Bayesian statistics. This method applies the same principle spatially: marginalise over the non-identity axes of the observation. **Difference:** Bayesian marginalisation is typically over model parameters. This is marginalisation over spatial/temporal dimensions of the input signal, used as a representation strategy rather than an inference technique.

### Dunn Index / Silhouette Score
Cluster validity indices that measure within-cluster vs between-cluster separation. The discrimination gap is structurally similar to the Dunn index. **Difference:** Cluster validity indices are used to evaluate a *completed* clustering. The discrimination gap is used as an *iterative search signal* to guide representation selection, combined with causal diagnosis of failure modes. The min/max formulation (rather than means) makes it a worst-case diagnostic — it asks "does the *weakest* within-class pair still beat the *strongest* between-class pair?"

## The Method

An autonomous agent executes a diagnostic loop:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   1. PROPOSE a representation                       │
│      (projection axis, bin count, combination)      │
│                                                     │
│   2. MEASURE the discrimination gap                 │
│      gap = min(within-class cosine similarity)      │
│          - max(between-class cosine similarity)      │
│      using only the seed examples                   │
│                                                     │
│   3. DIAGNOSE                                       │
│      - Which classes are confused? (confusion matrix)│
│      - WHY? (shared spectral content? similar       │
│        duration? same onset shape?)                 │
│      - Is the representation sharpening or          │
│        averaging the raw signal?                    │
│                                                     │
│   4. REVISE the representation                      │
│      - If sharpening destroyed signal → marginalise │
│      - If wrong axis → try another                  │
│      - If confused classes differ in X → add X      │
│      - If a feature has too much variance →         │
│        increase dimensionality for redundancy       │
│                                                     │
│   5. RE-MEASURE and compare to previous iteration   │
│                                                     │
└──────────────────── repeat ─────────────────────────┘
```

### The Discrimination Gap

For a set of seed examples grouped by class, compute cosine similarity between all pairs:

- **Within-class pairs**: same-class seed examples compared to each other
- **Between-class pairs**: different-class seed examples compared to each other

The **gap** = min(within) - max(between).

- **Gap > 0**: every same-class pair is more similar than every cross-class pair. The representation preserves identity.
- **Gap < 0**: some cross-class pair is more similar than some same-class pair. The representation confuses classes.
- **Less negative gap = better representation**, even when all gaps are negative.

The gap requires only the seed examples (3 per class). No test set. No training. Computable in milliseconds.

### The Diagnostic Reasoning

What distinguishes this from black-box optimisation (AutoML, random search, Bayesian optimisation) is that the agent reasons about **why** a representation fails:

1. **Read the confusion matrix**: which specific classes are being confused?
2. **Analyse the confused classes**: what do they share? What distinguishes them?
3. **Identify the missing axis**: the representation captures X but the confused classes differ in Y.
4. **Propose a targeted fix**: add a marginalisation along Y, or change the resolution of X.

This is **causal diagnostic reasoning**, not gradient descent or random search. It works with 3 examples because it uses domain knowledge to constrain the search, not data volume to explore blindly.

### The Marginalisation Principle

When choosing representations, the agent follows a design principle:

> **Average before you compare. Don't sharpen before you compare.**

- **Sharpening operations** (edge detection, peak detection, deconvolution, thresholding) amplify fine structure, including noise. They make same-class instances look MORE different.
- **Marginalisation operations** (histograms, projection profiles, pooling, binning) suppress fine structure by averaging over dimensions. They make same-class instances look MORE similar.

In the few-shot, high-noise regime, sharpening is catastrophic because the noise it amplifies cannot be averaged out by the 3 seed examples. Marginalisation suppresses noise first, at the cost of fine detail that isn't estimable from 3 examples anyway.

**When marginalisation wins**: variable-size observations, structured noise, extreme few-shot.
**When raw comparison wins**: fixed-size observations, iid Gaussian noise, moderate+ sample sizes.

## What's Novel

| Component | Status |
|-----------|--------|
| LLM running experimental loops | Established (AI Scientist, Coscientist, FunSearch) |
| Automated feature selection | Established (AutoML) |
| **Worst-case gap as iterative diagnostic** | **Novel as a diagnostic tool.** Dunn index is structurally similar but used for cluster validation, not iterative representation revision |
| **Semantic causal diagnosis of confusion** | **Novel.** No existing system reasons about WHY classes are confused and proposes targeted representational fixes |
| **Diagnostic-driven representation revision** | **Novel.** AutoML optimises blindly; this diagnoses with causal reasoning |
| **All of the above with 3 examples per class** | **Novel operating regime** |
| **Marginalisation-before-comparison principle** | **Novel articulation** of an idea with Bayesian roots |

The novelty is in the **integration**: a diagnostic loop where causal reasoning about failure modes drives targeted representation revision in extreme few-shot. No single component is new; the methodology is.

## Repository Structure

```
few-shot-gap-diagnostic/
├── README.md                                  ← this file
├── marginalisation-before-sharpening.md       ← full hypothesis document with theory
└── applications/
    └── seismic-event-classification/          ← first validated application
        ├── README.md                          ← application-specific description
        ├── investigation_log.md               ← self-directed investigation journal
        ├── experiment_results.md              ← honest results with revisions
        ├── experiment_seismic.py              ← v1: initial 5-class experiment
        ├── experiment_seismic_v2.py           ← v2: SNR sweep, axis discovery
        ├── seismic_heatmap.py                 ← heat map: 9 reps × 8 SNR levels
        ├── iteration1_hard_seismic.py         ← 7-class variable-size problem
        ├── iteration2_informed_features.py    ← scalar temporal descriptors (failed)
        └── iteration3_waveform_profiles.py    ← waveform envelope profiles (succeeded)
```

## How to Apply to a New Domain

1. **Generate or obtain observations** with known class labels for a small seed set (3+ per class)
2. **Identify candidate identity axes** — dimensions along which class-defining structure might be organised
3. **Run the diagnostic loop**:
   - Compute marginal profiles along each candidate axis
   - Measure the discrimination gap on seeds
   - Select the axis with the least-negative (or most-positive) gap
4. **Analyse the confusion matrix** for the selected representation
5. **If classes are confused**: identify what distinguishes them, add that as an additional axis
6. **Combine axes** that capture complementary identity information
7. **Classify** unlabeled observations by cosine similarity to seed-set means

## Validation

Validated on synthetic seismic event classification (7 classes, variable duration, coloured noise, 3 seeds per class). Combined spectral + waveform envelope profile (66%) beat raw pixels (62%) at SNR 3 dB. Sharpening (edge detection) was worst at every noise level. Gap diagnostic correctly ranked all representation configurations. See `applications/seismic-event-classification/` for full experimental record.

## Licence

This work is released for research purposes. If you use this method, please cite the source repository.
