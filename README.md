# Few-Shot Gap Diagnostic

## What this is

When someone hands you three examples of something and says "find more like these," the hard part isn't the searching — it's figuring out what "like these" means. What should you pay attention to? The shape? The timing? The frequency content? The spatial pattern?

With thousands of examples, a neural network can figure this out by itself. With three examples, it can't. Something has to decide how to look at the data before comparison becomes meaningful.

The method presented here lets an AI agent make that decision autonomously. It proposes ways of looking at the signal, tests each one against the few examples it has, diagnoses what's going wrong when things don't work, and revises — converging on a representation that separates the classes, even from minimal data.

## The core idea

Every signal has structure along different axes. A spectrogram has time and frequency. A micrograph has x and y. A diffraction pattern has angle and radius. The question is: **which axis carries the identity?**

If you're classifying bird calls, the identity is in *when* the energy appears (the temporal pattern of chirps and trills). If you're classifying crystal phases, the identity is in *where* the peaks fall (the angular distribution of diffraction). If you're classifying seismic events, it might be frequency content, or temporal shape, or both — and you don't know in advance.

The method works by trying each axis, measuring whether it helps, and reasoning about why it doesn't when it doesn't.

## Why it works

There are really only two things you can do to a signal before comparing it: **sharpen** it or **smooth** it.

Sharpening (edge detection, peak picking, thresholding) amplifies fine detail. That's useful when the fine detail IS the signal. But it also amplifies noise. And when you only have 3 examples, you can't tell the difference between amplified signal and amplified noise.

Smoothing — specifically, **marginalising** along an axis (computing the average projection) — does the opposite. It throws away fine detail and keeps the broad shape. That loses information, but the information it loses is exactly the information you can't reliably estimate from 3 examples anyway.

The statistical argument is simple: **with 3 examples, you can estimate a mean but not a variance.** Marginalisation gives you the mean structure along an axis. Sharpening gives you the fine structure, which you'd need hundreds of examples to interpret reliably. So marginalise first, compare second.

This is the same logic as why the sample mean is a better estimator than any single observation. It's Bayesian marginalisation applied spatially: integrate out the dimensions you can't afford to model, keep the dimensions that carry identity.

## The diagnostic loop

The agent doesn't just pick a representation and hope. It runs a loop:

```
1. PROPOSE — pick an axis to marginalise along (e.g., "try the time axis")

2. MEASURE — compute the discrimination gap:
   gap = min(within-class similarity) - max(between-class similarity)
   using only the 3 seed examples per class

3. DIAGNOSE — if the gap is poor:
   - Which classes are confused?
   - What do the confused classes share?
   - What distinguishes them that the current representation misses?

4. REVISE — based on the diagnosis:
   - Wrong axis → try another
   - Right axis, wrong resolution → adjust bins
   - Missing information → add a second axis
   - Too much noise → increase smoothing

5. RE-MEASURE — did it help?
```

This isn't optimisation. The agent doesn't blindly maximise a score. It **understands why the score is bad** and makes a targeted fix. That's what makes it work with so little data — domain reasoning substitutes for statistical power.

## The discrimination gap

The gap is the method's compass. For any proposed representation:

- Compare all within-class seed pairs (cosine similarity)
- Compare all between-class seed pairs (cosine similarity)
- **Gap** = worst within-class similarity minus best between-class similarity

**Gap > 0** means every same-class pair is more similar than every cross-class pair. The representation works.

**Gap < 0** means there's overlap. But even then, **less negative = better**. The gap ranks representations correctly even when none of them are perfect. And it's computable from 3 examples per class, in milliseconds, with no training.

## What makes this different from existing approaches

**Prototype Networks** (Snell et al., 2017) also average then compare — but they need a learned embedding trained on thousands of prior tasks. This method needs no prior tasks. It works on the raw signal.

**MAML and meta-learning** (Finn et al., 2017) learn how to learn from few examples — but they need a distribution of similar tasks to meta-train on. This method needs zero prior experience.

**AutoML and NAS** search over architectures using validation performance — but they need hundreds of examples to evaluate each candidate. The gap diagnostic needs 3.

**CAAFE** (Hollmann et al., 2023) uses an LLM to propose features — but it proposes based on dataset descriptions, not based on measured failure modes. It doesn't know *why* a feature failed.

**The AI Scientist** (Sakana AI, 2024) runs LLM-directed experimental loops — a similar general framework. But it operates at the level of "try something else." The gap diagnostic tells the agent *specifically* what's wrong and what to change.

**Bayesian marginalisation** integrates out nuisance parameters before inference — the same underlying principle. But it's applied to model parameters, not to spatial dimensions of the input signal. This method applies marginalisation as a representation strategy.

**Dunn Index** measures cluster separation similarly (min inter / max intra). But it's used to evaluate a completed clustering, not as an iterative search signal paired with causal diagnosis.

The novelty isn't in any single component. It's in the integration: a **diagnostic loop where causal reasoning about failure modes drives targeted representation revision**, using a worst-case gap as the compass, operating from 3 examples per class.

## When it works (and when it doesn't)

The method earns its keep when:
- **Few labeled examples** (2-10 per class) — too few for learned representations to converge
- **High noise** — sharpening amplifies noise; marginalisation suppresses it
- **Variable-size observations** — raw pixel comparison requires normalisation that distorts; marginals are inherently size-invariant
- **No pretrained model** — the agent must work from the raw signal

It loses its advantage when:
- Abundant labeled data is available (100+ per class) — learned features will do better
- Observations have fixed, uniform size — raw cosine comparison works fine
- Noise is low and Gaussian — high-dimensional comparison implicitly averages it

## Validation

Validated on synthetic seismic event classification: 7 event types, variable duration (0.8-32s), coloured noise, 3 seeds per class. Over 3 self-directed iterations, the agent:

1. Discovered that temporal profiles fail for variable-length events (diagnosis: degenerate spectrograms for short events)
2. Discovered that scalar temporal descriptors fail with 3 seeds (diagnosis: too few dimensions, too much variance)
3. Succeeded by combining spectral profile + waveform envelope — beating raw pixels at SNR 3 dB (66% vs 62%), while sharpening (edge detection) scored 29%

The gap diagnostic correctly ranked all representation configurations. The agent found the right combination by diagnosing each failure and targeting the fix. Full experimental record in `applications/seismic-event-classification/`.

## Repository Structure

```
ai-self-directed-gap-diagnostic/
├── README.md                                  ← you are here
├── marginalisation-before-sharpening.md       ← formal hypothesis document
└── applications/
    └── seismic-event-classification/          ← first validated application
        ├── README.md                          ← what worked, what failed, key numbers
        ├── investigation_log.md               ← the agent's decision journal
        ├── experiment_results.md              ← honest results with revisions
        └── *.py                               ← all experimental code (numpy + scipy)
```

## How to apply to a new domain

1. Collect 3+ labeled examples per class
2. Identify candidate axes of the raw observation (time, frequency, space, wavelength...)
3. Compute marginal profiles along each axis
4. Measure the discrimination gap — pick the axis with the best gap
5. Read the confusion matrix — which classes are still confused?
6. Ask *why* — what do the confused classes share that the representation misses?
7. Add that missing axis, re-measure
8. Classify by cosine similarity to seed-set means

## Licence

Released for research purposes. If you use this method, please cite this repository.
