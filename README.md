# Few-Shot Gap Diagnostic: Self-Directed Signal Analysis for AI Agents

## The problem of analysing with almost no examples

When someone hands you three examples of something and says "find more like these," the hard part isn't the searching — it's figuring out what "like these" means. What should you pay attention to? The shape? The timing? The frequency content? The spatial pattern?

With thousands of examples, a neural network can figure this out by itself. **With three examples, it can't.** Something has to decide *how to look at the data* before comparison becomes meaningful.

The method presented here lets an AI agent make that decision autonomously. It proposes ways of looking at the signal, **tests each one** against the few examples it has, **diagnoses** what's going wrong when things don't work, and **revises** — converging on a representation that separates the classes, even from minimal data.

## Identity lives along an axis

Every structured signal carries information along multiple axes. A spectrogram varies across time and frequency. A micrograph varies across x and y. A diffraction pattern varies across angle and radius.

Some of these axes carry information about *what kind of thing* the signal represents. Others carry noise, or context, or irrelevant variation. The central question is: **which axis carries the identity?**

For bird calls, identity lives in the **temporal axis** — the pattern of chirps and silences over time. For crystal phases, identity lives in the **angular axis** — where the diffraction peaks fall. For seismic events, it could be frequency content, temporal envelope, or both. You often don't know in advance which it is.

The method works by trying each axis in turn, measuring how well it separates the known classes, and — crucially — **reasoning about *why* a given axis fails** when it does, so the next attempt is informed rather than random.

## Why it works

Before you can compare two signals, you have to transform them into something comparable. There are fundamentally only two kinds of transformation: you can **sharpen** the signal or you can **smooth** it.

**Sharpening** — edge detection, peak picking, thresholding — amplifies fine-scale structure. When that fine structure is the signal you care about, this is exactly right. But sharpening amplifies noise equally. With hundreds of examples you can average out the noise after sharpening. With three, you can't. The noise dominates, and same-class instances end up looking *more different* from each other in the sharpened representation than they did in the raw signal. **The representation has made the problem harder.**

**Smoothing** — specifically, **marginalising** along an axis by computing the average projection — does the opposite. It collapses fine-scale variation into a summary. You lose detail, but the detail you lose is precisely the detail that three examples cannot reliably characterise. What survives marginalisation is the broad structure along the chosen axis: where energy concentrates, how the signal is distributed, what the overall shape looks like. **That broad structure is estimable from three examples. Fine structure is not.**

The statistical intuition is clean: **with three examples you can estimate a mean, but not a variance.** Marginalisation gives you the mean profile along an axis. Sharpening gives you fine structure that requires variance estimates to interpret. So: marginalise first, then compare.

This is Bayesian marginalisation applied to signal representation — integrate out the dimensions you can't afford to model, keep the dimensions that carry identity. (The full theoretical argument is developed in [marginalisation-before-sharpening.md](marginalisation-before-sharpening.md).)

## The diagnostic loop

Choosing the right axis isn't a one-shot decision. The agent runs an **iterative diagnostic loop**:

```
1. PROPOSE — pick an axis to marginalise along (e.g., "try the time axis")

2. MEASURE — compute the discrimination gap on the seed examples:
   gap = min(within-class similarity) - max(between-class similarity)

3. DIAGNOSE — if the gap is poor, ask:
   - Which specific classes are being confused?
   - What do those confused classes have in common?
   - What distinguishes them that the current representation doesn't capture?

4. REVISE — make a targeted change based on the diagnosis:
   - Wrong axis → try a different one
   - Right axis, wrong resolution → adjust the bin count
   - Two axes each capture part of the identity → combine them
   - Representation too noisy → increase smoothing

5. RE-MEASURE — did the revision improve the gap? If not, diagnose again.
```

What distinguishes this from black-box optimisation is **step 3**. The agent doesn't just register that the score went down — it examines the confusion matrix, identifies which classes overlap, reasons about what those classes share and how they differ, and proposes a fix that addresses the **specific failure**. Domain reasoning substitutes for statistical power. That's how it works with three examples.

## The discrimination gap

The gap is the method's compass. Given a proposed representation and a set of seed examples with known class labels:

- Compute **cosine similarity** between every pair of same-class seeds *(within-class)*
- Compute **cosine similarity** between every pair of different-class seeds *(between-class)*
- **Gap** = the **worst** within-class similarity **minus** the **best** between-class similarity

A **positive gap** means the representation cleanly separates all classes — even the most similar cross-class pair is less similar than the least similar within-class pair. A **negative gap** means there is overlap, but the magnitude still tells you something: a gap of -0.001 is much closer to working than a gap of -0.5.

In practice, the gap rarely goes positive with only 3 seeds. But it **consistently ranks representations correctly**: the representation with the least-negative gap is the one with the highest classification accuracy. That ranking property is what makes the diagnostic loop converge.

The gap is computable in milliseconds from just the seed examples. **No training. No test set. No gradient computation.**

## How this differs from existing approaches

- **Prototype Networks** (Snell et al., 2017) also average then compare, but require a **learned embedding** trained on thousands of prior tasks. This method operates on marginals of the raw signal — no learned embedding, no prior tasks.

- **MAML and meta-learning** (Finn et al., 2017) learn an initialisation that adapts quickly from few examples, but require a **distribution of similar tasks** to meta-train on. This method starts from the raw signal and three labels with no prior experience.

- **AutoML and neural architecture search** evaluate candidates using validation performance, needing **hundreds of examples** to distinguish them. The gap diagnostic evaluates from 3.

- **CAAFE** (Hollmann et al., 2023) uses an LLM to propose features from dataset descriptions, but **does not diagnose** why a feature fails or revise based on failure analysis. The gap diagnostic provides exactly that feedback signal.

- **The AI Scientist** (Sakana AI, 2024) runs LLM-directed experimental loops — a similar general framework, but operating at the level of "try something else." The gap diagnostic provides a **specific diagnosis**: which classes are confused, what they share, what distinguishes them — enabling targeted rather than exploratory revision.

- **Bayesian marginalisation** applies the same underlying principle — integrate out what you can't model — but to **model parameters**. This method applies it to **spatial and temporal dimensions** of the input signal as a representation strategy.

- **The Dunn Index** measures cluster separation with a similar min/max formulation, but evaluates a **completed clustering** after the fact. The discrimination gap is an **iterative steering signal** during representation search, paired with causal diagnosis.

No single component here is new. The novelty is in their integration: **a diagnostic loop where causal reasoning about failure modes — grounded in a measurable worst-case gap — drives targeted representation revision**, all operating in the extreme few-shot regime.

## When it works and when it doesn't

The method is designed for a specific operating regime. It is not a general-purpose classifier.

**It works well when:**
- There are **very few labeled examples** (2-10 per class) — too few for learned representations to converge
- The observations are **noisy** — sharpening would amplify the noise, but marginalisation suppresses it
- The observations **vary in size or shape** — raw comparison requires normalisation that distorts; marginal profiles are inherently size-invariant
- There is **no pretrained model** for the domain — the agent must work from the raw signal alone

**It loses its advantage when:**
- Labeled data is **abundant** (100+ per class) — learned features outperform fixed projections
- Observations are **uniform in size** and format — raw high-dimensional comparison already works
- Noise is **low** — there is enough signal for sharpening-based features to work without amplifying noise beyond recovery

## Validation

The method was validated on **synthetic seismic event classification**: 7 event types with variable duration (0.8-32 seconds), coloured 1/f noise, and only 3 seed examples per class. The agent ran three self-directed iterations:

1. **Iteration 1** tested temporal and spectral profiles on variable-length spectrograms. Temporal profiles **failed catastrophically** (36% accuracy) because short events produced degenerate spectrograms with as few as 2 time bins. The gap diagnostic correctly flagged this — temporal configurations had gaps of -0.92, while spectral configurations had gaps of -0.03. *Diagnosis: the temporal axis has inconsistent resolution across events of different duration.*

2. **Iteration 2** attempted to fix the temporal problem by adding scalar temporal descriptors (duration, onset sharpness, decay rate). This **made accuracy worse** (41%), not better. *Diagnosis: 7 scalar features estimated from 3 seeds have too much variance — a single outlier seed shifts the class centroid enough to misclassify the majority.* The fix needed more dimensions, not fewer.

3. **Iteration 3** replaced spectrogram-derived temporal features with **waveform envelope profiles** — energy distribution computed directly from the raw signal, which has consistent time resolution regardless of event duration. Combined with the spectral profile, this produced **66% accuracy at SNR 3 dB**, beating raw pixel comparison (62%) and far exceeding sharpening-based edge detection (29%).

At each iteration the gap diagnostic **correctly ranked all configurations**, and the confusion matrix pointed to the specific classes and failure modes that informed the next revision.

Full code, data, and the agent's [decision journal](applications/seismic-event-classification/investigation_log.md) are in [applications/seismic-event-classification/](applications/seismic-event-classification/).

## Further reading

- **[Formal hypothesis and theory](marginalisation-before-sharpening.md)** — the full argument for why marginalisation should precede comparison, with testable hypotheses, experimental revisions, and operating regime analysis
- **[Seismic event classification](applications/seismic-event-classification/)** — the first validated application, including the agent's [decision journal](applications/seismic-event-classification/investigation_log.md), [honest results](applications/seismic-event-classification/experiment_results.md), and all experimental code

## How to apply to a new domain

1. **Collect** 3 or more labeled examples per class
2. **Identify candidate axes** of the raw observation — time, frequency, space, wavelength, or any other dimension along which the signal varies
3. **Compute marginal profiles** along each candidate axis
4. **Measure the discrimination gap** for each — pick the axis with the best (least-negative) gap
5. **Examine the confusion matrix** — which classes are still being confused?
6. **Reason about *why*** — what do those confused classes share that the current representation doesn't distinguish?
7. **Add a second axis** that captures the missing distinction, and re-measure
8. **Classify** unlabeled observations by cosine similarity to seed-class means

## Licence

Released for research purposes. If you use this method, please cite this repository.
