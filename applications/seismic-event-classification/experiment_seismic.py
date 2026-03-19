"""
Experiment: Marginalisation Before Sharpening
Domain: Seismic Event Classification

Problem: Classify seismic events from waveform data with only 3 labeled
seeds per class. Five event types with distinct temporal signatures
buried in realistic noise.

Event types:
  1. tectonic   — sharp P-wave, strong S-wave, long coda decay
  2. volcanic   — emergent onset, harmonic tremor, no clear S-wave
  3. explosion  — sharp onset, no S-wave, rapid decay, high frequency
  4. icequake   — very sharp onset, short duration, no coda
  5. rockfall   — emergent, broadband, irregular envelope, long duration

Methodology:
  1. Generate spectrograms (2D: frequency x time) for each event type
  2. Add realistic noise at controlled SNR
  3. Test 5 representations:
     a. Raw spectrogram pixels
     b. Edge detection (sharpening)
     c. Spectral peak detection (sharpening)
     d. Temporal marginal profile (marginalisation along identity axis)
     e. Spectral marginal profile (marginalisation along WRONG axis — control)
  4. Measure discrimination gap for each
  5. Run seed-and-propagate with best representation
  6. Report accuracy and validate hypotheses H1, H2
"""

import numpy as np
from scipy import signal
import os

np.random.seed(42)

# ============================================================================
# Synthetic seismogram generation
# ============================================================================

SAMPLE_RATE = 100  # Hz
DURATION = 20.0    # seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES)

def make_spectrogram(waveform, nperseg=64, noverlap=48):
    """Compute spectrogram, return as 2D image (freq x time)."""
    f, t, Sxx = signal.spectrogram(waveform, fs=SAMPLE_RATE,
                                    nperseg=nperseg, noverlap=noverlap,
                                    scaling='spectrum')
    # Log scale, clip, normalise to [0, 1]
    Sxx_log = np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-10)
    return f, t, Sxx_norm


def add_noise(waveform, snr_db=10):
    """Add Gaussian noise at specified SNR."""
    sig_power = np.mean(waveform**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
    return waveform + noise


def generate_tectonic(variation=0.3):
    """Tectonic earthquake: P-wave at ~3s, S-wave at ~7s, long coda."""
    w = np.zeros(N_SAMPLES)

    # P-wave arrival (sharp, high-freq)
    p_time = 3.0 + np.random.randn() * variation
    p_idx = int(p_time * SAMPLE_RATE)
    p_freq = 8 + np.random.randn() * 2
    p_env = np.exp(-np.abs(T - p_time) / 0.5) * (T >= p_time)
    w += p_env * np.sin(2 * np.pi * p_freq * T) * 0.5

    # S-wave arrival (larger, lower-freq)
    s_time = p_time + 4.0 + np.random.randn() * variation * 0.5
    s_freq = 3 + np.random.randn() * 1
    s_env = np.exp(-(T - s_time) / 2.0) * (T >= s_time)
    w += s_env * np.sin(2 * np.pi * s_freq * T) * 1.0

    # Coda (long exponential decay, low-freq)
    coda_start = s_time + 1.0
    coda_env = np.exp(-(T - coda_start) / 5.0) * (T >= coda_start)
    coda_freq = 1.5 + np.random.randn() * 0.5
    w += coda_env * np.sin(2 * np.pi * coda_freq * T) * 0.3

    return w


def generate_volcanic(variation=0.3):
    """Volcanic tremor: emergent onset, harmonic, sustained."""
    w = np.zeros(N_SAMPLES)

    # Emergent onset (gradual envelope)
    onset = 2.0 + np.random.randn() * variation
    rise_time = 3.0 + np.random.randn() * 0.5
    sustain = 8.0 + np.random.randn() * 1.0

    env = np.clip((T - onset) / rise_time, 0, 1) * np.exp(-np.maximum(0, T - onset - sustain) / 3.0)

    # Harmonic content (fundamental + harmonics)
    f0 = 2.0 + np.random.randn() * 0.3
    w += env * np.sin(2 * np.pi * f0 * T) * 0.8
    w += env * np.sin(2 * np.pi * 2 * f0 * T) * 0.4
    w += env * np.sin(2 * np.pi * 3 * f0 * T) * 0.2

    # Amplitude modulation (gliding)
    mod_freq = 0.3 + np.random.randn() * 0.1
    w *= (1 + 0.3 * np.sin(2 * np.pi * mod_freq * T))

    return w


def generate_explosion(variation=0.3):
    """Explosion: very sharp onset, no S-wave, rapid decay, broadband."""
    w = np.zeros(N_SAMPLES)

    # Sharp onset
    onset = 2.0 + np.random.randn() * variation * 0.5
    env = np.exp(-(T - onset) / 0.8) * (T >= onset)

    # Broadband: multiple frequencies
    for freq in [5, 10, 15, 20, 25]:
        phase = np.random.uniform(0, 2 * np.pi)
        amp = 1.0 / (1 + freq / 10)
        w += env * np.sin(2 * np.pi * freq * T + phase) * amp

    # No S-wave — just the initial blast decaying rapidly
    return w * 0.8


def generate_icequake(variation=0.3):
    """Icequake: extremely sharp, very short, high-frequency."""
    w = np.zeros(N_SAMPLES)

    onset = 3.0 + np.random.randn() * variation
    # Very short envelope (< 1 second)
    env = np.exp(-((T - onset) / 0.15)**2) * (T >= onset - 0.1)

    # High frequency content
    f1 = 15 + np.random.randn() * 3
    f2 = 25 + np.random.randn() * 5
    w += env * np.sin(2 * np.pi * f1 * T) * 1.0
    w += env * np.sin(2 * np.pi * f2 * T) * 0.5

    return w


def generate_rockfall(variation=0.3):
    """Rockfall: emergent, broadband, irregular, long."""
    w = np.zeros(N_SAMPLES)

    onset = 1.5 + np.random.randn() * variation
    duration = 10.0 + np.random.randn() * 2.0

    # Irregular envelope (sum of random bumps)
    env = np.zeros(N_SAMPLES)
    n_bumps = np.random.randint(8, 15)
    for _ in range(n_bumps):
        bump_time = onset + np.random.uniform(0, duration)
        bump_width = np.random.uniform(0.3, 1.5)
        bump_amp = np.random.uniform(0.3, 1.0)
        env += bump_amp * np.exp(-((T - bump_time) / bump_width)**2)

    # Broadband noise-like content
    for freq in np.random.uniform(2, 30, size=10):
        phase = np.random.uniform(0, 2 * np.pi)
        w += env * np.sin(2 * np.pi * freq * T + phase) * 0.2

    return w


GENERATORS = {
    'tectonic': generate_tectonic,
    'volcanic': generate_volcanic,
    'explosion': generate_explosion,
    'icequake': generate_icequake,
    'rockfall': generate_rockfall,
}


# ============================================================================
# Generate dataset
# ============================================================================

N_SEEDS = 3
N_TEST = 20
SNR_DB = 5  # challenging noise level

print("=" * 70)
print("  Generating synthetic seismic dataset")
print("=" * 70)
print(f"  5 event types, {N_SEEDS} seeds + {N_TEST} test per type")
print(f"  SNR = {SNR_DB} dB, duration = {DURATION}s, sample rate = {SAMPLE_RATE} Hz")
print()

dataset = []  # (class_name, is_seed, spectrogram, waveform)

for class_name, generator in GENERATORS.items():
    for i in range(N_SEEDS + N_TEST):
        waveform = generator(variation=0.4)
        waveform = add_noise(waveform, snr_db=SNR_DB)
        f, t, spec = make_spectrogram(waveform)
        is_seed = i < N_SEEDS
        dataset.append((class_name, is_seed, spec, waveform))

    print(f"  {class_name}: {N_SEEDS} seeds + {N_TEST} test, "
          f"spectrogram shape = {spec.shape}")

print(f"\n  Total: {len(dataset)} instances")
spec_shape = dataset[0][2].shape
print(f"  Spectrogram: {spec_shape[0]} freq bins x {spec_shape[1]} time bins")


# ============================================================================
# Representation functions
# ============================================================================

def rep_raw_pixels(spec):
    """Raw spectrogram flattened."""
    return spec.flatten()


def rep_edge(spec):
    """Edge detection (Sobel) — a sharpening operation."""
    from scipy.ndimage import sobel
    gx = sobel(spec, axis=1)
    gy = sobel(spec, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    return edges.flatten()


def rep_spectral_peaks(spec):
    """Peak detection per time frame — sharpening operation."""
    n_freq, n_time = spec.shape
    features = []
    for t_idx in range(n_time):
        col = spec[:, t_idx]
        # Find peaks
        peaks, props = signal.find_peaks(col, height=0.2, distance=3)
        # Encode as binary mask
        mask = np.zeros(n_freq)
        if len(peaks) > 0:
            mask[peaks] = col[peaks]
        features.append(mask)
    return np.array(features).flatten()


def rep_temporal_profile(spec, n_bins=32):
    """Marginal along TIME axis (column profile) — averaging operation.
    This IS the identity axis for seismic events."""
    n_freq, n_time = spec.shape
    edges = np.linspace(0, n_time, n_bins + 1).astype(int)
    profile = np.zeros(n_bins)
    for b in range(n_bins):
        s = spec[:, edges[b]:edges[b+1]]
        if s.size > 0:
            profile[b] = np.mean(s)
    return profile


def rep_spectral_profile(spec, n_bins=32):
    """Marginal along FREQUENCY axis (row profile) — averaging along WRONG axis.
    Control: this marginalises along the non-identity axis."""
    n_freq, n_time = spec.shape
    edges = np.linspace(0, n_freq, n_bins + 1).astype(int)
    profile = np.zeros(n_bins)
    for b in range(n_bins):
        s = spec[edges[b]:edges[b+1], :]
        if s.size > 0:
            profile[b] = np.mean(s)
    return profile


def rep_combined_profile(spec, col_bins=32, row_bins=16):
    """Both marginals: temporal (identity) + spectral (secondary)."""
    return np.concatenate([
        rep_temporal_profile(spec, col_bins),
        rep_spectral_profile(spec, row_bins),
    ])


REPRESENTATIONS = {
    'raw_pixels': rep_raw_pixels,
    'edge_detection': rep_edge,
    'spectral_peaks': rep_spectral_peaks,
    'temporal_profile': rep_temporal_profile,
    'spectral_profile': rep_spectral_profile,
    'combined_profile': rep_combined_profile,
}


# ============================================================================
# Compute representations
# ============================================================================

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


print()
print("=" * 70)
print("  Computing representations")
print("=" * 70)
print()

rep_vectors = {}  # rep_name -> list of (class, is_seed, vector)

for rep_name, rep_fn in REPRESENTATIONS.items():
    vectors = []
    for class_name, is_seed, spec, waveform in dataset:
        vec = rep_fn(spec)
        vectors.append((class_name, is_seed, vec))
    rep_vectors[rep_name] = vectors
    dim = len(vectors[0][2])
    print(f"  {rep_name}: {dim} dimensions")


# ============================================================================
# Measure discrimination gap for each representation
# ============================================================================

print()
print("=" * 70)
print("  Discrimination Gap Analysis (using seeds only)")
print("=" * 70)
print()

class_names = list(GENERATORS.keys())

results = {}

for rep_name, vectors in rep_vectors.items():
    # Extract seed vectors per class
    seeds_by_class = {c: [] for c in class_names}
    for class_name, is_seed, vec in vectors:
        if is_seed:
            seeds_by_class[class_name].append(vec)

    # Compute within-class similarities (all seed pairs within same class)
    within_sims = []
    for c in class_names:
        vecs = seeds_by_class[c]
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within_sims.append(cosine(vecs[i], vecs[j]))

    # Compute between-class similarities (all seed pairs across classes)
    between_sims = []
    for i, c1 in enumerate(class_names):
        for c2 in class_names[i+1:]:
            for v1 in seeds_by_class[c1]:
                for v2 in seeds_by_class[c2]:
                    between_sims.append(cosine(v1, v2))

    mean_within = np.mean(within_sims)
    mean_between = np.mean(between_sims)
    max_between = np.max(between_sims)
    min_within = np.min(within_sims)

    gap = min_within - max_between
    mean_gap = mean_within - mean_between

    results[rep_name] = {
        'mean_within': mean_within,
        'min_within': min_within,
        'mean_between': mean_between,
        'max_between': max_between,
        'gap': gap,
        'mean_gap': mean_gap,
    }

    status = "POSITIVE" if gap > 0 else "negative"
    print(f"  {rep_name}:")
    print(f"    within-class:  mean={mean_within:.4f}  min={min_within:.4f}")
    print(f"    between-class: mean={mean_between:.4f}  max={max_between:.4f}")
    print(f"    GAP (min_within - max_between): {gap:+.4f}  [{status}]")
    print(f"    mean gap: {mean_gap:+.4f}")
    print()


# ============================================================================
# Seed-and-propagate with each representation
# ============================================================================

print("=" * 70)
print("  Seed-and-Propagate Classification")
print("=" * 70)
print()

for rep_name, vectors in rep_vectors.items():
    # Build concept models (mean profile per class from seeds)
    concept_models = {}
    for class_name, is_seed, vec in vectors:
        if is_seed:
            if class_name not in concept_models:
                concept_models[class_name] = []
            concept_models[class_name].append(vec)

    for c in concept_models:
        concept_models[c] = np.mean(concept_models[c], axis=0)

    # Classify test instances
    correct = 0
    total = 0
    per_class_correct = {c: 0 for c in class_names}
    per_class_total = {c: 0 for c in class_names}
    confusion = {c1: {c2: 0 for c2 in class_names} for c1 in class_names}

    for class_name, is_seed, vec in vectors:
        if is_seed:
            continue

        # Find best matching concept
        best_class = None
        best_sim = -1
        for c, model in concept_models.items():
            sim = cosine(vec, model)
            if sim > best_sim:
                best_sim = sim
                best_class = c

        total += 1
        per_class_total[class_name] += 1
        confusion[class_name][best_class] += 1
        if best_class == class_name:
            correct += 1
            per_class_correct[class_name] += 1

    accuracy = correct / total if total > 0 else 0

    print(f"  {rep_name}: accuracy = {correct}/{total} ({accuracy:.1%})")
    for c in class_names:
        c_acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
        print(f"    {c:>12}: {per_class_correct[c]}/{per_class_total[c]} ({c_acc:.0%})")

    # Print confusion matrix if accuracy < 100%
    if accuracy < 1.0:
        print(f"    Confusion matrix (rows=true, cols=predicted):")
        print(f"    {'':>12}", end='')
        for c in class_names:
            print(f"  {c[:5]:>5}", end='')
        print()
        for c1 in class_names:
            print(f"    {c1:>12}", end='')
            for c2 in class_names:
                print(f"  {confusion[c1][c2]:5d}", end='')
            print()
    print()


# ============================================================================
# Summary table
# ============================================================================

print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print()
print(f"  {'Representation':>20}  {'Op type':>12}  {'Gap':>8}  {'Accuracy':>8}  {'Verdict':>10}")
print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}")

op_types = {
    'raw_pixels': 'neither',
    'edge_detection': 'sharpening',
    'spectral_peaks': 'sharpening',
    'temporal_profile': 'averaging',
    'spectral_profile': 'avg(wrong)',
    'combined_profile': 'averaging',
}

for rep_name in REPRESENTATIONS:
    gap = results[rep_name]['gap']
    # Recompute accuracy
    concept_models = {}
    for class_name, is_seed, vec in rep_vectors[rep_name]:
        if is_seed:
            if class_name not in concept_models:
                concept_models[class_name] = []
            concept_models[class_name].append(vec)
    for c in concept_models:
        concept_models[c] = np.mean(concept_models[c], axis=0)

    correct = 0
    total = 0
    for class_name, is_seed, vec in rep_vectors[rep_name]:
        if is_seed:
            continue
        best_class = max(concept_models.keys(),
                         key=lambda c: cosine(vec, concept_models[c]))
        total += 1
        if best_class == class_name:
            correct += 1
    acc = correct / total

    op = op_types[rep_name]
    verdict = "PASS" if gap > 0 else "FAIL"
    print(f"  {rep_name:>20}  {op:>12}  {gap:+8.4f}  {acc:8.1%}  {verdict:>10}")

print()

# Hypothesis evaluation
print("=" * 70)
print("  HYPOTHESIS EVALUATION")
print("=" * 70)
print()

# H1: Averaging outperforms sharpening in low-data high-noise
avg_gaps = [results[r]['gap'] for r in ['temporal_profile', 'combined_profile']]
sharp_gaps = [results[r]['gap'] for r in ['edge_detection', 'spectral_peaks']]
h1 = all(a > s for a in avg_gaps for s in sharp_gaps)
print(f"  H1 (averaging > sharpening in gap): {h1}")
print(f"      averaging gaps:  {[f'{g:+.4f}' for g in avg_gaps]}")
print(f"      sharpening gaps: {[f'{g:+.4f}' for g in sharp_gaps]}")
print()

# H2: Gap diagnostic predicts accuracy
pos_gap_reps = [r for r in results if results[r]['gap'] > 0]
neg_gap_reps = [r for r in results if results[r]['gap'] <= 0]
print(f"  H2 (gap diagnostic predicts accuracy):")
for r in pos_gap_reps:
    concept_models = {}
    for cn, is_s, v in rep_vectors[r]:
        if is_s:
            concept_models.setdefault(cn, []).append(v)
    for c in concept_models:
        concept_models[c] = np.mean(concept_models[c], axis=0)
    correct = sum(1 for cn, is_s, v in rep_vectors[r]
                  if not is_s and max(concept_models, key=lambda c: cosine(v, concept_models[c])) == cn)
    total = sum(1 for _, is_s, _ in rep_vectors[r] if not is_s)
    print(f"    gap>0: {r} -> accuracy {correct}/{total} ({correct/total:.0%})")
for r in neg_gap_reps:
    concept_models = {}
    for cn, is_s, v in rep_vectors[r]:
        if is_s:
            concept_models.setdefault(cn, []).append(v)
    for c in concept_models:
        concept_models[c] = np.mean(concept_models[c], axis=0)
    correct = sum(1 for cn, is_s, v in rep_vectors[r]
                  if not is_s and max(concept_models, key=lambda c: cosine(v, concept_models[c])) == cn)
    total = sum(1 for _, is_s, _ in rep_vectors[r] if not is_s)
    print(f"    gap<=0: {r} -> accuracy {correct}/{total} ({correct/total:.0%})")

# Check: does spectral (wrong axis) underperform temporal (right axis)?
print()
right_gap = results['temporal_profile']['gap']
wrong_gap = results['spectral_profile']['gap']
print(f"  Identity axis test:")
print(f"    temporal (right axis): gap = {right_gap:+.4f}")
print(f"    spectral (wrong axis): gap = {wrong_gap:+.4f}")
print(f"    Right axis wins: {right_gap > wrong_gap}")

print()
print("Done.")
