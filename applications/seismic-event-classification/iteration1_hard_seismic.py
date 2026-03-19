"""
ITERATION 1: Make the seismic problem realistic and hard.

Changes from v1/v2:
  1. Variable-duration events (1s to 30s) — spectrograms have DIFFERENT SIZES
  2. Coloured noise (1/f spectrum, not white Gaussian) — structured noise
  3. More confusable classes (added subtypes that share spectral content)
  4. Variable onset times and amplitudes
  5. Spectrograms must be NORMALISED to compare — this is where profiles should win

New classes (7, more confusable):
  1. tectonic_shallow  — sharp P, strong S, short coda
  2. tectonic_deep     — weak P, delayed S, long coda (CONFUSABLE with shallow)
  3. volcanic_tremor   — harmonic, sustained
  4. volcanic_lp       — low-period event, single pulse (CONFUSABLE with tremor)
  5. explosion         — sharp, broadband, no S
  6. icequake          — very short, high freq
  7. noise_burst       — just a loud noise transient (CONFUSABLE with explosion)

The methodology must:
  1. Compute profiles for variable-size spectrograms
  2. Use gap diagnostic to find best axis/config
  3. Classify from 3 seeds per class
  4. Beat raw pixel comparison (which now must normalise variable-size spectrograms)
"""

import numpy as np
from scipy import signal as sig
from scipy.ndimage import sobel

np.random.seed(42)

SAMPLE_RATE = 100
T_MAX = 40.0  # max possible duration


def coloured_noise(n_samples, slope=1.0):
    """Generate 1/f^slope coloured noise."""
    freqs = np.fft.rfftfreq(n_samples, d=1/SAMPLE_RATE)
    freqs[0] = 1  # avoid div by zero
    power = 1.0 / (freqs ** slope)
    phases = np.random.uniform(0, 2*np.pi, len(freqs))
    spectrum = np.sqrt(power) * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    noise = noise / (np.std(noise) + 1e-10)
    return noise


def make_spectrogram(waveform, nperseg=64, noverlap=48):
    f, t, Sxx = sig.spectrogram(waveform, fs=SAMPLE_RATE,
                                 nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    Sxx_log = np.log10(Sxx + 1e-10)
    vmin, vmax = np.percentile(Sxx_log, [2, 98])
    Sxx_norm = np.clip((Sxx_log - vmin) / (vmax - vmin + 1e-10), 0, 1)
    return Sxx_norm


def add_coloured_noise(waveform, snr_db):
    sp = np.mean(waveform**2)
    noise = coloured_noise(len(waveform), slope=1.0)
    noise_power = sp / (10**(snr_db/10))
    return waveform + noise * np.sqrt(noise_power)


# ============================================================================
# Generators — variable duration, more confusable
# ============================================================================

def gen_tectonic_shallow():
    """2-8s event. Sharp P, strong S, short coda."""
    duration = np.random.uniform(4, 10)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    w = np.zeros(n)
    p_time = np.random.uniform(0.3, 1.0)
    s_time = p_time + np.random.uniform(1.5, 3.0)
    w += np.exp(-np.abs(t-p_time)/0.3)*(t>=p_time) * np.sin(2*np.pi*(10+np.random.randn()*2)*t)*0.6
    w += np.exp(-(t-s_time)/1.0)*(t>=s_time) * np.sin(2*np.pi*(4+np.random.randn())*t)*1.0
    w += np.exp(-(t-s_time-0.5)/2.0)*(t>=s_time+0.5) * np.sin(2*np.pi*2*t)*0.2
    return w

def gen_tectonic_deep():
    """8-25s event. Weak P, delayed S, long coda. CONFUSABLE with shallow."""
    duration = np.random.uniform(10, 28)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    w = np.zeros(n)
    p_time = np.random.uniform(0.5, 2.0)
    s_time = p_time + np.random.uniform(4.0, 8.0)
    w += np.exp(-np.abs(t-p_time)/0.8)*(t>=p_time) * np.sin(2*np.pi*(5+np.random.randn())*t)*0.3
    w += np.exp(-(t-s_time)/3.0)*(t>=s_time) * np.sin(2*np.pi*(2+np.random.randn()*0.5)*t)*0.8
    w += np.exp(-(t-s_time-1)/6.0)*(t>=s_time+1) * np.sin(2*np.pi*1.0*t)*0.4
    return w

def gen_volcanic_tremor():
    """10-30s sustained harmonic tremor."""
    duration = np.random.uniform(12, 32)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    onset = np.random.uniform(0.5, 2.0)
    sustain = duration * np.random.uniform(0.5, 0.8)
    env = np.clip((t-onset)/3.0, 0, 1) * np.exp(-np.maximum(0, t-onset-sustain)/3.0)
    f0 = np.random.uniform(1.5, 3.0)
    w = env*(np.sin(2*np.pi*f0*t)*0.7 + np.sin(2*np.pi*2*f0*t)*0.3 + np.sin(2*np.pi*3*f0*t)*0.15)
    w *= (1 + 0.2*np.sin(2*np.pi*0.2*t))
    return w

def gen_volcanic_lp():
    """2-6s single low-period pulse. CONFUSABLE with tremor."""
    duration = np.random.uniform(2, 7)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    onset = np.random.uniform(0.3, 1.0)
    f0 = np.random.uniform(1.0, 2.5)
    env = np.exp(-((t-onset)/0.8)**2) + 0.3*np.exp(-((t-onset-1.5)/1.2)**2)
    w = env*(np.sin(2*np.pi*f0*t)*0.8 + np.sin(2*np.pi*2*f0*t)*0.3)
    return w

def gen_explosion():
    """1-4s sharp blast, broadband, no S-wave."""
    duration = np.random.uniform(1.5, 5)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    onset = np.random.uniform(0.2, 0.8)
    env = np.exp(-(t-onset)/0.5)*(t>=onset)
    w = np.zeros(n)
    for freq in np.random.uniform(5, 35, size=8):
        w += env * np.sin(2*np.pi*freq*t + np.random.uniform(0, 2*np.pi)) / (1+freq/15)
    return w * 0.7

def gen_icequake():
    """0.5-2s extremely short, high frequency."""
    duration = np.random.uniform(0.8, 2.5)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    onset = np.random.uniform(0.1, 0.4)
    env = np.exp(-((t-onset)/0.1)**2)
    f1 = np.random.uniform(12, 20)
    f2 = np.random.uniform(22, 35)
    w = env*(np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t))
    return w

def gen_noise_burst():
    """1-5s broadband transient. CONFUSABLE with explosion."""
    duration = np.random.uniform(1, 6)
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n)
    onset = np.random.uniform(0.2, 1.0)
    env = np.exp(-(t-onset)/np.random.uniform(0.5, 2.0))*(t>=onset)
    # Broadband noise-like content (more irregular than explosion)
    w = np.zeros(n)
    for freq in np.random.uniform(3, 30, size=12):
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.1, 0.5)
        w += env * np.sin(2*np.pi*freq*t + phase) * amp
    # Add random amplitude modulation (distinguishes from clean explosion)
    w *= (1 + 0.4*np.random.randn(n).cumsum()/np.sqrt(n))
    return w


GENERATORS = {
    'tect_shallow': gen_tectonic_shallow,
    'tect_deep': gen_tectonic_deep,
    'volc_tremor': gen_volcanic_tremor,
    'volc_lp': gen_volcanic_lp,
    'explosion': gen_explosion,
    'icequake': gen_icequake,
    'noise_burst': gen_noise_burst,
}


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10)


def profile_1d(arr, n_bins):
    """Profile of a 1D or 2D array along axis 0 (columns)."""
    if arr.ndim == 1:
        n = len(arr)
    else:
        n = arr.shape[1]
    edges = np.linspace(0, n, n_bins+1).astype(int)
    p = np.zeros(n_bins)
    for b in range(n_bins):
        if arr.ndim == 1:
            s = arr[edges[b]:edges[b+1]]
        else:
            s = arr[:, edges[b]:edges[b+1]]
        if s.size > 0:
            p[b] = np.mean(s)
    return p


def profile_rows(arr, n_bins):
    """Profile along axis 1 (rows)."""
    n = arr.shape[0]
    edges = np.linspace(0, n, n_bins+1).astype(int)
    p = np.zeros(n_bins)
    for b in range(n_bins):
        s = arr[edges[b]:edges[b+1], :]
        if s.size > 0:
            p[b] = np.mean(s)
    return p


def normalise_spec(spec, target_cols=64):
    """Resize spectrogram to fixed number of time columns (for raw pixel comparison)."""
    from scipy.ndimage import zoom
    if spec.shape[1] == target_cols:
        return spec
    factor = target_cols / spec.shape[1]
    return zoom(spec, (1, factor), order=1)


# ============================================================================
# Generate dataset
# ============================================================================
N_SEEDS = 3
N_TEST = 25
SNR_DB = 5  # moderate noise

print("=" * 70)
print("  Iteration 1: Hard Seismic Problem")
print("=" * 70)
print(f"  7 classes (including confusable pairs), {N_SEEDS} seeds + {N_TEST} test")
print(f"  Variable duration, coloured noise, SNR={SNR_DB}dB")
print()

dataset = []
for cname, gen in GENERATORS.items():
    durations = []
    for i in range(N_SEEDS + N_TEST):
        w = gen()
        w = add_coloured_noise(w, snr_db=SNR_DB)
        spec = make_spectrogram(w)
        is_seed = i < N_SEEDS
        dataset.append((cname, is_seed, spec, len(w)))
        durations.append(len(w) / SAMPLE_RATE)
    print(f"  {cname:>15}: duration {min(durations):.1f}-{max(durations):.1f}s, "
          f"spec shapes vary ({dataset[-1][2].shape})")

# Show size variation
shapes = set(s.shape for _, _, s, _ in dataset)
print(f"\n  Unique spectrogram shapes: {len(shapes)}")
print(f"  Shape range: {min(s[1] for s in shapes)}-{max(s[1] for s in shapes)} time bins")


# ============================================================================
# Representations
# ============================================================================

def rep_raw_normalised(spec, target=64):
    """Raw pixels after resizing to common width."""
    return normalise_spec(spec, target).flatten()

def rep_edges_normalised(spec, target=64):
    """Edge detection on normalised spectrogram."""
    norm = normalise_spec(spec, target)
    gx = sobel(norm, axis=1)
    gy = sobel(norm, axis=0)
    return np.sqrt(gx**2 + gy**2).flatten()

def rep_temporal(spec, n_bins=32):
    """Temporal profile — marginal along time axis."""
    return profile_1d(spec, n_bins)

def rep_spectral(spec, n_bins=16):
    """Spectral profile — marginal along frequency axis."""
    return profile_rows(spec, n_bins)

def rep_combined(spec, t_bins=32, f_bins=16):
    """Combined temporal + spectral profiles."""
    return np.concatenate([profile_1d(spec, t_bins), profile_rows(spec, f_bins)])

def rep_waveform_envelope(spec, n_bins=32):
    """Temporal energy envelope — sum of all frequency bins per time step."""
    energy = np.sum(spec, axis=0)  # total energy per time step
    return profile_1d(energy, n_bins)


REPS = {
    'raw_64': lambda s: rep_raw_normalised(s, 64),
    'raw_128': lambda s: rep_raw_normalised(s, 128),
    'edges_64': lambda s: rep_edges_normalised(s, 64),
    'temporal_32': lambda s: rep_temporal(s, 32),
    'temporal_64': lambda s: rep_temporal(s, 64),
    'spectral_16': lambda s: rep_spectral(s, 16),
    'spectral_32': lambda s: rep_spectral(s, 32),
    'combined_32+16': lambda s: rep_combined(s, 32, 16),
    'combined_64+16': lambda s: rep_combined(s, 64, 16),
    'envelope_32': lambda s: rep_waveform_envelope(s, 32),
}


# ============================================================================
# Compute gap and accuracy for each representation
# ============================================================================
print()
print("=" * 70)
print("  Gap Diagnostic + Accuracy")
print("=" * 70)
print()

results = {}

for rname, rfn in REPS.items():
    seeds_by_class = {c: [] for c in GENERATORS}
    test = []

    for cname, is_seed, spec, _ in dataset:
        vec = rfn(spec)
        if is_seed:
            seeds_by_class[cname].append(vec)
        else:
            test.append((cname, vec))

    # Gap
    within, between = [], []
    classes = list(seeds_by_class.keys())
    for c in classes:
        vv = seeds_by_class[c]
        for i in range(len(vv)):
            for j in range(i+1, len(vv)):
                within.append(cosine(vv[i], vv[j]))
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            for v1 in seeds_by_class[c1]:
                for v2 in seeds_by_class[c2]:
                    between.append(cosine(v1, v2))

    gap = min(within) - max(between)
    mean_gap = np.mean(within) - np.mean(between)

    # Accuracy
    models = {c: np.mean(vecs, axis=0) for c, vecs in seeds_by_class.items()}
    correct = 0
    total = len(test)
    confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for tc, v in test:
        pred = max(models, key=lambda c: cosine(v, models[c]))
        confusion[tc][pred] += 1
        if pred == tc:
            correct += 1
    acc = correct / total

    results[rname] = {'gap': gap, 'mean_gap': mean_gap, 'acc': acc, 'confusion': confusion}
    dim = len(rfn(dataset[0][2]))

    print(f"  {rname:>15} ({dim:4d}d): gap={gap:+.4f}  mean_gap={mean_gap:+.4f}  acc={acc:.0%}")


# ============================================================================
# Summary table
# ============================================================================
print()
print("=" * 70)
print("  SUMMARY (sorted by accuracy)")
print("=" * 70)
print()
print(f"  {'rep':>15}  {'dim':>5}  {'gap':>8}  {'mean_gap':>8}  {'acc':>6}  {'type':>10}")
print(f"  {'-'*15}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")

op_types = {
    'raw_64': 'raw+resize', 'raw_128': 'raw+resize',
    'edges_64': 'sharpening',
    'temporal_32': 'averaging', 'temporal_64': 'averaging',
    'spectral_16': 'averaging', 'spectral_32': 'averaging',
    'combined_32+16': 'averaging', 'combined_64+16': 'averaging',
    'envelope_32': 'averaging',
}

sorted_reps = sorted(results.items(), key=lambda x: -x[1]['acc'])
for rname, r in sorted_reps:
    dim = len(REPS[rname](dataset[0][2]))
    print(f"  {rname:>15}  {dim:5d}  {r['gap']:+8.4f}  {r['mean_gap']:+8.4f}  "
          f"{r['acc']:6.0%}  {op_types.get(rname, '?'):>10}")

# ============================================================================
# Confusion matrix for best method
# ============================================================================
best_name = sorted_reps[0][0]
best_r = sorted_reps[0][1]
print(f"\n  Best: {best_name} ({best_r['acc']:.0%})")
print(f"\n  Confusion matrix (rows=true, cols=predicted):")

classes = sorted(GENERATORS.keys())
print(f"  {'':>15}", end='')
for c in classes:
    print(f"  {c[:6]:>6}", end='')
print()
for c1 in classes:
    print(f"  {c1:>15}", end='')
    for c2 in classes:
        print(f"  {best_r['confusion'][c1][c2]:6d}", end='')
    print()

# Also show confusion for worst profile method for comparison
worst_profile = None
for rname, r in reversed(sorted_reps):
    if 'temporal' in rname or 'combined' in rname or 'spectral' in rname:
        worst_profile = (rname, r)
        break

if worst_profile:
    wp_name, wp_r = worst_profile
    print(f"\n  Worst profile: {wp_name} ({wp_r['acc']:.0%})")
    print(f"  Confusion matrix:")
    print(f"  {'':>15}", end='')
    for c in classes:
        print(f"  {c[:6]:>6}", end='')
    print()
    for c1 in classes:
        print(f"  {c1:>15}", end='')
        for c2 in classes:
            print(f"  {wp_r['confusion'][c1][c2]:6d}", end='')
        print()

# ============================================================================
# Key question: do profiles beat raw when sizes vary?
# ============================================================================
print()
print("=" * 70)
print("  KEY QUESTION: Does variable size give profiles an advantage?")
print("=" * 70)
print()

raw_acc = max(results[r]['acc'] for r in results if 'raw' in r)
profile_acc = max(results[r]['acc'] for r in results if r not in ('raw_64', 'raw_128', 'edges_64'))
edge_acc = results.get('edges_64', {}).get('acc', 0)

print(f"  Best raw pixel accuracy:  {raw_acc:.0%}")
print(f"  Best profile accuracy:    {profile_acc:.0%}")
print(f"  Edge detection accuracy:  {edge_acc:.0%}")
print()

if profile_acc > raw_acc:
    print(f"  >>> PROFILES WIN by {profile_acc - raw_acc:.0%} <<<")
    print(f"  Variable-size hypothesis CONFIRMED:")
    print(f"  When spectrograms have different sizes, normalisation hurts raw pixels")
    print(f"  but profiles are inherently size-invariant.")
elif profile_acc == raw_acc:
    print(f"  >>> TIE — profiles match raw pixels <<<")
else:
    print(f"  >>> RAW PIXELS STILL WIN by {raw_acc - profile_acc:.0%} <<<")
    print(f"  Variable-size hypothesis NOT YET CONFIRMED at this noise level.")
    print(f"  Will test at lower SNR in next iteration.")

print()
print("Done. See investigation_log.md for next steps.")
