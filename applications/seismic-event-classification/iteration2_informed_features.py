"""
ITERATION 2: Domain-informed feature design guided by confusion analysis.

Iteration 1 showed:
  - Spectral profile (59%) is the best marginalisation axis
  - Temporal profile fails for variable-length events
  - explosion/icequake/noise_burst are confused (all short, broadband)
  - tect_deep/volc_tremor are confused (both long, low-freq)

The confused classes differ in TEMPORAL SHAPE at event level:
  - Duration: icequake < explosion < noise_burst
  - Onset sharpness: explosion > icequake > noise_burst
  - Decay rate: icequake > explosion > noise_burst
  - Temporal regularity: tremor regular, deep tectonic has P+S structure

Strategy: Add scalar temporal descriptors to the spectral profile.
These are EVENT-LEVEL marginalisations (single numbers that summarise
the temporal dimension) rather than BIN-LEVEL profiles.

This tests the refined hypothesis: marginalise at the APPROPRIATE SCALE
for each axis. Fine-grained for axes with consistent resolution (spectral),
coarse for axes with variable resolution (temporal).
"""

import numpy as np
from scipy import signal as sig
from scipy.ndimage import sobel

np.random.seed(42)

SAMPLE_RATE = 100

def coloured_noise(n, slope=1.0):
    freqs = np.fft.rfftfreq(n, d=1/SAMPLE_RATE)
    freqs[0] = 1
    spectrum = np.sqrt(1.0/(freqs**slope)) * np.exp(1j*np.random.uniform(0,2*np.pi,len(freqs)))
    noise = np.fft.irfft(spectrum, n=n)
    return noise / (np.std(noise) + 1e-10)

def make_spec(w, nperseg=64, noverlap=48):
    f, t, S = sig.spectrogram(w, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    S_log = np.log10(S + 1e-10)
    lo, hi = np.percentile(S_log, [2, 98])
    return np.clip((S_log - lo)/(hi - lo + 1e-10), 0, 1)

def add_noise(w, snr):
    s = np.mean(w**2)
    n = coloured_noise(len(w))
    return w + n * np.sqrt(s / (10**(snr/10)))

# --- Generators (same as iteration 1) ---
def gen_tect_shallow():
    dur = np.random.uniform(4, 10); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    w = np.zeros(n); pt = np.random.uniform(0.3,1.0); st = pt+np.random.uniform(1.5,3.0)
    w += np.exp(-np.abs(t-pt)/0.3)*(t>=pt)*np.sin(2*np.pi*(10+np.random.randn()*2)*t)*0.6
    w += np.exp(-(t-st)/1.0)*(t>=st)*np.sin(2*np.pi*(4+np.random.randn())*t)
    w += np.exp(-(t-st-0.5)/2.0)*(t>=st+0.5)*np.sin(2*np.pi*2*t)*0.2
    return w

def gen_tect_deep():
    dur = np.random.uniform(10, 28); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    w = np.zeros(n); pt = np.random.uniform(0.5,2.0); st = pt+np.random.uniform(4,8)
    w += np.exp(-np.abs(t-pt)/0.8)*(t>=pt)*np.sin(2*np.pi*(5+np.random.randn())*t)*0.3
    w += np.exp(-(t-st)/3.0)*(t>=st)*np.sin(2*np.pi*(2+np.random.randn()*0.5)*t)*0.8
    w += np.exp(-(t-st-1)/6.0)*(t>=st+1)*np.sin(2*np.pi*1.0*t)*0.4
    return w

def gen_volc_tremor():
    dur = np.random.uniform(12, 32); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    on = np.random.uniform(0.5,2.0); sus = dur*np.random.uniform(0.5,0.8)
    env = np.clip((t-on)/3,0,1)*np.exp(-np.maximum(0,t-on-sus)/3)
    f0 = np.random.uniform(1.5,3.0)
    w = env*(np.sin(2*np.pi*f0*t)*0.7+np.sin(2*np.pi*2*f0*t)*0.3+np.sin(2*np.pi*3*f0*t)*0.15)
    w *= (1+0.2*np.sin(2*np.pi*0.2*t))
    return w

def gen_volc_lp():
    dur = np.random.uniform(2, 7); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    on = np.random.uniform(0.3,1.0); f0 = np.random.uniform(1.0,2.5)
    env = np.exp(-((t-on)/0.8)**2)+0.3*np.exp(-((t-on-1.5)/1.2)**2)
    return env*(np.sin(2*np.pi*f0*t)*0.8+np.sin(2*np.pi*2*f0*t)*0.3)

def gen_explosion():
    dur = np.random.uniform(1.5, 5); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    on = np.random.uniform(0.2,0.8); env = np.exp(-(t-on)/0.5)*(t>=on)
    w = np.zeros(n)
    for f in np.random.uniform(5,35,8): w += env*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))/(1+f/15)
    return w*0.7

def gen_icequake():
    dur = np.random.uniform(0.8, 2.5); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    on = np.random.uniform(0.1,0.4); env = np.exp(-((t-on)/0.1)**2)
    return env*(np.sin(2*np.pi*(15+np.random.randn()*3)*t)+0.5*np.sin(2*np.pi*(25+np.random.randn()*5)*t))

def gen_noise_burst():
    dur = np.random.uniform(1, 6); n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n)
    on = np.random.uniform(0.2,1.0); env = np.exp(-(t-on)/np.random.uniform(0.5,2))*(t>=on)
    w = np.zeros(n)
    for f in np.random.uniform(3,30,12): w += env*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))*np.random.uniform(0.1,0.5)
    w *= (1+0.4*np.random.randn(n).cumsum()/np.sqrt(n))
    return w

GENERATORS = {'tect_shallow': gen_tect_shallow, 'tect_deep': gen_tect_deep,
              'volc_tremor': gen_volc_tremor, 'volc_lp': gen_volc_lp,
              'explosion': gen_explosion, 'icequake': gen_icequake, 'noise_burst': gen_noise_burst}


# ============================================================================
# Event-level temporal descriptors
# ============================================================================

def temporal_descriptors(waveform, spec):
    """
    Coarse temporal features at EVENT level:
      0: log_duration     — log10(duration in seconds)
      1: onset_sharpness  — energy in first 10% / energy in first 50%
      2: decay_rate       — energy in last 30% / total energy
      3: peak_position    — position of peak energy (0=start, 1=end)
      4: temporal_spread  — std of energy distribution over time
      5: energy_kurtosis  — peakedness of temporal energy envelope
      6: n_energy_peaks   — number of peaks in temporal energy envelope
    """
    energy = np.sum(spec, axis=0)  # energy per time step
    if energy.sum() < 1e-10:
        return np.zeros(7)

    n = len(energy)
    dur = len(waveform) / SAMPLE_RATE

    # Normalise energy to distribution
    e_norm = energy / (energy.sum() + 1e-10)

    # 0: log duration
    log_dur = np.log10(dur + 0.1)

    # 1: onset sharpness
    tenth = max(1, n // 10)
    half = max(1, n // 2)
    onset = energy[:tenth].sum() / (energy[:half].sum() + 1e-10)

    # 2: decay rate
    last30 = max(1, int(n * 0.7))
    decay = energy[last30:].sum() / (energy.sum() + 1e-10)

    # 3: peak position (normalised)
    peak_pos = np.argmax(energy) / (n - 1 + 1e-10)

    # 4: temporal spread
    positions = np.arange(n) / (n - 1 + 1e-10)
    mean_pos = np.sum(positions * e_norm)
    spread = np.sqrt(np.sum((positions - mean_pos)**2 * e_norm))

    # 5: kurtosis of energy envelope
    std_e = np.std(energy) + 1e-10
    kurtosis = np.mean(((energy - np.mean(energy)) / std_e)**4) - 3

    # 6: number of energy peaks
    peaks, _ = sig.find_peaks(energy, height=0.3*energy.max(), distance=max(1, n//10))
    n_peaks = min(len(peaks), 5) / 5.0  # normalise to [0, 1]

    return np.array([log_dur, onset, decay, peak_pos, spread, kurtosis, n_peaks])


def spectral_profile(spec, n_bins=16):
    n = spec.shape[0]
    edges = np.linspace(0, n, n_bins+1).astype(int)
    p = np.zeros(n_bins)
    for b in range(n_bins):
        s = spec[edges[b]:edges[b+1], :]
        if s.size > 0: p[b] = np.mean(s)
    return p


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10)


# ============================================================================
# Generate dataset
# ============================================================================
N_SEEDS = 3
N_TEST = 25
SNR = 5

print("=" * 70)
print("  Iteration 2: Spectral Profile + Temporal Descriptors")
print("=" * 70)
print()

dataset = []
for cname, gen in GENERATORS.items():
    for i in range(N_SEEDS + N_TEST):
        w = gen()
        w = add_noise(w, SNR)
        spec = make_spec(w)
        td = temporal_descriptors(w, spec)
        sp = spectral_profile(spec, 16)
        is_seed = i < N_SEEDS
        dataset.append((cname, is_seed, spec, w, td, sp))

print(f"  {len(dataset)} instances, 7 classes, SNR={SNR}dB")

# Show temporal descriptor distributions per class
print()
td_names = ['log_dur', 'onset', 'decay', 'peak_pos', 'spread', 'kurtosis', 'n_peaks']
print("  Temporal descriptor means per class (seeds only):")
print(f"  {'class':>15}", end='')
for n in td_names:
    print(f"  {n[:7]:>7}", end='')
print()

for cname in GENERATORS:
    tds = [td for cn, is_s, _, _, td, _ in dataset if cn == cname and is_s]
    means = np.mean(tds, axis=0)
    print(f"  {cname:>15}", end='')
    for v in means:
        print(f"  {v:7.3f}", end='')
    print()


# ============================================================================
# Test representations
# ============================================================================
print()
print("=" * 70)
print("  Representation Comparison")
print("=" * 70)
print()

from scipy.ndimage import zoom

REPS = {
    'raw_64': lambda s, w, td, sp: zoom(s, (1, 64/s.shape[1]), order=1).flatten(),
    'spectral_16': lambda s, w, td, sp: sp,
    'temporal_desc': lambda s, w, td, sp: td,
    'spec+temp_desc': lambda s, w, td, sp: np.concatenate([sp, td]),
    'spec+temp_desc_w': lambda s, w, td, sp: np.concatenate([sp, td * 2.0]),  # weight temporal
    'spec32+temp': lambda s, w, td, sp: np.concatenate([spectral_profile(s, 32), td]),
    'spec16+temp+spec32': lambda s, w, td, sp: np.concatenate([sp, td, spectral_profile(s, 32)]),
}

for rname, rfn in REPS.items():
    seeds_by_class = {c: [] for c in GENERATORS}
    test = []

    for cname, is_seed, spec, w, td, sp in dataset:
        vec = rfn(spec, w, td, sp)
        if is_seed:
            seeds_by_class[cname].append(vec)
        else:
            test.append((cname, vec))

    # Gap
    within, between = [], []
    classes = list(GENERATORS.keys())
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

    # Accuracy + confusion
    models = {c: np.mean(vecs, axis=0) for c, vecs in seeds_by_class.items()}
    correct = 0
    confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for tc, v in test:
        pred = max(models, key=lambda c: cosine(v, models[c]))
        confusion[tc][pred] += 1
        if pred == tc: correct += 1
    acc = correct / len(test)

    dim = len(rfn(dataset[0][2], dataset[0][3], dataset[0][4], dataset[0][5]))
    print(f"  {rname:>20} ({dim:3d}d): gap={gap:+.4f}  mg={mean_gap:+.4f}  acc={acc:.0%}")

    # Print confusion for key methods
    if rname in ('raw_64', 'spec+temp_desc', 'spectral_16'):
        print(f"    Confusion (rows=true, cols=pred):")
        abbr = [c[:5] for c in classes]
        print(f"    {'':>15}", end='')
        for a in abbr: print(f" {a:>5}", end='')
        print()
        for c1 in classes:
            print(f"    {c1:>15}", end='')
            for c2 in classes:
                print(f" {confusion[c1][c2]:5d}", end='')
            print()
        print()


# ============================================================================
# Per-class accuracy comparison: which classes improved?
# ============================================================================
print()
print("=" * 70)
print("  Per-Class Improvement: raw_64 vs spec+temp_desc")
print("=" * 70)
print()

for rname in ('raw_64', 'spec+temp_desc'):
    rfn = REPS[rname]
    seeds_by_class = {c: [] for c in GENERATORS}
    test = []
    for cname, is_seed, spec, w, td, sp in dataset:
        vec = rfn(spec, w, td, sp)
        if is_seed: seeds_by_class[cname].append(vec)
        else: test.append((cname, vec))
    models = {c: np.mean(vecs, axis=0) for c, vecs in seeds_by_class.items()}

    print(f"  {rname}:")
    for c in sorted(GENERATORS.keys()):
        c_test = [(tc, v) for tc, v in test if tc == c]
        c_correct = sum(1 for tc, v in c_test if max(models, key=lambda x: cosine(v, models[x])) == tc)
        print(f"    {c:>15}: {c_correct}/{len(c_test)} ({c_correct/len(c_test):.0%})")
    print()


print("Done.")
