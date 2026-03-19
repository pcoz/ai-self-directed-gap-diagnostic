"""
Heat map: discrimination gap across projection axes and SNR levels.

Visualises WHERE the marginalisation approach works and where it breaks down.
Each cell = gap value for a specific (axis configuration, SNR) combination.
Colour = green (positive gap, method works) to red (negative, fails).

Also: accuracy heat map alongside gap heat map — do they correlate?
"""

import numpy as np
from scipy import signal as sig
import os

np.random.seed(42)

SAMPLE_RATE = 100
DURATION = 20.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES)
N_SEEDS = 3
N_TEST = 30


def add_noise(w, snr_db):
    sp = np.mean(w**2)
    noise_power = sp / (10**(snr_db / 10))
    return w + np.random.randn(len(w)) * np.sqrt(noise_power)


def make_spectrogram(waveform, nperseg=64, noverlap=48):
    f, t, Sxx = sig.spectrogram(waveform, fs=SAMPLE_RATE,
                                 nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    Sxx_log = np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-10)
    return Sxx_norm


def gen_tectonic():
    w = np.zeros(N_SAMPLES)
    pt = 3.0 + np.random.randn()*0.4
    w += np.exp(-np.abs(T-pt)/0.5)*(T>=pt) * np.sin(2*np.pi*(8+np.random.randn()*2)*T)*0.5
    st = pt + 4.0 + np.random.randn()*0.2
    w += np.exp(-(T-st)/2.0)*(T>=st) * np.sin(2*np.pi*(3+np.random.randn())*T)*1.0
    w += np.exp(-(T-st-1)/5.0)*(T>=st+1) * np.sin(2*np.pi*1.5*T)*0.3
    return w

def gen_volcanic():
    w = np.zeros(N_SAMPLES)
    onset = 2.0 + np.random.randn()*0.4
    env = np.clip((T-onset)/3.0, 0, 1) * np.exp(-np.maximum(0, T-onset-8)/3.0)
    f0 = 2.0 + np.random.randn()*0.3
    w += env*(np.sin(2*np.pi*f0*T)*0.8 + np.sin(2*np.pi*2*f0*T)*0.4 + np.sin(2*np.pi*3*f0*T)*0.2)
    w *= (1 + 0.3*np.sin(2*np.pi*0.3*T))
    return w

def gen_explosion():
    w = np.zeros(N_SAMPLES)
    onset = 2.0 + np.random.randn()*0.2
    env = np.exp(-(T-onset)/0.8) * (T>=onset)
    for freq in [5,10,15,20,25]:
        w += env * np.sin(2*np.pi*freq*T + np.random.uniform(0,2*np.pi)) / (1+freq/10)
    return w*0.8

def gen_icequake():
    w = np.zeros(N_SAMPLES)
    onset = 3.0 + np.random.randn()*0.4
    env = np.exp(-((T-onset)/0.15)**2) * (T>=onset-0.1)
    w += env * np.sin(2*np.pi*(15+np.random.randn()*3)*T)
    w += env * np.sin(2*np.pi*(25+np.random.randn()*5)*T)*0.5
    return w

def gen_rockfall():
    w = np.zeros(N_SAMPLES)
    onset = 1.5 + np.random.randn()*0.4
    env = np.zeros(N_SAMPLES)
    for _ in range(np.random.randint(8,15)):
        env += np.random.uniform(0.3,1.0) * np.exp(-((T-onset-np.random.uniform(0,10))/np.random.uniform(0.3,1.5))**2)
    for freq in np.random.uniform(2, 30, size=10):
        w += env * np.sin(2*np.pi*freq*T + np.random.uniform(0,2*np.pi))*0.2
    return w


GENERATORS = {'tectonic': gen_tectonic, 'volcanic': gen_volcanic,
              'explosion': gen_explosion, 'icequake': gen_icequake, 'rockfall': gen_rockfall}


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10)


def profile_axis(spec, axis, n_bins):
    """Marginal along axis. axis=0: temporal profile; axis=1: spectral profile."""
    n = spec.shape[1-axis]
    edges = np.linspace(0, n, n_bins+1).astype(int)
    p = np.zeros(n_bins)
    for b in range(n_bins):
        if axis == 0:
            s = spec[:, edges[b]:edges[b+1]]
        else:
            s = spec[edges[b]:edges[b+1], :]
        if s.size > 0:
            p[b] = np.mean(s)
    return p


def run_trial(snr, rep_fn):
    """Generate data, compute gap and accuracy for one (SNR, representation) config."""
    seeds_by_class = {c: [] for c in GENERATORS}
    test = []

    for cname, gen in GENERATORS.items():
        for i in range(N_SEEDS + N_TEST):
            w = gen()
            w = add_noise(w, snr_db=snr)
            spec = make_spectrogram(w)
            vec = rep_fn(spec)
            if i < N_SEEDS:
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

    gap = min(within) - max(between) if within and between else -1
    mean_gap = np.mean(within) - np.mean(between) if within and between else -1

    # Accuracy
    models = {c: np.mean(vecs, axis=0) for c, vecs in seeds_by_class.items()}
    correct = sum(1 for tc, v in test if max(models, key=lambda c: cosine(v, models[c])) == tc)
    acc = correct / len(test)

    return gap, mean_gap, acc


# ================================================================
# Build the heat maps
# ================================================================

# Axes: SNR levels x representation configurations
snr_levels = [0, 2, 5, 8, 10, 15, 20, 30]

from scipy.ndimage import sobel as sobel_filter

rep_configs = [
    ("raw_pixels", lambda s: s.flatten()),
    ("edges(sharp)", lambda s: np.sqrt(sobel_filter(s,1)**2 + sobel_filter(s,0)**2).flatten()),
    ("temporal_16", lambda s: profile_axis(s, 0, 16)),
    ("temporal_32", lambda s: profile_axis(s, 0, 32)),
    ("temporal_64", lambda s: profile_axis(s, 0, 64)),
    ("spectral_16", lambda s: profile_axis(s, 1, 16)),
    ("spectral_32", lambda s: profile_axis(s, 1, 32)),
    ("comb_32+16", lambda s: np.concatenate([profile_axis(s,0,32), profile_axis(s,1,16)])),
    ("comb_32+32", lambda s: np.concatenate([profile_axis(s,0,32), profile_axis(s,1,32)])),
]

n_snr = len(snr_levels)
n_rep = len(rep_configs)

gap_matrix = np.zeros((n_rep, n_snr))
mgap_matrix = np.zeros((n_rep, n_snr))
acc_matrix = np.zeros((n_rep, n_snr))

print("Computing heat map data...")
print(f"  {n_rep} representations x {n_snr} SNR levels = {n_rep * n_snr} trials")
print()

for j, snr in enumerate(snr_levels):
    for i, (rname, rfn) in enumerate(rep_configs):
        gap, mgap, acc = run_trial(snr, rfn)
        gap_matrix[i, j] = gap
        mgap_matrix[i, j] = mgap
        acc_matrix[i, j] = acc
        print(f"  SNR={snr:2d}  {rname:>15}: gap={gap:+.4f}  mean_gap={mgap:+.4f}  acc={acc:.0%}")
    print()


# ================================================================
# Print heat maps as ASCII tables
# ================================================================

rep_names = [name for name, _ in rep_configs]

def print_heatmap(title, matrix, fmt="+.3f"):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    print(f"  {'':>15}", end='')
    for snr in snr_levels:
        print(f"  {snr:>6}dB", end='')
    print()
    print(f"  {'':>15}", end='')
    for _ in snr_levels:
        print(f"  {'-'*8}", end='')
    print()

    for i, name in enumerate(rep_names):
        print(f"  {name:>15}", end='')
        for j in range(n_snr):
            val = matrix[i, j]
            if fmt == "pct":
                print(f"  {val:7.0%} ", end='')
            else:
                # Colour indicator
                if val > 0:
                    mark = "+"
                elif val > -0.01:
                    mark = "~"
                else:
                    mark = "-"
                print(f"  {val:+7.4f}{mark}", end='')
        print()
    print()


print()
print_heatmap("DISCRIMINATION GAP (min_within - max_between)", gap_matrix)
print_heatmap("MEAN GAP (mean_within - mean_between)", mgap_matrix)

print("=" * 70)
print("  ACCURACY HEAT MAP")
print("=" * 70)
print()
print(f"  {'':>15}", end='')
for snr in snr_levels:
    print(f"  {snr:>6}dB", end='')
print()
print(f"  {'':>15}", end='')
for _ in snr_levels:
    print(f"  {'-'*8}", end='')
print()
for i, name in enumerate(rep_names):
    print(f"  {name:>15}", end='')
    for j in range(n_snr):
        val = acc_matrix[i, j]
        if val >= 0.90:
            mark = " **"
        elif val >= 0.80:
            mark = " * "
        elif val >= 0.70:
            mark = "   "
        else:
            mark = " ! "
        print(f"  {val:5.0%}{mark}", end='')
    print()

# ================================================================
# Key findings
# ================================================================
print()
print("=" * 70)
print("  KEY FINDINGS FROM HEAT MAP")
print("=" * 70)

# Where does temporal profile beat raw pixels?
wins_temporal = []
wins_raw = []
for j, snr in enumerate(snr_levels):
    t_acc = acc_matrix[rep_names.index("temporal_32"), j]
    r_acc = acc_matrix[rep_names.index("raw_pixels"), j]
    if t_acc > r_acc:
        wins_temporal.append(snr)
    elif r_acc > t_acc:
        wins_raw.append(snr)

print(f"\n  Temporal profile wins at SNR: {wins_temporal}")
print(f"  Raw pixels wins at SNR: {wins_raw}")

# Where does sharpening perform worst?
edge_idx = rep_names.index("edges(sharp)")
edge_accs = acc_matrix[edge_idx, :]
print(f"\n  Edge detection accuracy range: {edge_accs.min():.0%} - {edge_accs.max():.0%}")
print(f"  Always worst or near-worst: {all(edge_accs[j] <= acc_matrix[:, j].max() - 0.05 for j in range(n_snr))}")

# Best overall representation
mean_accs = acc_matrix.mean(axis=1)
best_idx = np.argmax(mean_accs)
print(f"\n  Best average accuracy across all SNRs: {rep_names[best_idx]} ({mean_accs[best_idx]:.1%})")

# Correlation between gap and accuracy
from numpy import corrcoef
all_gaps = gap_matrix.flatten()
all_accs = acc_matrix.flatten()
r = corrcoef(all_gaps, all_accs)[0, 1]
print(f"\n  Correlation between gap and accuracy: r = {r:.3f}")

all_mgaps = mgap_matrix.flatten()
r2 = corrcoef(all_mgaps, all_accs)[0, 1]
print(f"  Correlation between mean_gap and accuracy: r = {r2:.3f}")

print()
print("Done.")
