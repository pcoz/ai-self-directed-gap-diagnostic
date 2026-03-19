"""
ITERATION 3: Marginalise the RAW WAVEFORM, not the spectrogram.

Key insight from iteration 2 failure: the spectrogram's time axis has
variable resolution (2-191 bins depending on event duration). Temporal
profiles on the spectrogram are degenerate for short events.

But the raw waveform has a CONSISTENT time axis (samples at 100Hz).
The energy envelope can be directly binned into N equal segments,
giving a duration-normalised temporal profile at any resolution.

Strategy:
  1. Waveform energy envelope → fixed N-bin profile (temporal identity)
  2. Spectrogram spectral profile (frequency identity)
  3. Combine both
  4. Compare against raw pixels baseline (68%)
"""

import numpy as np
from scipy import signal as sig
from scipy.ndimage import zoom

np.random.seed(42)
SAMPLE_RATE = 100


def coloured_noise(n, slope=1.0):
    freqs = np.fft.rfftfreq(n, d=1/SAMPLE_RATE)
    freqs[0] = 1
    spectrum = np.sqrt(1.0/(freqs**slope)) * np.exp(1j*np.random.uniform(0,2*np.pi,len(freqs)))
    noise = np.fft.irfft(spectrum, n=n)
    return noise / (np.std(noise)+1e-10)

def add_noise(w, snr):
    s = np.mean(w**2)
    return w + coloured_noise(len(w)) * np.sqrt(s/(10**(snr/10)))

def make_spec(w, nperseg=64, noverlap=48):
    f,t,S = sig.spectrogram(w, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    S_log = np.log10(S+1e-10)
    lo,hi = np.percentile(S_log,[2,98])
    return np.clip((S_log-lo)/(hi-lo+1e-10),0,1)

# --- Generators (same as before) ---
def gen_tect_shallow():
    dur=np.random.uniform(4,10); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n); w=np.zeros(n)
    pt=np.random.uniform(0.3,1); st=pt+np.random.uniform(1.5,3)
    w+=np.exp(-np.abs(t-pt)/0.3)*(t>=pt)*np.sin(2*np.pi*(10+np.random.randn()*2)*t)*0.6
    w+=np.exp(-(t-st)/1)*(t>=st)*np.sin(2*np.pi*(4+np.random.randn())*t)
    w+=np.exp(-(t-st-.5)/2)*(t>=st+.5)*np.sin(2*np.pi*2*t)*.2; return w

def gen_tect_deep():
    dur=np.random.uniform(10,28); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n); w=np.zeros(n)
    pt=np.random.uniform(.5,2); st=pt+np.random.uniform(4,8)
    w+=np.exp(-np.abs(t-pt)/0.8)*(t>=pt)*np.sin(2*np.pi*(5+np.random.randn())*t)*.3
    w+=np.exp(-(t-st)/3)*(t>=st)*np.sin(2*np.pi*(2+np.random.randn()*.5)*t)*.8
    w+=np.exp(-(t-st-1)/6)*(t>=st+1)*np.sin(2*np.pi*t)*.4; return w

def gen_volc_tremor():
    dur=np.random.uniform(12,32); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n)
    on=np.random.uniform(.5,2); sus=dur*np.random.uniform(.5,.8)
    env=np.clip((t-on)/3,0,1)*np.exp(-np.maximum(0,t-on-sus)/3)
    f0=np.random.uniform(1.5,3)
    w=env*(np.sin(2*np.pi*f0*t)*.7+np.sin(2*np.pi*2*f0*t)*.3+np.sin(2*np.pi*3*f0*t)*.15)
    w*=(1+.2*np.sin(2*np.pi*.2*t)); return w

def gen_volc_lp():
    dur=np.random.uniform(2,7); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n)
    on=np.random.uniform(.3,1); f0=np.random.uniform(1,2.5)
    env=np.exp(-((t-on)/.8)**2)+.3*np.exp(-((t-on-1.5)/1.2)**2)
    return env*(np.sin(2*np.pi*f0*t)*.8+np.sin(2*np.pi*2*f0*t)*.3)

def gen_explosion():
    dur=np.random.uniform(1.5,5); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n)
    on=np.random.uniform(.2,.8); env=np.exp(-(t-on)/.5)*(t>=on); w=np.zeros(n)
    for f in np.random.uniform(5,35,8): w+=env*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))/(1+f/15)
    return w*.7

def gen_icequake():
    dur=np.random.uniform(.8,2.5); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n)
    on=np.random.uniform(.1,.4); env=np.exp(-((t-on)/.1)**2)
    return env*(np.sin(2*np.pi*(15+np.random.randn()*3)*t)+.5*np.sin(2*np.pi*(25+np.random.randn()*5)*t))

def gen_noise_burst():
    dur=np.random.uniform(1,6); n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n)
    on=np.random.uniform(.2,1); env=np.exp(-(t-on)/np.random.uniform(.5,2))*(t>=on); w=np.zeros(n)
    for f in np.random.uniform(3,30,12): w+=env*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))*np.random.uniform(.1,.5)
    w*=(1+.4*np.random.randn(n).cumsum()/np.sqrt(n)); return w

GENERATORS = {'tect_shallow': gen_tect_shallow, 'tect_deep': gen_tect_deep,
              'volc_tremor': gen_volc_tremor, 'volc_lp': gen_volc_lp,
              'explosion': gen_explosion, 'icequake': gen_icequake, 'noise_burst': gen_noise_burst}


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-10)


# ============================================================================
# NEW: Waveform energy envelope profile
# ============================================================================

def waveform_envelope_profile(waveform, n_bins=32):
    """
    Compute energy envelope of the raw waveform, normalised to n_bins.

    This works on the RAW WAVEFORM which has consistent time resolution
    regardless of event duration. The profile captures the temporal
    shape: where energy concentrates relative to event duration.
    """
    # Compute instantaneous energy (squared amplitude, smoothed)
    energy = waveform**2
    # Smooth with a short window
    win = max(3, len(waveform) // 50)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    energy_smooth = np.convolve(energy, kernel, mode='same')

    # Bin into n_bins equal segments
    edges = np.linspace(0, len(energy_smooth), n_bins+1).astype(int)
    profile = np.zeros(n_bins)
    for b in range(n_bins):
        seg = energy_smooth[edges[b]:edges[b+1]]
        if len(seg) > 0:
            profile[b] = np.mean(seg)

    # Normalise to unit sum (so it's a distribution)
    total = profile.sum()
    if total > 1e-10:
        profile = profile / total

    return profile


def spectral_profile(spec, n_bins=16):
    n = spec.shape[0]
    edges = np.linspace(0, n, n_bins+1).astype(int)
    p = np.zeros(n_bins)
    for b in range(n_bins):
        s = spec[edges[b]:edges[b+1], :]
        if s.size > 0: p[b] = np.mean(s)
    return p


# ============================================================================
# Generate and test
# ============================================================================
N_SEEDS = 3
N_TEST = 25

for SNR in [3, 5, 8]:
    print()
    print("=" * 70)
    print(f"  SNR = {SNR} dB")
    print("=" * 70)

    dataset = []
    for cname, gen in GENERATORS.items():
        for i in range(N_SEEDS + N_TEST):
            w = gen()
            w = add_noise(w, SNR)
            spec = make_spec(w)
            dataset.append((cname, i < N_SEEDS, spec, w))

    REPS = {
        'raw_64': lambda s, w: zoom(s, (1, 64/s.shape[1]), order=1).flatten(),
        'spectral_16': lambda s, w: spectral_profile(s, 16),
        'spectral_32': lambda s, w: spectral_profile(s, 32),
        'envelope_16': lambda s, w: waveform_envelope_profile(w, 16),
        'envelope_32': lambda s, w: waveform_envelope_profile(w, 32),
        'envelope_64': lambda s, w: waveform_envelope_profile(w, 64),
        'spec16+env16': lambda s, w: np.concatenate([spectral_profile(s,16), waveform_envelope_profile(w,16)]),
        'spec16+env32': lambda s, w: np.concatenate([spectral_profile(s,16), waveform_envelope_profile(w,32)]),
        'spec32+env32': lambda s, w: np.concatenate([spectral_profile(s,32), waveform_envelope_profile(w,32)]),
        'spec16+env32+env64': lambda s, w: np.concatenate([spectral_profile(s,16), waveform_envelope_profile(w,32), waveform_envelope_profile(w,64)]),
    }

    print(f"\n  {'rep':>20}  {'dim':>4}  {'gap':>8}  {'mg':>7}  {'acc':>5}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}")

    best_acc = 0
    best_name = ""

    for rname, rfn in REPS.items():
        seeds = {c: [] for c in GENERATORS}
        test = []
        for cn, is_s, sp, w in dataset:
            v = rfn(sp, w)
            if is_s: seeds[cn].append(v)
            else: test.append((cn, v))

        within, between = [], []
        for c in GENERATORS:
            vv = seeds[c]
            for i in range(len(vv)):
                for j in range(i+1,len(vv)):
                    within.append(cosine(vv[i],vv[j]))
        for i,c1 in enumerate(list(GENERATORS)):
            for c2 in list(GENERATORS)[i+1:]:
                for v1 in seeds[c1]:
                    for v2 in seeds[c2]:
                        between.append(cosine(v1,v2))
        gap = min(within)-max(between)
        mg = np.mean(within)-np.mean(between)

        models = {c: np.mean(v, axis=0) for c,v in seeds.items()}
        correct = sum(1 for tc,v in test if max(models, key=lambda c: cosine(v,models[c]))==tc)
        acc = correct/len(test)

        dim = len(rfn(dataset[0][2], dataset[0][3]))
        mark = ""
        if acc > best_acc:
            best_acc = acc
            best_name = rname
            mark = " <-- BEST"

        print(f"  {rname:>20}  {dim:4d}  {gap:+8.4f}  {mg:+7.4f}  {acc:5.0%}{mark}")

    print(f"\n  Winner: {best_name} ({best_acc:.0%})")

    # Show detailed confusion for winner and raw baseline
    for rname in ('raw_64', best_name):
        if rname == 'raw_64' and rname == best_name:
            continue
        rfn = REPS[rname]
        seeds = {c: [] for c in GENERATORS}
        test = []
        for cn, is_s, sp, w in dataset:
            v = rfn(sp, w)
            if is_s: seeds[cn].append(v)
            else: test.append((cn, v))
        models = {c: np.mean(v, axis=0) for c,v in seeds.items()}

        classes = sorted(GENERATORS.keys())
        print(f"\n  {rname} confusion:")
        print(f"  {'':>15}", end='')
        for c in classes: print(f" {c[:5]:>5}", end='')
        print()
        conf = {c1:{c2:0 for c2 in classes} for c1 in classes}
        for tc,v in test:
            pred = max(models, key=lambda c: cosine(v,models[c]))
            conf[tc][pred] += 1
        for c1 in classes:
            print(f"  {c1:>15}", end='')
            for c2 in classes:
                print(f" {conf[c1][c2]:5d}", end='')
            print()


print("\n" + "=" * 70)
print("  INVESTIGATION SUMMARY")
print("=" * 70)
print("""
  The waveform envelope profile marginalises the temporal axis at
  CONSISTENT resolution (raw waveform samples, not spectrogram bins).

  Results show whether combining spectral + temporal envelope profiles
  beats raw pixel comparison on variable-size seismic spectrograms.
""")
