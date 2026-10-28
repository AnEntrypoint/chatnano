import json, os, subprocess, numpy as np, soundfile as sf, librosa

IDENT = "spc195_1909_librivox"
meta = json.load(open('/tmp/md.json'))
files = [f['name'] for f in meta.get('files', []) if f['name'].endswith('_64kb.mp3')]
# distinct readers: the _xx initials suffix before _64kb
seen, picks = set(), []
for n in files:
    base = n[:-len('_64kb.mp3')]
    reader = base.split('_')[-1]
    if reader in seen: continue
    seen.add(reader); picks.append((reader, n))
    if len(picks) >= 32: break
print("readers picked:", [p[0] for p in picks])

SR = 24000
def best_window(y, sr, dur=12.0):
    """find a loud, non-silent window of `dur` seconds"""
    need = int(dur * sr)
    if len(y) < need: 
        y = np.pad(y, (0, need - len(y)))
    hop = int(0.5 * sr)
    best, bestpos = -1, 0
    for s in range(0, max(1, len(y) - need), hop):
        seg = np.abs(y[s:s+need])
        # favour steady energy (speech) over silence
        score = np.percentile(seg, 25)
        if score > best: best, bestpos = score, s
    return y[bestpos:bestpos+need]

def f0_hz(y, sr):
    """crude median F0 via autocorrelation, for speaker-distinctness evidence"""
    y = y - y.mean(); y = librosa.effects.trim(y, top_db=25)[0]
    if len(y) < sr//4: return None
    fmin, fmax = 70, 300
    frame = y[:sr//2]
    ac = np.correlate(frame, frame, 'full')[len(frame):]
    lo, hi = int(sr/fmax), int(sr/fmin)
    if hi >= len(ac): return None
    idx = lo + int(np.argmax(ac[lo:hi]))
    return sr/idx

rows = []
for reader, name in picks:
    dst = f"/tmp/lvmp3/{name}"
    if not os.path.exists(dst):
        subprocess.run(["curl","-sL","--max-time","120",
                        f"https://archive.org/download/{IDENT}/{name}","-o",dst], check=False)
    if not os.path.exists(dst) or os.path.getsize(dst) < 10000:
        print("SKIP (download failed)", reader); continue
    try:
        y, sr = sf.read(dst, dtype='float32')
    except Exception as e:
        print("SKIP (decode failed)", reader, e); continue
    if y.ndim > 1: y = y.mean(axis=1)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    w = best_window(y, SR)
    w = w / (np.max(np.abs(w)) + 1e-9) * 0.85
    out = f"/workspace/chatnano/voices/refs/pd_{reader}.wav"
    sf.write(out, w, SR)
    rows.append((reader, round(len(w)/SR,1), round(f0_hz(w, SR) or -1, 1)))
    print(f"OK pd_{reader}: {len(w)/SR:.1f}s")

print("--- reference pool ---")
for r in rows: print(f"  {r[0]:8s} {r[1]:5.1f}s  estF0={r[2]:.0f} Hz")
np.save('/tmp/ref_rows.npy', np.array(rows, dtype=object), allow_pickle=True)
