"""Fetch CC0 acted voice packs from OpenGameArt and build reference clips.

These are ACTED character barks (shouted lines), not read speech, so they suit
game characters better than audiobook readers. All CC0, verified on each page.

| pack | url | licence | voices |
|---|---|---|---|
| Voiceover Pack | /content/voiceover-pack-40-lines | CC0 | female, male (long previews) |
| kenney voiceoverFighter | /content/voiceover-pack-fighter-40-taunts | CC0 | fighter (long preview) |
| RPG Voice Starter Pack | /content/female-rpg-voice-starter-pack | CC0 | 3 types (short clips, concatenated) |

Not usable (licence): 24-cute-fighting-game-voice-recordings, female-soldier-voice,
voices-sound-effects-library, soliloquy - all CC-BY 3.0, which needs attribution.

Writes voices/refs/pd_act_*.wav (12 s, 24 kHz mono).
"""
import glob, os, subprocess, zipfile
import numpy as np, soundfile as sf, librosa

SR = 24000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
PACKS = {
    "vop": "https://opengameart.org/sites/default/files/Voiceover%20Pack.zip",
    "fighter": "https://opengameart.org/sites/default/files/kenney_voiceoverFighter.zip",
    "rpg": "https://opengameart.org/sites/default/files/RPG%20Voice%20Starter%20Pack.zip",
}
COMPILATIONS = [  # already single-speaker and long enough
    ("vop/Preview (Female).ogg", "act_female"),
    ("vop/Preview (Male).ogg", "act_male"),
    ("fighter/Preview.ogg", "act_fighter"),
]


def norm(y):
    return y / (np.max(np.abs(y)) + 1e-9) * 0.85


def to_sr(y, sr):
    return librosa.resample(y, orig_sr=sr, target_sr=SR)


def best_slice(y, dur=12.0):
    need = int(dur * SR)
    if len(y) < need:
        return y
    hop = int(0.5 * SR)
    best, bp = -1, 0
    for s in range(0, max(1, len(y) - need), hop):
        sc = np.percentile(np.abs(y[s:s + need]), 25)
        if sc > best:
            best, bp = sc, s
    return y[bp:bp + need]


def main():
    tmp = "/tmp/cc0_acted"
    os.makedirs(OUT, exist_ok=True)
    for name, url in PACKS.items():
        z = f"{tmp}/{name}.zip"
        os.makedirs(tmp, exist_ok=True)
        if not os.path.exists(z):
            subprocess.run(["curl", "-sL", "--max-time", "300", url, "-o", z], check=False)
        zipfile.ZipFile(z).extractall(f"{tmp}/{name}")

    for rel, key in COMPILATIONS:
        p = f"{tmp}/{rel}"
        if not os.path.exists(p):
            print("missing", rel); continue
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        sf.write(f"{OUT}/pd_{key}.wav", norm(best_slice(to_sr(y, sr))), SR)
        print("built pd_" + key)

    for t in ["Type 1", "Type 2", "Type 3"]:
        parts = []
        for p in sorted(glob.glob(f"{tmp}/rpg/RPG Voice Starter Pack/{t}/*")):
            try:
                y, sr = sf.read(p, dtype="float32")
            except Exception:
                continue
            if y.ndim > 1: y = y.mean(1)
            parts.append(to_sr(y, sr))
            if sum(len(x) for x in parts) > SR * 14:
                break
        if not parts:
            continue
        key = "act_rpg" + t.split()[-1]
        sf.write(f"{OUT}/pd_{key}.wav", norm(best_slice(np.concatenate(parts))), SR)
        print("built pd_" + key)


if __name__ == "__main__":
    main()
