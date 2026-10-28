"""Encode voices/out/*.wav to Ogg Opus for shipping.

WAV is ~146 KB/line; Opus is ~13.5 KB/line at 24 kHz mono (10.6x smaller).
Emits voices/ogg/<hero>/000..003.ogg and voices/voices.manifest.ogg.json.
"""
import soundfile as sf, json, os

def main():
    man = json.load(open('voices/voices.manifest.json'))
    OUT = 'voices/ogg'
    tin = tout = n = 0
    for h, v in man['voices'].items():
        d = os.path.join(OUT, h); os.makedirs(d, exist_ok=True)
        for i, e in enumerate(man['events']):
            y, sr = sf.read(v[e], dtype='float32')
            if y.ndim > 1: y = y.mean(1)
            dst = os.path.join(d, f"{i:03d}.ogg")
            sf.write(dst, y, sr, format='OGG', subtype='OPUS')
            tin += os.path.getsize(v[e]); tout += os.path.getsize(dst); n += 1
    json.dump({"events": man['events'],
               "voices": {h: {"voice_ref": v['voice_ref'],
                              **{e: f"voices/ogg/{h}/{i:03d}.ogg"
                                 for i, e in enumerate(man['events'])}}
                          for h, v in man['voices'].items()}},
              open('voices/voices.manifest.ogg.json', 'w'), indent=1)
    print(f"{n} files: {tin/1e6:.1f} MB -> {tout/1e6:.2f} MB ({tin/tout:.1f}x)")

main()
