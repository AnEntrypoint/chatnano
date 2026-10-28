# Hero voices for oscraft

70 hero voices, 4 barks each (spawn, attack, ability, death) = 280 WAVs,
24 kHz mono, generated with Chatterbox-Nano.

## Layout

- `voices/out/<hero>/<000..003>.wav`  — the audio (000 spawn, 001 attack, 002 ability, 003 death)
- `voices/manifest.json`              — input to make_voices.py (hero -> ref + lines)
- `voices/voices.manifest.json`       — what oscraft imports (hero -> voice_ref + per-event paths)
- `voices/refs/pd_*.wav`              — reference clips, 12 distinct public-domain speakers
- `hero_lines.py`                     — the 280 lines, one set of 4 per hero
- `make_voices.py`                    — generator (manifest -> WAVs)
- `build_manifest.py`                 — assigns a reference voice per hero by archetype
- `fetch_refs.py`                     — rebuilds the reference pool from LibriVox

## Licensing

Reference clips are public-domain LibriVox recordings (Short Poetry Collection 195,
archive.org, each poem read by a different volunteer). Line text is original, written
for this project — it is not the original game's script. No game assets were used.

## Reproduce

```
python3 -I voices/fetch_refs.py          # rebuild reference pool (needs network)
python3 -I voices/build_manifest.py      # assign voices, write manifests
python3 -I voices/make_voices.py voices/manifest.json --fast
```

## Measured

- 280 lines, 70 heroes, ~1.9 s/line on RTX 3060 Laptop (6 GB).
- Pitch clusters by reference voice: across-voice spread 44.3 Hz vs within-voice 18.8 Hz,
  spanning 105-254 Hz. 12 distinct voices shared across 70 heroes.
- Turbo/Nano ignore cfg_weight and exaggeration, so voice identity comes from the
  reference clip; distinct timbres therefore require distinct reference speakers.
