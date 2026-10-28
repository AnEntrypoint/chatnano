"""Generate N variants of each bark, encoded straight to Ogg Opus at 16 kbps.

WAV is written only as an intermediate and deleted after encoding.

Per hero: 4 events (spawn, attack, ability, death) x 4 variants = 16 files,
named 000..015 in order spawn_0..spawn_3, attack_0..attack_3, ...

Manifest stays the same shape oscraft already reads (hero -> event -> path
string); the event key carries a _N suffix, so stripping it gives the event
and the game can pick a random variant.
"""
import argparse, json, os, subprocess, sys, time, tempfile
import torch, soundfile as sf


def ffmpeg_opus(src, dst, bitrate="16k", sr=24000):
    exe = ffmpeg_exe()
    subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", src, "-c:a", "libopus", "-b:a", bitrate,
                    "-ar", str(sr), "-ac", "1", dst], check=True)


_FFMPEG = None
def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heroes-file", default="/tmp/heroes_remaining.txt")
    ap.add_argument("--out", default="voices/ogg57")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--bitrate", default="16k")
    ap.add_argument("--lines", default="variants", choices=["variants","variants70"])
    ap.add_argument("--manifest", default="voices/voices.manifest.57.json")
    ap.add_argument("--assign", default=None,
                    help="json {hero: refname} to force specific reference voices")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hero_lines import EVENTS
    from hero_lines_variants import VARIANTS
    from hero_lines_variants70 import VARIANTS70
    LINES = VARIANTS if args.lines == 'variants' else VARIANTS70
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    sys.path.insert(0, "/workspace/chatnano/nano_fast")
    from fast_nano import FastNano

    heroes = [h.strip() for h in open(args.heroes_file) if h.strip()]
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=True)
    fast = FastNano(model)

    # reference voices: use the wider 31-speaker pool, round-robin
    refs = sorted(f for f in os.listdir("voices/refs") if f.endswith(".wav"))
    print(f"reference pool: {len(refs)} speakers", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="varwav_")
    manifest, done, t_all = {}, 0, time.perf_counter()

    assign = json.load(open(args.assign)) if args.assign else {}
    for hi, hero in enumerate(heroes):
        if hero in assign:
            refname = assign[hero]
            ref = os.path.join("voices/refs", f"pd_{refname}.wav")
            if not os.path.exists(ref):
                print(f"  WARN {hero}: ref pd_{refname}.wav missing, using round-robin")
                ref = os.path.join("voices/refs", refs[hi % len(refs)])
        else:
            ref = os.path.join("voices/refs", refs[hi % len(refs)])
        model.prepare_conditionals(ref)
        fast.generate(LINES[hero][EVENTS[0]][0])  # warm compiled path
        events = {}
        idx = 0
        for ev in EVENTS:
            for v, line in enumerate(LINES[hero][ev][:args.variants]):
                torch.manual_seed(hi * 1000 + idx)
                with torch.inference_mode():
                    wav = fast.generate(line)
                wavp = os.path.join(tmpdir, f"{hero}_{idx:03d}.wav")
                sf.write(wavp, wav.squeeze().float().cpu().numpy(), model.sr)
                oggp = os.path.join(args.out, hero, f"{idx:03d}.ogg")
                os.makedirs(os.path.dirname(oggp), exist_ok=True)
                ffmpeg_opus(wavp, oggp, args.bitrate, model.sr)
                os.remove(wavp)
                events[f"{ev}_{v}"] = oggp
                idx += 1
                done += 1
        manifest[hero] = {"voice_ref": os.path.basename(ref)[:-4], **events}
        el = time.perf_counter() - t_all
        print(f"RESULT {hero}: 16 clips ({done} total) {el:.0f}s elapsed", flush=True)

    json.dump({"events": [f"{e}_{v}" for e in EVENTS for v in range(args.variants)],
               "variants_per_event": args.variants,
               "bitrate": args.bitrate,
               "voices": manifest},
              open(args.manifest, "w"), indent=1)
    print(f"RESULT total: {done} clips -> {args.out}")


if __name__ == "__main__":
    main()
