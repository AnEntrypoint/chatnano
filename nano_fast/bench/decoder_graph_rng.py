"""RNG and output check for the decoder graphs: the full S3Gen call with graphs off and on, same tokens and seed.

Reports the CUDA generator state after each call and the max abs difference of the waves. The first graphs-on call is
the one that captures, so its state is compared with the graphs-off state too.
Usage: python3 -I decoder_graph_rng.py OUT.json
"""
import hashlib
import json
import sys

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SIL  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")


def digest(t):
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
s3 = base.s3gen
result = {"runs": {}}
with torch.inference_mode():
    torch.manual_seed(0)
    toks = fast.speech_tokens(TEXT).flatten()
    toks = toks[toks < 6561].to("cuda")
    toks = torch.cat([toks, torch.tensor([S3GEN_SIL] * 3, device="cuda").long()])

    def full(on, seed):
        fast.enable_decoder_graphs(on)
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            wav, _ = s3.inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2)
        wav = wav.float().cpu()
        return wav, digest(torch.cuda.get_rng_state()), digest(wav)

    waves = {}
    for name, on, seed in [("off", False, 0), ("on_first", True, 0), ("on_again", True, 0),
                           ("off_again", False, 0), ("off_seed1", False, 1), ("on_seed1", True, 1)]:
        wav, rng, wh = full(on, seed)
        waves[name] = wav
        result["runs"][name] = {"rng_after": rng, "wav_hash": wh, "samples": int(wav.shape[-1])}
    result["max_abs"] = {
        "off_vs_on_first": float((waves["off"] - waves["on_first"]).abs().max()),
        "on_first_vs_on_again": float((waves["on_first"] - waves["on_again"]).abs().max()),
        "off_vs_off_again": float((waves["off"] - waves["off_again"]).abs().max()),
        "seed1_off_vs_on": float((waves["off_seed1"] - waves["on_seed1"]).abs().max()),
    }
fast.enable_decoder_graphs(True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
