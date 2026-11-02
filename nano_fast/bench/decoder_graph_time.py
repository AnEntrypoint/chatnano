"""Timing of the flow decoder graphs against eager: the first graphs-on call per text (capture included), then
five interleaved rounds of the flow alone and of generate(), medians. generate is seeded each time so both modes
sample the same tokens. Usage: python3 -I decoder_graph_time.py OUT.json
"""
import json
import statistics
import sys
import time

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SIL  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXTS = {
    "hello": "Hello there.",
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
    "paragraph": ("The new model arrived last week, and the team has been testing it against the old one for "
                  "three days. Most of the difference is in the first two seconds of each reply."),
}
ROUNDS = 5


def once(fn):
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
s3 = base.s3gen
result = {"rows": {}}
with torch.inference_mode():
    for name, text in TEXTS.items():
        torch.manual_seed(0)
        toks = fast.speech_tokens(text).flatten()
        toks = toks[toks < 6561].to("cuda")
        toks = torch.cat([toks, torch.tensor([S3GEN_SIL] * 3, device="cuda").long()])

        def flow():
            with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
                return s3.flow_inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2,
                                         finalize=True)

        def gen():
            torch.manual_seed(0)
            return fast.generate(text)

        fast.enable_decoder_graphs(True)
        first = once(flow)
        fast.enable_decoder_graphs(False)
        flow()
        ef, gf, eg, gg = [], [], [], []
        for _ in range(ROUNDS):
            fast.enable_decoder_graphs(False)
            ef.append(once(flow))
            eg.append(once(gen))
            fast.enable_decoder_graphs(True)
            gf.append(once(flow))
            gg.append(once(gen))
        result["rows"][name] = {
            "speech_tokens": int(toks.numel()),
            "first_graph_flow_ms": round(first, 1),
            "flow_eager_ms": round(statistics.median(ef), 1),
            "flow_graph_ms": round(statistics.median(gf), 1),
            "generate_eager_ms": round(statistics.median(eg), 1),
            "generate_graph_ms": round(statistics.median(gg), 1),
        }
    result["graph_keys"] = [list(map(str, k)) for k in fast.decoder_graphs.graphs]
fast.enable_decoder_graphs(True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
