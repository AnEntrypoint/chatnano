"""CFM decoder timing and replay check: the S3Gen flow's decoder (the two-step Euler solve), eager against one CUDA graph.

The decoder's inputs are captured from a real eager flow call by a forward pre-hook, so the graph replays the same
tensors. The decoder draws its noise for the prompt region from the global CUDA generator, so every call differs
unless the generator is seeded first; each eager call and each graph replay is seeded with the same value, and the
max abs difference between them is the replay check. The eager-against-eager difference is the control.

Two things stop the capture as the code stands. The encoder copies a CPU length tensor to the device, so only the
decoder is captured. Inside the decoder, add_optional_chunk_mask ends with a host sync that forces all-false rows
to true; with no chunking and no padding that never fires, so this probe swaps in the same function without the
sync and counts the rows it would have forced (empty_rows, which must be 0).
Usage: python3 -I flow_graph_probe.py OUT.json
"""
import json
import statistics
import sys
import time

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
import chatterbox.models.s3gen.decoder as cfm_decoder_mod  # noqa: E402
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SIL  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")
captured = {}
empty_rows = []


def chunk_mask_no_sync(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                       static_chunk_size, num_decoding_left_chunks, *args, **kwargs):
    assert not use_dynamic_chunk and static_chunk_size == 0
    if not torch.cuda.is_current_stream_capturing():
        empty_rows.append((masks.sum(dim=-1) == 0).sum())
    return masks


cfm_decoder_mod.add_optional_chunk_mask = chunk_mask_no_sync


def pre_hook(module, args, kwargs):
    captured["kwargs"] = dict(kwargs)


def median_ms(fn, n=10):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return round(statistics.median(times) * 1000, 1)


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
s3 = base.s3gen
decoder = s3.flow.decoder
result = {}
with torch.inference_mode():
    torch.manual_seed(0)
    toks = fast.speech_tokens(TEXT).flatten()
    toks = toks[toks < 6561].to("cuda")
    toks = torch.cat([toks, torch.tensor([S3GEN_SIL] * 3, device="cuda").long()])
    handle = decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    torch.manual_seed(1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder, cache_enabled=False):
        s3.flow_inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2, finalize=True)
    handle.remove()
    kw = captured["kwargs"]
    result["mel_frames"] = int(kw["mu"].shape[-1])

    def dec():
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder, cache_enabled=False):
            return decoder(**kw)[0]

    def eager(seed):
        torch.manual_seed(seed)
        return dec().clone()

    e5 = eager(5)
    result["eager_seed5_vs_seed5_max_abs"] = float((eager(5) - e5).abs().max())
    result["eager_seed5_vs_seed6_max_abs"] = float((eager(6) - e5).abs().max())
    result["decoder_eager_ms"] = median_ms(dec)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            dec()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = dec()

    def replay(seed):
        torch.manual_seed(seed)
        graph.replay()
        return static_out.clone()

    g5 = replay(5)
    result["graph_seed5_vs_eager_seed5_max_abs"] = float((g5 - e5).abs().max())
    result["graph_seed5_vs_seed5_again_max_abs"] = float((replay(5) - g5).abs().max())
    result["graph_seed5_vs_seed6_max_abs"] = float((replay(6) - g5).abs().max())
    result["decoder_graph_ms"] = median_ms(lambda: replay(5))
    result["decoder_eager_ms_again"] = median_ms(dec)
    result["empty_rows"] = int(sum(int(x) for x in empty_rows))
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
