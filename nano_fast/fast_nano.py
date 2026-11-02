"""Chatterbox-Nano with a static KV cache and CUDA-graph decode step.

Same speech tokens as ChatterboxTurboTTS.generate() for a fixed seed. The autoregressive loop
differs: prefill runs eagerly, the 1-token decode step is compiled with
torch.compile(mode="reduce-overhead"), and the cache length is bucketed. Two stages differ by
default: S3Gen runs under fp16 autocast (enable_fp16_decoder(False) restores fp32), its CFM decoder
replays as CUDA graphs over length buckets (enable_decoder_graphs(False) runs it eagerly), and the
watermark runs on the GPU. serve.py is the resident server built on this module.

Decode-step timings on RTX 3060 Laptop (6 GB), torch 2.6.0 / cu126, from the first fast-path
commit (see reports/Chatterbox Nano maximum speed RTX 3060.md for current figures):
  dynamic cache 7.6 ms, static+CUDA graphs 5.3 ms (greedy, same tokens).
"""
import sys
import time

import torch
from transformers import StaticCache
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)
import torch.nn.functional as F

from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm, S3GEN_SIL
from chatterbox.models.s3gen import decoder as cfm_decoder
from chatterbox.models.s3gen.s3gen import S3Token2Mel

MAX_NEW_TOKENS = 1000
DECODER_BUCKET = 64  # mel frames per graph bucket (32 speech tokens)
DECODER_WARM_FRAMES = 2560  # covers a 1000-token reply: 500 reference frames plus two mel frames per token
CACHE_BUCKET = 256
FIRST_BUDGET = 384  # generated tokens the first, tight-cache attempt may use
STOP_CHECK_EVERY = 16  # steps between host syncs for the stop token (measured faster than a check every step)



def cut_at_stop_token(tokens, stop_token):
    # first stop after position 0 ends generation, matching T3 inference_turbo
    stops = (tokens[0, 1:] == stop_token).nonzero()
    if stops.numel() > 0:
        return tokens[:, :int(stops[0, 0]) + 1]
    return tokens


_chunk_mask = cfm_decoder.add_optional_chunk_mask


def _chunk_mask_capture_safe(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                             static_chunk_size, num_decoding_left_chunks, *args, **kwargs):
    # The original ends with a host sync that only warns when a row is all false. Without chunking the masks
    # are the key mask, which has a valid frame in every row, so that branch never changes the result; it is
    # skipped while a graph is captured, where a sync is not allowed. Eager calls keep the original.
    if torch.cuda.is_current_stream_capturing() and not use_dynamic_chunk and static_chunk_size == 0:
        return masks
    return _chunk_mask(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                       static_chunk_size, num_decoding_left_chunks, *args, **kwargs)


cfm_decoder.add_optional_chunk_mask = _chunk_mask_capture_safe


class GraphedFlowDecoder:
    """The S3Gen flow's CFM decoder (meanflow path, batch 1) replayed as one CUDA graph per length bucket.

    The decoder's two Euler steps are about a thousand small kernels per call, and the graph replays them in one
    launch. Each call pads the real mel length up to the next DECODER_BUCKET, so one graph serves a range of
    lengths. The padding is zeroed and masked, so it changes the valid frames by rounding only (see
    bench/decoder_graph_check.py). The noise is drawn the way the eager decoder draws it: randn_like over the real
    length, from the same generator, so the generator advances as it would without the graph. Any other call goes
    to the eager forward.
    """

    def __init__(self, cfm, fp16):
        self.cfm = cfm
        self.fp16 = fp16  # callable: True while S3Gen runs under fp16 autocast
        self.enabled = True
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self._eager_forward = cfm.forward
        cfm.forward = self.forward  # nn.Module.__call__ looks up forward on the instance

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False):
        if not (self.enabled and meanflow and mu.shape[0] == 1 and noised_mels is not None
                and spks is not None and cond is not None):
            return self._eager_forward(mu, mask, n_timesteps, temperature, spks, cond, noised_mels, meanflow)
        L = mu.shape[2]
        bucket = -(-L // DECODER_BUCKET) * DECODER_BUCKET
        # the eager CFM draws z over the real length, then sets its generated tail to noised_mels
        z = torch.randn_like(mu)
        z[..., L - noised_mels.shape[2]:] = noised_mels
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        key = (bucket, int(n_timesteps), self.fp16(), mu.dtype, mask.dtype, cond.dtype, spks.dtype)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.graphs[key] = self._new_entry(key, mu.device)
        static = entry["in"]
        static["z"].zero_()
        static["z"][..., :L].copy_(z)
        static["mu"].zero_()
        static["mu"][..., :L].copy_(mu)
        static["mask"].zero_()
        static["mask"][..., :L].copy_(mask)
        static["cond"].zero_()
        static["cond"][..., :L].copy_(cond)
        static["spks"].copy_(spks)
        static["t"].copy_(t_span)
        if entry["graph"] is None:
            # the first call of a key warms up and captures on the inputs just copied, so the warm-up sees a real mask
            self._capture(entry)
        entry["graph"].replay()
        return entry["out"][..., :L].clone(), None

    def warm(self, max_frames, prompt_len, device="cuda"):
        """Capture a graph for every bucket up to max_frames, so no request pays for a capture (about 0.5 s each).

        Runs on zeros with the dtypes of a real fp16 call, and puts back the CUDA generator state it drew from.
        """
        rng = torch.cuda.get_rng_state(device)
        was_enabled, self.enabled = self.enabled, True
        buckets = sorted({-(-length // DECODER_BUCKET) * DECODER_BUCKET
                          for length in range(prompt_len + 2, max_frames + 1)})
        try:
            with torch.inference_mode():
                for bucket in buckets:
                    mu = torch.zeros(1, 80, bucket, dtype=torch.float16, device=device)
                    mask = torch.ones(1, 1, bucket, dtype=torch.float16, device=device)  # a zero mask is not a call
                    cond = torch.zeros(1, 80, bucket, dtype=torch.float16, device=device)
                    spks = torch.zeros(1, 80, dtype=torch.float16, device=device)
                    noised = torch.zeros(1, 80, bucket - prompt_len, dtype=torch.float32, device=device)
                    self.forward(mu, mask, 2, 1.0, spks, cond, noised, True)
        finally:
            self.enabled = was_enabled
            torch.cuda.set_rng_state(rng, device)
        return len(self.graphs)

    def _new_entry(self, key, device):
        bucket, n_timesteps, _, mu_dtype, mask_dtype, cond_dtype, spks_dtype = key
        static = {
            "z": torch.zeros(1, 80, bucket, dtype=mu_dtype, device=device),
            "mu": torch.zeros(1, 80, bucket, dtype=mu_dtype, device=device),
            "mask": torch.zeros(1, 1, bucket, dtype=mask_dtype, device=device),
            "cond": torch.zeros(1, 80, bucket, dtype=cond_dtype, device=device),
            "spks": torch.zeros(1, 80, dtype=spks_dtype, device=device),
            "t": torch.zeros(n_timesteps + 1, dtype=mu_dtype, device=device),
        }
        return {"graph": None, "in": static, "out": None, "fp16": key[2]}

    def _capture(self, entry):
        static = entry["in"]

        def run():
            with torch.autocast("cuda", dtype=torch.float16, enabled=entry["fp16"], cache_enabled=False):
                return self.cfm.basic_euler(static["z"], static["t"], static["mu"], static["mask"],
                                            static["spks"], static["cond"])

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = run()
        entry["graph"], entry["out"] = graph, out


class FastNano:
    def __init__(self, model: ChatterboxTurboTTS):
        self.m = model
        self.t3 = model.t3
        self.tfmr = model.t3.tfmr
        # eager attention, not sdpa: for a single decode row the sdpa path runs a memory-efficient
        # cutlass kernel that is slower. Interleaved runs: 25-35% faster generate, speech tokens
        # identical on all bench texts. Set before the step is compiled, so the trace uses it.
        self.tfmr.config._attn_implementation = "eager"
        # max-autotune measured 6% faster per step and 2.2 s faster to compile than reduce-overhead
        # (first generate 1.9 s vs 4.1 s), with identical tokens; max_autotune_gemm is off on
        # this GPU (too few SMs) so the autotune search stays cheap
        self.step = torch.compile(self._step, mode="max-autotune", fullgraph=False)
        # watermark on the GPU: 6-7 ms against 24-25 ms on CPU, max diff 2e-5, same decoded message
        model.watermarker.perth_net.to(model.device)
        # speaker conditionals per reference clip, so a repeated reference skips prepare_conditionals
        self._refs = {None: model.conds}
        self.fp16_decoder = True
        self.fp16_t3 = False  # _apply_fp16_t3 flips it
        self._apply_fp16_t3()
        self.decoder_graphs = GraphedFlowDecoder(model.s3gen.flow.decoder, lambda: self.fp16_decoder)

    def enable_decoder_graphs(self, on=True):
        # on by default; False runs the flow's CFM decoder eagerly (the output differs by rounding only)
        self.decoder_graphs.enabled = on

    def warm_decoder_graphs(self, max_frames=DECODER_WARM_FRAMES):
        # capture every length bucket now, for the reference clip in use (graphs are shared across clips)
        return self.decoder_graphs.warm(max_frames, self.m.conds.gen["prompt_feat"].shape[1], self.m.device)

    def enable_fp16_decoder(self, on=True):
        # default on: S3Gen under fp16 autocast, ~30% faster per clip. Spectral deviation from fp32 is
        # about 1.1-1.5 dB LSD and speaker similarity stays >= 0.99999. A listening check found no
        # audible difference (nano_fast/bench/ab/). Pass False to get the fp32 decoder.
        self.fp16_decoder = on

    def _fp16_step(self, embed, cache, cache_pos):
        torch.compiler.cudagraph_mark_step_begin()
        out = self.tfmr(inputs_embeds=embed, past_key_values=cache, cache_position=cache_pos, use_cache=True)
        # final projection in fp32: in half, near-tied logits among 6563 flip and the sampled
        # tokens change on longer text. The head itself stays half so the non-fast paths still
        # work; these are fp32 copies of its weights, cached once.
        h = out[0][:, -1:].float()
        return F.linear(h, self._head_weight_f32, self._head_bias_f32)[:, -1, :].clone()

    def _apply_fp16_t3(self):
        # Half the whole T3, then put the speech head back in fp32 and cast to it in the step.
        # Measured 0.859 s -> 0.506 s on a 6.6 s reply (1.70x). Keeping the head in half is also
        # fast but changes the tokens; keeping it in fp32 is output-neutral here: LSD 0.50/0.38 dB
        # against a 0.39 dB fp32-vs-fp32 control, speaker similarity 1.00000, tokens identical on
        # both bench texts. Note autocast is NOT a substitute: it casts weights on every call.
        self.t3.half()
        for k, v in list(vars(self.m.conds.t3).items()):
            if torch.is_tensor(v) and v.is_floating_point():
                setattr(self.m.conds.t3, k, v.half())
        self._head_weight_f32 = self.t3.speech_head.weight.detach().float().contiguous()

        # S3Gen keeps fp32 weights and runs under fp16 autocast (generate). Storing its weights in
        # half garbled the speech: with the same tokens and seed the flow's mel came out scrambled
        # (transcripts wrong, LSD 11.8 dB against fp32). Autocast over fp32 weights transcribes
        # like fp32 with LSD 1.47 dB. The half-stored figure of 0.265 dB could not be reproduced.
        self._head_bias_f32 = (self.t3.speech_head.bias.detach().float().contiguous()
                              if self.t3.speech_head.bias is not None else None)
        self._step = self._fp16_step            # prefill calls _step
        self.step = torch.compile(self._fp16_step, mode="max-autotune", fullgraph=False)
        self.fp16_t3 = True

    def enable_fp16_t3(self, on=True):
        """Half-precision T3 weights. On by default; pass False for the fp32 path."""
        if on and not self.fp16_t3:
            self._apply_fp16_t3()
        elif not on and self.fp16_t3:
            raise RuntimeError("fp16 T3 is applied at construction; recreate FastNano to disable")
        return self.fp16_t3

    def use_reference(self, wav_path=None):
        if wav_path not in self._refs:
            self.m.prepare_conditionals(wav_path)
            self._refs[wav_path] = self.m.conds
        self.m.conds = self._refs[wav_path]

    def _step(self, embed, cache, cache_pos):
        torch.compiler.cudagraph_mark_step_begin()
        out = self.tfmr(inputs_embeds=embed, past_key_values=cache, cache_position=cache_pos, use_cache=True)
        # clone: CUDA graphs reuse the output buffer on the next replay
        return self.t3.speech_head(out[0][:, -1:])[:, -1, :].clone()

    def _sampler(self, temperature, top_k, top_p, repetition_penalty):
        procs = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            procs.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        return procs

    @torch.inference_mode()
    def speech_tokens(self, text, temperature=0.8, top_k=1000, top_p=0.95,
                      repetition_penalty=1.2, max_new_tokens=MAX_NEW_TOKENS):
        t3, tfmr = self.t3, self.tfmr
        # one sequence needs no padding; padding=True breaks tensor creation on transformers 5.16 (Kaggle)
        ids = self.m.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(self.m.device)
        sst = t3.hp.start_speech_token * torch.ones_like(ids[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=self.m.conds.t3, text_tokens=ids,
                                            speech_tokens=sst, cfg_weight=0.0)
        L = embeds.shape[1]
        procs = self._sampler(temperature, top_k, top_p, repetition_penalty)
        # Attention runs over the whole cache every step, so a tight cache is faster. Try a small
        # budget first; if the reply does not stop within it, retry with the full budget from the
        # same RNG state, so the tokens are the same as one full-budget run.
        rng = torch.cuda.get_rng_state(self.m.device)
        for budget in dict.fromkeys((min(FIRST_BUDGET, max_new_tokens), max_new_tokens)):
            torch.cuda.set_rng_state(rng, self.m.device)
            out = self._decode(embeds, sst, L, procs, budget, max_new_tokens)
            if out is not None:
                return out
        raise RuntimeError("unreachable: the full budget always completes")

    def _decode(self, embeds, sst, L, procs, budget, max_new_tokens):
        t3, tfmr = self.t3, self.tfmr
        stop = t3.hp.stop_speech_token
        # Round the cache length up to a bucket so the compiled step is reused across texts.
        # Slots past the current position are masked by cache_position, so tokens are unchanged.
        cache_len = -(-(L + budget) // CACHE_BUCKET) * CACHE_BUCKET
        cache = StaticCache(config=tfmr.config, max_cache_len=cache_len,
                            device=self.m.device, dtype=embeds.dtype)
        # positions live on the device; a host tensor per step would copy host-to-device each time
        positions = torch.arange(L, L + budget, device=self.m.device)

        # prefill (eager: different shape from decode)
        logits = self._step(embeds, cache, torch.arange(L, device=self.m.device))
        generated = [sst]
        # the stop check is a host sync, so it runs every STOP_CHECK_EVERY steps; the tokens sampled after a
        # stop are cut below. The generator state after each draw is kept, so the RNG can be put back where
        # a per-step stop would have left it, and the vocoder's noise is the same as without the chunking.
        rng_after = []
        tok = self._sample(procs, sst, logits)
        generated.append(tok)
        rng_after.append(torch.cuda.get_rng_state(self.m.device))
        done = tok == stop
        for i in range(budget - 1):
            logits = self.step(t3.speech_emb(tok), cache, positions[i:i + 1])
            tok = self._sample(procs, torch.cat(generated[1:], dim=1), logits)
            generated.append(tok)
            rng_after.append(torch.cuda.get_rng_state(self.m.device))
            done = done | (tok == stop)
            if (i + 1) % STOP_CHECK_EVERY == 0 and bool(done.any()):
                break

        out = torch.cat(generated[1:], dim=1)
        if budget < max_new_tokens and not bool((out[0, 1:] == stop).any()):
            return None  # ran out of budget without a stop token
        stops = (out[0, 1:] == stop).nonzero()
        draw = int(stops[0, 0]) + 1 if stops.numel() > 0 else len(rng_after) - 1
        torch.cuda.set_rng_state(rng_after[draw], self.m.device)
        return cut_at_stop_token(out, stop)

    def _sample(self, procs, history, logits):
        logits = procs(history, logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(self, text, **kw):
        toks = self.speech_tokens(text, **kw)
        toks = toks[toks < 6561].to(self.m.device)
        sil = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.m.device)
        toks = torch.cat([toks, sil])
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16_decoder):
            wav, _ = self.m.s3gen.inference(speech_tokens=toks, ref_dict=self.m.conds.gen, n_cfm_timesteps=2)
        wav = wav.squeeze(0).detach().cpu().numpy()
        wav = self.m.watermarker.apply_watermark(wav, sample_rate=self.m.sr)
        return torch.from_numpy(wav).unsqueeze(0)

if __name__ == "__main__":
    import soundfile as sf
    text = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
            "Would you like me to get some prices for you?")
    base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
    fast = FastNano(base)
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    # the reference voice comes from the model's built-in conditionals
    assert base.conds is not None, "model has no default conditionals"

    def timed(fn, label):
        fn()  # warmup / compile
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); w = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        med = sorted(ts)[1]; audio = w.shape[-1] / base.sr
        print(f"RESULT {label}: median={med:.2f}s audio={audio:.2f}s RTF={med/audio:.3f}")
        return w

    if mode in ("both", "base"):
        torch.manual_seed(0)
        timed(lambda: base.generate(text), "baseline generate")
    if mode in ("both", "fast"):
        torch.manual_seed(0)
        w = timed(lambda: fast.generate(text), "fast generate")
        sf.write("/workspace/chatnano/nano_fast/fast_nano_out.wav", w.squeeze().float().cpu().numpy(), base.sr)
        print("RESULT fast wav written")
