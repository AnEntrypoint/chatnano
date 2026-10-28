"""Chatterbox-Nano with a static KV cache and CUDA-graph decode step.

Same speech tokens as ChatterboxTurboTTS.generate() for a fixed seed. The autoregressive loop
differs: prefill runs eagerly, the 1-token decode step is compiled with
torch.compile(mode="reduce-overhead"), and the cache length is bucketed. Two stages differ by
default: S3Gen runs under fp16 autocast (enable_fp16_decoder(False) restores fp32), and the
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
from chatterbox.models.s3gen.s3gen import S3Token2Mel

MAX_NEW_TOKENS = 1000
CACHE_BUCKET = 256
FIRST_BUDGET = 384  # generated tokens the first, tight-cache attempt may use
STOP_CHECK_EVERY = 16  # steps between host syncs for the stop token (measured faster than a check every step)
SPEECH_VOCAB = 6561  # speech token ids at or above this are control tokens, not audio

# streaming: tokens before the first audio, tokens between renders, left context, and the newest tokens
# held back: LOOKAHEAD for the encoder inside the window, plus TAIL so that the cross-faded tail has
# right context from the flow decoder (a tail rendered at the window's edge did not match the next window)
STREAM_FIRST = 20
STREAM_STEP = 30
STREAM_CONTEXT = 40
LOOKAHEAD = 3
TAIL = 10
STREAM_MARGIN = LOOKAHEAD + TAIL


def cut_at_stop_token(tokens, stop_token):
    # first stop after position 0 ends generation, matching T3 inference_turbo
    stops = (tokens[0, 1:] == stop_token).nonzero()
    if stops.numel() > 0:
        return tokens[:, :int(stops[0, 0]) + 1]
    return tokens


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

        # the vocoder too: storing its weights in half saves about 861 MB of VRAM and about
        # 508 MB of RSS. Measured LSD 0.265 dB against a 0.097 dB fp32-vs-fp32 control, tokens
        # identical, so it sits well under the 1.5 dB bar judged inaudible earlier.
        self.m.s3gen.half()
        for k, v in list(self.m.conds.gen.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                self.m.conds.gen[k] = v.half()
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

    @torch.inference_mode()
    def _token_stream(self, text, temperature, top_k, top_p, repetition_penalty, max_new_tokens):
        # One sampled speech token at a time, with a full-budget cache: once audio is out, the reply
        # cannot be restarted with a bigger budget. Same sampling as _decode, so tokens match it.
        t3, tfmr = self.t3, self.tfmr
        stop = t3.hp.stop_speech_token
        # one sequence needs no padding; padding=True breaks tensor creation on transformers 5.16 (Kaggle)
        ids = self.m.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(self.m.device)
        sst = t3.hp.start_speech_token * torch.ones_like(ids[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=self.m.conds.t3, text_tokens=ids,
                                            speech_tokens=sst, cfg_weight=0.0)
        L = embeds.shape[1]
        procs = self._sampler(temperature, top_k, top_p, repetition_penalty)
        cache_len = -(-(L + max_new_tokens) // CACHE_BUCKET) * CACHE_BUCKET
        cache = StaticCache(config=tfmr.config, max_cache_len=cache_len,
                            device=self.m.device, dtype=embeds.dtype)
        positions = torch.arange(L, L + max_new_tokens, device=self.m.device)
        logits = self._step(embeds, cache, torch.arange(L, device=self.m.device))
        tok = self._sample(procs, sst, logits)
        history = [tok]
        for i in range(max_new_tokens):
            if i > 0 and int(tok) == stop:
                return  # same stop rule as cut_at_stop_token: a stop after position 0 ends the reply
            yield int(tok)
            if i == max_new_tokens - 1:
                return
            logits = self.step(t3.speech_emb(tok), cache, positions[i:i + 1])
            tok = self._sample(procs, torch.cat(history, dim=1), logits)
            history.append(tok)

    @torch.inference_mode()
    def _render_mel(self, tokens, first, final, noise, mel1):
        # Mel for the window of speech tokens [s, win_end), s = first - STREAM_CONTEXT. Mel frame j of the
        # result is absolute frame 2*s + j. The noise is indexed by absolute frame, so windows share it.
        T = len(tokens)
        win_end = T if final else T - LOOKAHEAD
        s = max(0, first - STREAM_CONTEXT)
        win = torch.tensor(tokens[s:win_end], device=self.m.device, dtype=torch.long)
        if final:
            win = torch.cat([win, torch.tensor([S3GEN_SIL] * 3, device=self.m.device, dtype=torch.long)])
        M = 2 * win.numel()
        noise_win = torch.cat([noise[:, :, :mel1], noise[:, :, mel1 + 2 * s: mel1 + 2 * s + M]], dim=2)
        # the audio side draws random numbers; fork the RNG so token sampling is unaffected
        with torch.random.fork_rng(devices=[self.m.device]), \
                torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16_decoder):
            mel = S3Token2Mel.forward(self.m.s3gen, win[None], None, None, ref_dict=self.m.conds.gen,
                                      n_cfm_timesteps=2, finalize=True, noised_mels=noise_win)
        return s, mel[0].float()

    @torch.inference_mode()
    def _vocode(self, mel):
        # mel (80, frames) -> float32 samples, 480 per frame (24 kHz)
        with torch.random.fork_rng(devices=[self.m.device]), \
                torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16_decoder):
            wav, _ = self.m.s3gen.hift_inference(mel[None].to(dtype=self.m.s3gen.dtype))
        return wav.squeeze(0).float()

    @torch.inference_mode()
    def _advance(self, st, tokens, final, noise, mel1):
        # Commits the mel for speech tokens up to the render point and returns the newly final audio.
        # Frames of the previous render that are not committed yet are blended with this render's frames
        # over their overlap (mel domain, raised-cosine ramp), so the vocoder sees one continuous mel.
        T = len(tokens)
        E = T if final else T - STREAM_MARGIN
        if E <= st["emitted"]:
            return None
        s, mel = self._render_mel(tokens, st["emitted"], final, noise, mel1)
        seg = mel[:, 2 * st["emitted"] - 2 * s:]
        if st["pending"] is not None:
            ov = min(st["pending"].shape[1], seg.shape[1])
            w = (0.5 - 0.5 * torch.cos(torch.pi * torch.linspace(0, 1, ov, device=seg.device))).to(seg)
            seg = seg.clone()
            seg[:, :ov] = (1 - w) * st["pending"][:, :ov] + w * seg[:, :ov]
        n_commit = 2 * E - 2 * st["emitted"]
        st["mel"] = seg[:, :n_commit] if st["mel"] is None else torch.cat([st["mel"], seg[:, :n_commit]], dim=1)
        st["pending"] = seg[:, n_commit:]
        st["emitted"] = E
        wav = self._vocode(torch.cat([st["mel"], st["pending"]], dim=1))
        end = 480 * st["mel"].shape[1]
        piece = wav[st["samples"]:end]
        if st["samples"] == 0:
            fade = self.m.s3gen.trim_fade[:piece.numel()].to(piece)
            piece = piece.clone()
            piece[:fade.numel()] *= fade
        st["samples"] = end
        piece = piece.detach().cpu().numpy()
        return self.m.watermarker.apply_watermark(piece, sample_rate=self.m.sr).astype("float32")

    def _new_state(self):
        return {"emitted": 0, "mel": None, "pending": None, "samples": 0}

    @torch.inference_mode()
    def render_whole(self, tokens, noise, mel1):
        # one render of all tokens: the reference for the streamed audio
        return self._advance(self._new_state(), tokens, True, noise, mel1)

    @torch.inference_mode()
    def stream(self, text, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
               max_new_tokens=MAX_NEW_TOKENS, noise_seed=0):
        # Yields float32 audio pieces while the reply is still being sampled. The first piece is ready
        # after STREAM_FIRST tokens. The flow noise has its own generator, so the tokens match _token_stream.
        mel1 = self.m.conds.gen["prompt_feat"].shape[1]
        g = torch.Generator(device=self.m.device).manual_seed(noise_seed)
        noise = torch.randn(1, 80, mel1 + 2 * max_new_tokens + 16, device=self.m.device,
                            dtype=torch.float32, generator=g)
        tokens, st, next_render = [], self._new_state(), STREAM_FIRST
        for tok in self._token_stream(text, temperature, top_k, top_p, repetition_penalty, max_new_tokens):
            if tok >= SPEECH_VOCAB:
                continue
            tokens.append(tok)
            if len(tokens) >= next_render:
                piece = self._advance(st, tokens, False, noise, mel1)
                if piece is not None:
                    yield piece
                next_render = len(tokens) + STREAM_STEP
        piece = self._advance(st, tokens, True, noise, mel1)
        if piece is not None:
            yield piece

if __name__ == "__main__":
    import soundfile as sf
    text = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
            "Would you like me to get some prices for you?")
    base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
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
