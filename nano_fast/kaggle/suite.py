"""Bundled GPU experiments for Chatterbox-Nano, many ideas measured in one process.

Each experiment runs even if an earlier one fails. Results are written as they are measured, so a
run that dies partway still leaves its numbers behind.

    CHATNANO_SRC=<repo root> RESULTS=<json path> python suite.py [--quick] [--only a,b]

Experiments: env, probe, baseline, breakdown, cache, verify, batch, sampling, modes, quant.
--quick uses fewer shapes and skips the slow compiles and quantisation (a local smoke test).
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", default="")
args = ap.parse_args()
ONLY = {s for s in args.only.split(",") if s}

SRC = os.environ.get("CHATNANO_SRC", "/kaggle/input/chatnano-src")
RESULTS = os.environ.get("RESULTS", "/kaggle/working/results.json")
sys.path[:0] = [os.path.join(SRC, "chatterbox", "src"), os.path.join(SRC, "nano_fast")]


def _stub_torchaudio_if_broken():
    # chatterbox imports torchaudio at module level for speaker-embedding helpers this suite never calls.
    # On images whose torchaudio does not match their torch, that import itself fails; stub it only then.
    try:
        import torchaudio  # noqa: F401
    except Exception:
        import types
        ta = types.ModuleType("torchaudio")
        comp = types.ModuleType("torchaudio.compliance")
        kaldi = types.ModuleType("torchaudio.compliance.kaldi")
        ta.compliance, comp.kaldi = comp, kaldi
        sys.modules.update({"torchaudio": ta, "torchaudio.compliance": comp,
                            "torchaudio.compliance.kaldi": kaldi})


_stub_torchaudio_if_broken()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import StaticCache  # noqa: E402
from transformers.generation.logits_process import (  # noqa: E402
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm  # noqa: E402
from fast_nano import FastNano  # noqa: E402

if os.environ.get("CHATNANO_FP16_REDUCED") == "0":  # diagnostic: fp16 matmuls accumulate without reduced-precision reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
TRACE_POS = os.environ.get("CHATNANO_TRACE_POS") == "1"  # diagnostic: print decode positions as they run

DEV = "cuda"
SHORT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
         "Would you like me to get some prices for you?")
LONG = ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
        "tokens, so that the cache bucket and the stop check both get exercised across several "
        "boundaries in one run.")
TEXTS = {"short": SHORT, "long": LONG}
TEMP, TOPK, TOPP, PEN = 0.8, 1000, 0.95, 1.2
M_STATE = 768  # cache length for the timed state experiments
SPEECH_STOP = 6562  # the stop token; the timed loops suppress it so every timed step is a speech step
N_CFM = 2  # flow steps generate() uses (fast_nano.generate passes n_cfm_timesteps=2)

REF_TOKENS, REF_AUDIO = {}, {}
DATA = {"experiments": {}}


def record(name, **kw):
    DATA["experiments"][name] = kw
    os.makedirs(os.path.dirname(RESULTS) or ".", exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(DATA, f, indent=1, default=str)
    print("RESULT", name, json.dumps(kw, default=str), flush=True)


def run(name, fn, *a):
    if ONLY and name not in ONLY:
        return
    t = time.perf_counter()
    try:
        fn(*a)
    except Exception as e:  # keep going: one failed idea must not cost the rest of the run
        record(f"{name}_error", error=f"{type(e).__name__}: {str(e)[:300]}",
               tb=traceback.format_exc()[-1200:])
    print(f"RESULT {name}_wall_s {time.perf_counter() - t:.1f}", flush=True)
    release()


def release():
    gc.collect()
    torch.cuda.empty_cache()


def med_time(fn, n=7, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def event_ms(fn, n=10):
    """Per-call GPU time from CUDA events, with no profiler. Diagnostic alternative to gpu_ms."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def gpu_ms(fn, n=10):
    """Summed CUDA kernel time per call. Eager calls are CPU-bound (launch overhead), so wall time
    hides the GPU work; this is the work the device does per call."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    total_us = sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA")
    return total_us / n / 1000.0


def lsd_db(a, b):
    def spec(x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEV)
        s = torch.stft(x, n_fft=1024, hop_length=256, window=torch.hann_window(1024, device=DEV),
                       return_complex=True)
        return 20 * torch.log10(s.abs() + 1e-7)
    A, B = spec(a), spec(b)
    n = min(A.shape[-1], B.shape[-1])
    return float(((A[..., :n] - B[..., :n]) ** 2).mean().sqrt())


WEIGHTS_DIR = os.environ.get("CHATNANO_WEIGHTS", "/workspace/chatsrc/chatnano-fp16")


def load():
    # the fp16 directory when present (same weights the server serves); the hub otherwise
    if os.path.isdir(WEIGHTS_DIR):
        base = ChatterboxTurboTTS.from_local(WEIGHTS_DIR, DEV, nano=True)
    else:
        base = ChatterboxTurboTTS.from_pretrained(device=DEV, nano=True)
    return base, FastNano(base)


def peak_mb():
    return round(torch.cuda.max_memory_allocated() / 2 ** 20)


@torch.inference_mode()
def start_state(fast, text, M=M_STATE):
    # one sequence needs no padding; padding=True breaks tokenizer tensor creation on Kaggle's transformers 5.16
    ids = fast.m.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(DEV)
    sst = fast.t3.hp.start_speech_token * torch.ones_like(ids[:, :1])
    emb, _ = fast.t3.prepare_input_embeds(t3_cond=fast.m.conds.t3, text_tokens=ids,
                                          speech_tokens=sst, cfg_weight=0.0)
    cache = StaticCache(config=fast.tfmr.config, max_cache_len=M, device=DEV, dtype=emb.dtype)
    logits = fast._step(emb, cache, torch.arange(emb.shape[1], device=DEV))
    return cache, logits, emb.shape[1], int(sst[0, 0])


@torch.inference_mode()
def rewind(cache, n):
    """Set each layer's running length to n. transformers 5.16's StaticLayer writes at its own count
    (cumulative_length), not at the cache_position it is given, so a cache reused across calls must be
    rewound to the position each call writes at. Without this the count passes the cache length and
    index_copy_ asserts on the device (transformers 5.2 writes at cache_position, so it is a no-op there)."""
    for layer in cache.layers:
        if hasattr(layer, "cumulative_length"):  # transformers 5.16 only; 5.2 layers have no running count
            layer.cumulative_length.fill_(n)


def random_prefill(fast, batch, P, M):
    H = fast.tfmr.config.hidden_size
    cache = StaticCache(config=fast.tfmr.config, max_cache_len=M, device=DEV, dtype=torch.float16)
    e = torch.randn(batch, P, H, device=DEV, dtype=torch.float16) * 0.02
    fast._step(e, cache, torch.arange(P, device=DEV))
    return cache


# ---------------------------------------------------------------- experiments

def exp_env():
    import importlib
    import transformers

    def ver(mod):
        try:
            return importlib.import_module(mod).__version__
        except Exception:
            return None
    p = torch.cuda.get_device_properties(0)
    record("env", gpu=p.name, capability=f"{p.major}.{p.minor}", total_gb=round(p.total_memory / 2 ** 30, 2),
           torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
           triton=ver("triton"), torchao=ver("torchao"), python=sys.version.split()[0])


def retry_once(fn, *a):
    try:
        return fn(*a)
    except RuntimeError as e:  # a process's first CUDA-graph capture can fail once, then succeed
        print("RESULT retry_after", str(e)[:120], flush=True)
        return fn(*a)


@torch.inference_mode()
def exp_baseline(fast):
    torch.cuda.reset_peak_memory_stats()
    for name, text in TEXTS.items():
        torch.manual_seed(0)
        REF_TOKENS[name] = retry_once(fast.speech_tokens, text).flatten().tolist()
        torch.manual_seed(0)
        REF_AUDIO[name] = retry_once(fast.generate, text).squeeze(0).float().cpu().numpy()

        def run_once():
            torch.manual_seed(0)
            fast.generate(text)
        t = med_time(run_once, n=5, warm=2)
        audio_s = len(REF_AUDIO[name]) / 24000
        record(f"baseline_{name}", generate_s=round(t, 3), audio_s=round(audio_s, 2),
               rtf=round(t / audio_s, 4), tokens=len(REF_TOKENS[name]))
    recorded = json.load(open(os.path.join(SRC, "nano_fast", "bench", "new_tokens.json")))["fast"]
    record("baseline_vs_recorded_fp32", short_identical=REF_TOKENS["short"] == recorded[0],
           long_identical=REF_TOKENS["long"] == recorded[2], peak_mb=peak_mb())


@torch.inference_mode()
@torch.inference_mode()
def exp_attention(fast):
    """eager attention is the default; sdpa is measured on this GPU too, since the RTX 3060 result
    may not hold on other hardware. Same tokens are required."""
    fast.tfmr.config._attn_implementation = "sdpa"
    torch._dynamo.reset()
    try:
        torch.manual_seed(0)
        toks = retry_once(fast.speech_tokens, TEXTS["short"]).flatten().tolist()

        def run_once():
            torch.manual_seed(0)
            fast.generate(TEXTS["short"])
        t = med_time(run_once, n=5, warm=2)
        record("attention_sdpa", generate_s=round(t, 3), tokens_identical=toks == REF_TOKENS["short"])
    finally:
        fast.tfmr.config._attn_implementation = "eager"
        torch._dynamo.reset()


@torch.inference_mode()
def exp_breakdown(fast):
    text = TEXTS["short"]
    tok_t = med_time(lambda: fast.speech_tokens(text), n=5, warm=2)
    gen_t = med_time(lambda: fast.generate(text), n=5, warm=2)
    cache, logits, L, sst = start_state(fast, text)
    pos = torch.tensor([L], device=DEV)
    tok = torch.tensor([[sst]], device=DEV)
    step_t = med_time(lambda: (rewind(cache, L), fast.step(fast.t3.speech_emb(tok), cache, pos)), n=30, warm=5)
    procs = fast._sampler(TEMP, TOPK, TOPP, PEN)
    hist = torch.randint(0, 6561, (1, 120), device=DEV)
    samp_t = med_time(lambda: torch.multinomial(F.softmax(procs(hist, logits), dim=-1), 1), n=30, warm=5)
    record("breakdown", tokens_s=round(tok_t, 3), generate_s=round(gen_t, 3),
           vocoder_and_watermark_s=round(gen_t - tok_t, 3),
           device_step_ms=round(step_t * 1000, 3), host_sample_ms=round(samp_t * 1000, 3))


@torch.inference_mode()
def exp_cache_curve(fast):
    H = fast.tfmr.config.hidden_size
    out = {}
    points = [128, 384, 768] if args.quick else [128, 256, 512, 768, 1000]
    for P in points:
        cache = random_prefill(fast, 1, P, P + 24)
        e1 = torch.randn(1, 1, H, device=DEV, dtype=torch.float16)
        pos = torch.tensor([P], device=DEV)
        out[f"P{P}_gpu_ms"] = round(gpu_ms(lambda: (rewind(cache, P), fast._step(e1, cache, pos))), 3)
        del cache
        release()
    record("cache_curve_eager_ms", **out)


@torch.inference_mode()
def exp_verify(fast):
    H = fast.tfmr.config.hidden_size
    P = 400
    cache = random_prefill(fast, 1, P, P + 64)
    res = {}
    for k in (1, 2, 4, 8, 16):
        e = torch.randn(1, k, H, device=DEV, dtype=torch.float16)
        pos = torch.arange(P, P + k, device=DEV)

        def verify():
            # full-position logits, as speculative verification needs them
            rewind(cache, P)
            out = fast.tfmr(inputs_embeds=e, past_key_values=cache, cache_position=pos, use_cache=True)
            return F.linear(out[0].float(), fast._head_weight_f32, fast._head_bias_f32)
        if TRACE_POS:
            print("TRACE verify k", k, "positions", P, "to", P + k - 1, flush=True)
        # CHATNANO_VERIFY_EVENTS=1 times with CUDA events: the profiler path is the one that asserts on the T4
        timer = event_ms if os.environ.get("CHATNANO_VERIFY_EVENTS") == "1" else gpu_ms
        res[k] = timer(verify)
    record("verify_gpu_ms", **{f"k{k}": round(v, 3) for k, v in res.items()},
           ratio_k8_over_k1=round(res[8] / res[1], 3), ratio_k16_over_k1=round(res[16] / res[1], 3))


@torch.inference_mode()
def exp_probe(fast):
    """Finiteness of the decode step on the real sampling loop, and of the verify path on random input.
    A sync per step makes a device-side assert surface at the step that causes it. The T4 run asserted
    in the sampling loop and in the verify timing; this records which values go non-finite, and when."""
    H = fast.tfmr.config.hidden_size
    cache, logits, L, sst = start_state(fast, TEXTS["short"])
    procs = fast._sampler(TEMP, TOPK, TOPP, PEN)
    tok, hist, pos = (torch.tensor(v, device=DEV) for v in ([[sst]], [[sst]], [L]))
    bad_logit = bad_prob = None
    peak, steps = [], 0
    for i in range(200):
        steps = i + 1
        lg = fast.step(fast.t3.speech_emb(tok), cache, pos)
        if i % 25 == 0:
            peak.append(round(float(lg.float().abs().max()), 1))
        if not bool(torch.isfinite(lg).all()):
            bad_logit = i
            break
        probs = F.softmax(procs(hist, lg), dim=-1)
        if not bool(torch.isfinite(probs).all()):
            bad_prob = i
            break
        tok = torch.multinomial(probs, 1)
        hist = torch.cat([hist, tok], dim=1)
        pos.add_(1)
    record("probe_decode", steps=steps, first_nonfinite_logit_step=bad_logit,
           first_nonfinite_prob_step=bad_prob, max_abs_logit_every25=peak)

    finite = {}
    for scale in (0.02, 1.0):  # 0.02 is the real embedding scale random_prefill uses; 1.0 is exp_verify's input
        vc = random_prefill(fast, 1, 400, 464)
        for k in (1, 2, 4, 8, 16):
            e = torch.randn(1, k, H, device=DEV, dtype=torch.float16) * scale
            pos_v = torch.arange(400, 400 + k, device=DEV)
            rewind(vc, 400)
            out = fast.tfmr(inputs_embeds=e, past_key_values=vc, cache_position=pos_v, use_cache=True)
            finite[f"scale{scale}_k{k}"] = bool(torch.isfinite(out[0]).all())
        del vc
    record("probe_verify", finite=finite)


@torch.inference_mode()
def exp_batch(fast):
    H = fast.tfmr.config.hidden_size
    P, res = 200, {}
    for B in (1, 2, 4, 8):
        cache = random_prefill(fast, B, P, P + 32)
        e = torch.randn(B, 1, H, device=DEV, dtype=torch.float16)
        pos = torch.tensor([P], device=DEV)
        res[B] = gpu_ms(lambda: fast._step(e, cache, pos)) / 1000.0
        del cache
        release()
    record("batch_gpu", **{f"ms_B{b}": round(v * 1000, 3) for b, v in res.items()},
           **{f"tok_per_s_B{b}": round(b / v, 1) for b, v in res.items()})


def sample_logits(lg, seen, u):
    """Same operations, same order, as the HF processors in _sampler: temperature, top-k, top-p,
    then the repetition penalty on seen tokens. Gumbel-max over the result is a draw from softmax."""
    lg = lg / TEMP
    kth = torch.topk(lg, TOPK, dim=-1).values[..., -1:]
    lg = lg.masked_fill(lg < kth, float("-inf"))
    sa, ia = torch.sort(lg, dim=-1, descending=False)
    ca = torch.softmax(sa, dim=-1).cumsum(-1)
    rem = ca <= (1 - TOPP)
    rem[..., -1:] = False
    lg = torch.full_like(lg, float("-inf")).scatter(-1, ia, sa.masked_fill(rem, float("-inf")))
    lg = torch.where(seen, torch.where(lg < 0, lg * PEN, lg / PEN), lg)
    return torch.argmax(lg - torch.log(-torch.log(u)), dim=-1, keepdim=True)


@torch.inference_mode()
def exp_sampling(fast):
    cache, logits, L, sst = start_state(fast, TEXTS["short"])
    V = fast.t3.speech_head.weight.shape[0]
    procs = fast._sampler(TEMP, TOPK, TOPP, PEN)
    ar = torch.arange(V, device=DEV)

    # --- distribution check: host HF processors against the in-graph sampler, same logits
    N, CH = 20000, 4000
    seen_vec = (ar == sst).unsqueeze(0)  # (1, V), as the in-graph sampler expects
    hist = torch.tensor([[sst]], device=DEV)
    host_counts, graph_counts, ctrl_counts = (torch.zeros(V, device=DEV) for _ in range(3))
    lg0 = logits.float()
    with torch.inference_mode():
        for _ in range(N // CH):
            lg_b = lg0.expand(CH, -1).contiguous()
            h_b = hist.expand(CH, -1).contiguous()
            host_counts += torch.bincount(torch.multinomial(F.softmax(procs(h_b, lg_b.clone()), dim=-1), 1)
                                          .view(-1), minlength=V).float()
            ctrl_counts += torch.bincount(torch.multinomial(F.softmax(procs(h_b, lg_b.clone()), dim=-1), 1)
                                          .view(-1), minlength=V).float()
            u = torch.rand(CH, V, device=DEV).clamp(1e-10, 1 - 1e-10)
            seen_b = seen_vec.expand(CH, -1)
            graph_counts += torch.bincount(sample_logits(lg_b, seen_b, u).view(-1), minlength=V).float()
    ph, pg, pc = (c / c.sum() for c in (host_counts, graph_counts, ctrl_counts))
    tv = lambda p, q: float(0.5 * (p - q).abs().sum())  # noqa: E731
    record("sampling_distribution", n=N, tv_host_vs_graph=round(tv(ph, pg), 4),
           tv_host_vs_host_control=round(tv(ph, pc), 4), support_host=int((host_counts > 0).sum()))

    # --- speed: host loop (compiled step, host sampling) against one compiled step with sampling inside
    steps = 100
    pos_t = torch.tensor([L], device=DEV)
    tok_h = torch.tensor([[sst]], device=DEV)
    hist_h = torch.tensor([[sst]], device=DEV)
    cache_h, _, _, _ = start_state(fast, TEXTS["short"])

    @torch.inference_mode()
    def host_loop():
        nonlocal tok_h, hist_h
        for i in range(steps):
            if TRACE_POS:
                print("TRACE host_loop step", i, "pos", int(pos_t), "tok", int(tok_h), flush=True)
            lg = fast.step(fast.t3.speech_emb(tok_h), cache_h, pos_t)
            lgp = procs(hist_h, lg).clone()
            lgp[:, SPEECH_STOP] = float("-inf")  # the loop never ends, so a stop token is never fed back in
            nxt = torch.multinomial(F.softmax(lgp, dim=-1), 1)
            tok_h = nxt
            hist_h = torch.cat([hist_h, nxt], dim=1)
            pos_t.add_(1)
        pos_t.fill_(L)
        rewind(cache_h, L)
        hist_h = torch.tensor([[sst]], device=DEV)
    host_t = med_time(host_loop, n=3, warm=1) / steps

    def fused(tok, cache, pos, seen):
        torch.compiler.cudagraph_mark_step_begin()
        emb = fast.t3.speech_emb(tok)
        out = fast.tfmr(inputs_embeds=emb, past_key_values=cache, cache_position=pos, use_cache=True)
        lg = F.linear(out[0][:, -1:].float(), fast._head_weight_f32, fast._head_bias_f32)[:, -1, :]
        u = torch.rand(lg.shape, device=lg.device, dtype=lg.dtype).clamp(1e-10, 1 - 1e-10)
        lg = lg.masked_fill(ar == SPEECH_STOP, float("-inf"))  # as in host_loop: no stop feedback in a timed step
        nxt = sample_logits(lg, seen, u)
        return nxt.clone(), (seen | (ar == nxt)).clone()
    fused_c = torch.compile(fused, mode="max-autotune", fullgraph=False)
    cache_g, _, _, _ = start_state(fast, TEXTS["short"])
    state = {"tok": torch.tensor([[sst]], device=DEV), "seen": seen_vec.clone()}
    pos_g = torch.tensor([L], device=DEV)

    @torch.inference_mode()
    def graph_loop():
        for i in range(steps):
            nxt, seen_new = fused_c(state["tok"], cache_g, pos_g, state["seen"])
            # a graph output is overwritten by the next replay, so it is copied before it becomes an input
            state["tok"], state["seen"] = nxt.clone(), seen_new.clone()
            pos_g.add_(1)
        pos_g.fill_(L)
        rewind(cache_g, L)
        state["seen"] = seen_vec.clone()
    t0 = time.perf_counter()
    graph_loop()
    torch.cuda.synchronize()
    compile_s = round(time.perf_counter() - t0, 1)
    graph_t = med_time(graph_loop, n=3, warm=0) / steps
    record("sampling_speed", host_ms_per_token=round(host_t * 1000, 3),
           in_graph_ms_per_token=round(graph_t * 1000, 3),
           speedup=round(host_t / graph_t, 3), first_run_incl_compile_s=compile_s)


@torch.inference_mode()
def exp_vocoder(base, fast):
    """Where the time goes after the tokens: the flow stage (mel) and HiFT (audio). The mel is captured
    from the production call itself, so the stages are exactly what generate() runs."""
    from chatterbox.tts_turbo import S3GEN_SIL
    toks = fast.speech_tokens(TEXTS["short"])
    x = toks[toks < 6561].to(DEV)
    speech = torch.cat([x, torch.tensor([S3GEN_SIL] * 3, device=DEV, dtype=x.dtype)]).unsqueeze(0)
    ref = base.conds.gen

    def full():
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            return base.s3gen.inference(speech_tokens=speech, ref_dict=ref, n_cfm_timesteps=N_CFM)[0]

    captured = {}
    orig_hift = base.s3gen.hift_inference

    def spy(mel, *a, **k):
        captured["mel"] = mel.detach().clone()
        return orig_hift(mel, *a, **k)
    base.s3gen.hift_inference = spy
    try:
        full()
    finally:
        del base.s3gen.hift_inference
    mel = captured["mel"]

    def hift():
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            return orig_hift(mel.to(dtype=base.s3gen.dtype))[0]

    full_t, hift_t = med_time(full, n=5, warm=2), med_time(hift, n=5, warm=2)
    record("vocoder_split", mel_frames=int(mel.shape[-1]), full_ms=round(full_t * 1000, 1),
           flow_ms_by_difference=round((full_t - hift_t) * 1000, 1), hift_ms=round(hift_t * 1000, 1))

    torch.backends.cudnn.benchmark = True
    try:
        hift_bench = med_time(hift, n=5, warm=3)
    finally:
        torch.backends.cudnn.benchmark = False
    record("vocoder_cudnn_benchmark", hift_ms=round(hift_bench * 1000, 1), hift_ms_default=round(hift_t * 1000, 1))

    torch._dynamo.reset()
    hift_c = torch.compile(lambda m: orig_hift(m)[0], dynamic=True)

    def hift_compiled():
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            return hift_c(mel.to(dtype=base.s3gen.dtype))
    t0 = time.perf_counter()
    hift_compiled()
    torch.cuda.synchronize()
    first = round(time.perf_counter() - t0, 1)
    compiled_t = med_time(hift_compiled, n=5, warm=2)
    record("vocoder_hift_compiled", first_call_s=first, hift_ms=round(compiled_t * 1000, 1),
           speedup_vs_eager=round(hift_t / compiled_t, 3))
    torch._dynamo.reset()



@torch.inference_mode()
def exp_flow_decoder(base, fast):
    """The flow decoder (the CFM estimator), most of the vocoder time: capture its production arguments
    from one call, then time it eager and compiled on those same arguments."""
    from chatterbox.tts_turbo import S3GEN_SIL
    toks = fast.speech_tokens(TEXTS["short"])
    x = toks[toks < 6561].to(DEV)
    speech = torch.cat([x, torch.tensor([S3GEN_SIL] * 3, device=DEV, dtype=x.dtype)]).unsqueeze(0)
    ref = base.conds.gen

    def full():
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            return base.s3gen.inference(speech_tokens=speech, ref_dict=ref, n_cfm_timesteps=N_CFM)[0]
    dec = base.s3gen.flow.decoder
    orig_dec = dec.forward
    dec_call = {}

    def dec_spy(*a, **k):
        dec_call["a"], dec_call["k"] = a, k
        return orig_dec(*a, **k)
    dec.forward = dec_spy
    try:
        full()
    finally:
        del dec.forward

    def run_dec(fn):
        torch.manual_seed(0)  # the flow noise is drawn inside the decoder
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
            return fn(*dec_call["a"], **dec_call["k"])[0]
    eager_out = run_dec(orig_dec)
    eager_t = med_time(lambda: run_dec(orig_dec), n=5, warm=2)
    torch._dynamo.reset()
    dec_c = torch.compile(orig_dec, dynamic=True)
    t0 = time.perf_counter()
    run_dec(dec_c)
    torch.cuda.synchronize()
    dec_first = round(time.perf_counter() - t0, 1)
    comp_out = run_dec(dec_c)
    comp_t = med_time(lambda: run_dec(dec_c), n=5, warm=2)
    record("vocoder_flow_decoder", eager_ms=round(eager_t * 1000, 1), compiled_ms=round(comp_t * 1000, 1),
           compiled_first_call_s=dec_first, speedup=round(eager_t / comp_t, 3),
           max_abs_mel_diff=round(float((eager_out.float() - comp_out.float()).abs().max()), 5))
    torch._dynamo.reset()


@torch.inference_mode()
def exp_speculative(base, fast):
    """Self-speculative decoding, feasibility only: a shallow prefix of the same network (its hidden
    state after layer d, through the final norm and head) as the draft. Acceptance is the mean overlap
    of draft and full-model distributions under the production sampler, measured teacher-forced on the
    reference reply. The speedup estimate uses the step and verify costs measured earlier in this run."""
    ref = REF_TOKENS["short"]
    sst = fast.t3.hp.start_speech_token
    text = TEXTS["short"]
    ids = fast.m.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(DEV)
    speech = torch.tensor([[sst] + ref], device=DEV)
    emb, _ = fast.t3.prepare_input_embeds(t3_cond=base.conds.t3, text_tokens=ids, speech_tokens=speech,
                                          cfg_weight=0.0)
    T, N = emb.shape[1], len(ref)
    start = T - (N + 1)
    out = fast.tfmr(inputs_embeds=emb, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    procs = fast._sampler(TEMP, TOPK, TOPP, 1.0)
    dummy = torch.zeros(N, 1, dtype=torch.long, device=DEV)

    def probs(hidden):
        # the HF processors take (batch, vocab): a 3-D input makes their scatter index the wrong axis
        lg = F.linear(hidden[0].float(), fast._head_weight_f32, fast._head_bias_f32)
        return F.softmax(procs(dummy, lg), dim=-1)
    full = probs(hs[-1][:, start:start + N])
    depths = (4, 6, 8, 10)
    acc = {}
    for d in depths:
        h = fast.tfmr.ln_f(hs[d][:, start:start + N])
        draft = probs(h)
        acc[d] = float(torch.minimum(full, draft).sum(-1).mean())
    step_ms = DATA["experiments"].get("breakdown", {}).get("device_step_ms")
    record("speculative_acceptance", **{f"alpha_depth{d}": round(a, 4) for d, a in acc.items()},
           positions=N, step_ms_measured=step_ms)


@torch.inference_mode()
def exp_modes(fast):
    cache, logits, L, sst = start_state(fast, TEXTS["short"])
    tok = torch.tensor([[sst]], device=DEV)
    pos = torch.tensor([L], device=DEV)
    modes = ["max-autotune"] if args.quick else ["default", "reduce-overhead", "max-autotune"]
    res = {}
    with torch.inference_mode():
        for mode in modes:
            torch._dynamo.reset()
            f = torch.compile(fast._fp16_step, mode=mode, fullgraph=False)
            emb = fast.t3.speech_emb(tok)
            t0 = time.perf_counter()
            rewind(cache, L)
            f(emb, cache, pos)
            torch.cuda.synchronize()
            first = round(time.perf_counter() - t0, 1)
            step = med_time(lambda: (rewind(cache, L), f(emb, cache, pos)), n=20, warm=3) * 1000
            res[mode] = {"first_call_s": first, "step_ms": round(step, 3)}
            del f
            release()
    record("compile_modes", **res)


@torch.inference_mode()
def conv1d_to_linear(model):
    # GPT-2 projections are transformers' Conv1D (weight stored as (in, out)). torchao quantises only nn.Linear,
    # so without this conversion quantize_ changes nothing: the first Kaggle run's int8/int4 were no-ops.
    from transformers.pytorch_utils import Conv1D
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, Conv1D):
                nin, nout = child.weight.shape
                lin = torch.nn.Linear(nin, nout, bias=child.bias is not None,
                                      device=child.weight.device, dtype=child.weight.dtype)
                lin.weight.data.copy_(child.weight.data.t())
                if child.bias is not None:
                    lin.bias.data.copy_(child.bias.data)
                setattr(mod, name, lin)
    return model


def exp_quant(scheme):
    # "linear" is the control: the same conversion with no quantisation, which should reproduce the default
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    base, fast = load()
    conv1d_to_linear(fast.tfmr)
    if scheme != "linear":
        cfg = Int8WeightOnlyConfig() if scheme == "int8" else Int4WeightOnlyConfig(group_size=128)
        quantize_(fast.tfmr, cfg)  # the T3 decoder's projections only; embeddings and head stay as they are
    n_quant = sum(type(m.weight.data).__name__ != "Tensor" for m in fast.tfmr.modules()
                  if isinstance(m, torch.nn.Linear))
    if scheme != "linear":
        assert n_quant > 0, "quantize_ changed no layer"
    with torch.inference_mode():
        torch.manual_seed(0)
        toks = fast.speech_tokens(TEXTS["short"]).flatten().tolist()
        torch.manual_seed(0)
        w = fast.generate(TEXTS["short"]).squeeze(0).float().cpu().numpy()
        def run_once():
            torch.manual_seed(0)
            fast.generate(TEXTS["short"])
        t = med_time(run_once, n=5, warm=2)
    audio_s = len(w) / 24000
    n = min(len(w), len(REF_AUDIO["short"]))
    record(f"quant_{scheme}", quantized_layers=n_quant, generate_s=round(t, 3), rtf=round(t / audio_s, 4),
           tokens_identical_to_fp16_default=toks == REF_TOKENS["short"],
           first_diff=next((i for i, (a, b) in enumerate(zip(toks, REF_TOKENS["short"])) if a != b), None),
           lsd_db_vs_default=round(lsd_db(REF_AUDIO["short"][:n], w[:n]), 3), peak_mb=peak_mb())


# ---------------------------------------------------------------- driver

def main():
    run("env", exp_env)
    base, fast = None, None
    model_names = {"baseline", "attention", "breakdown", "vocoder", "cache", "verify", "batch",
                   "sampling", "speculative", "modes", "probe"}
    if not ONLY or ONLY & model_names:
        base, fast = load()
    if fast is not None and os.environ.get("CHATNANO_EAGER_STEP") == "1":
        fast.step = fast._step  # diagnostic: the decode step without torch.compile, to test the compiled path
    if fast is not None:
        run("baseline", exp_baseline, fast)
        run("probe", exp_probe, fast)
        run("attention", exp_attention, fast)
        run("breakdown", exp_breakdown, fast)
        run("vocoder", exp_vocoder, base, fast)
        run("cache", exp_cache_curve, fast)
        run("verify", exp_verify, fast)
        run("batch", exp_batch, fast)
        run("sampling", exp_sampling, fast)
        run("modes", exp_modes, fast)
        run("speculative", exp_speculative, base, fast)
        del fast, base
        release()
    if not args.quick:
        run("quant_linear", exp_quant, "linear")
        run("quant_int8", exp_quant, "int8")
        run("quant_int4", exp_quant, "int4")
    record("done", finished=True)


if __name__ == "__main__":
    main()
