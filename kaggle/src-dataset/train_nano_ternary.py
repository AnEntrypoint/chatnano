import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch

from nano_ternary import (
    builtin_cond,
    distill_loss,
    export_blob,
    freeze_except_latents,
    generate_speech_tokens,
    load_jsonl,
    load_nano,
    load_packed_student,
    log,
    make_cond,
    make_student,
    reload_agreement,
    synthesize,
    teacher_forced_inputs,
    teacher_forced_metrics,
    transcribe,
    normalize_for_wer,
)
from ternary_qat import TernaryLinear, clip_latents, ramp_alpha, set_alpha


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory from prepare_tokens.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8, help="utterances per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--ramp-frac", type=float, default=0.25)
    parser.add_argument("--clip", type=float, default=1.0, help="clamp latents to [-clip, clip] after each step; 0 disables")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--w-ce", type=float, default=1.0)
    parser.add_argument("--w-kd", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-count", type=int, default=0, help="held-out utterances for teacher-forced metrics (0 = all)")
    parser.add_argument("--gen-count", type=int, default=20, help="held-out utterances for free-running generation stats")
    parser.add_argument("--max-gen-len", type=int, default=400)
    parser.add_argument("--wer-count", type=int, default=0, help="utterances to synthesize and ASR for WER (0 = skip)")
    parser.add_argument("--asr-model", default="openai/whisper-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default=None, help="local Chatterbox Nano snapshot (skips the Hugging Face download)")
    return parser.parse_args(argv)


def save_checkpoint(student, latent_modules, optimizer, scaler, rng, out_dir, step, meta, keep_last):
    blob = export_blob(student, meta, step)
    directory = out_dir / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(blob, directory / "packed.pt")
    torch.save(
        {
            "step": step,
            "latents": {name: module.weight.detach().cpu() for name, module in latent_modules.items()},
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_state": rng.getstate(),
        },
        out_dir / "resume.pt",
    )
    steps = sorted(
        (path for path in out_dir.glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_", 1)[1]),
    )
    pruned = []
    for old in steps[: max(0, len(steps) - keep_last)]:
        shutil.rmtree(old)
        pruned.append(old.name)
    log("checkpoint", step=step, packed=str(directory / "packed.pt"), pruned=pruned)


def load_resume(path, latent_modules, optimizer, scaler, rng):
    state = torch.load(path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for name, module in latent_modules.items():
            module.weight.copy_(state["latents"][name].to(module.weight.device))
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    rng.setstate(state["rng_state"])
    return int(state["step"])


def evaluate(args, nano, teacher, student, pristine, eval_rows, device, autocast, meta, out_dir):
    cond = builtin_cond(nano, device)
    teacher.eval()
    student.eval()
    set_alpha(student, 1.0)
    summary = {"teacher_forced": {}, "generation": {}, "export": {}}

    summary["teacher_forced"]["teacher"] = teacher_forced_metrics(teacher, nano, eval_rows, cond, device, autocast)
    summary["teacher_forced"]["student"] = teacher_forced_metrics(student, nano, eval_rows, cond, device, autocast)
    log("eval_teacher_forced", **summary["teacher_forced"])

    blob = export_blob(student, meta, meta["step"])
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(blob, final_dir / "packed.pt")
    reloaded = load_packed_student(pristine, blob, device)
    summary["export"] = reload_agreement(student, reloaded, nano, eval_rows[0], cond, device)
    log("export_check", **summary["export"])
    del reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gen_rows = eval_rows[: args.gen_count]
    stats = {"teacher": {"lengths": [], "hit_cap": 0}, "student": {"lengths": [], "hit_cap": 0}}
    outputs = {"teacher": [], "student": []}
    for index, row in enumerate(gen_rows):
        for label, model in (("teacher", teacher), ("student", student)):
            tokens = generate_speech_tokens(model, nano, row["text"], device, autocast, args.max_gen_len, seed=index)
            length = int(tokens.numel())
            stats[label]["lengths"].append(length)
            stats[label]["hit_cap"] += int(length >= args.max_gen_len)
            outputs[label].append(tokens)
    for label in stats:
        lengths = stats[label].pop("lengths")
        summary["generation"][label] = {
            "utterances": len(lengths),
            "mean_tokens": sum(lengths) / max(1, len(lengths)),
            "hit_cap_rate": stats[label]["hit_cap"] / max(1, len(lengths)),
        }
    log("eval_generation", **summary["generation"])

    if args.wer_count:
        from jiwer import wer
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=args.asr_model,
            device=0 if device.type == "cuda" else -1,
        )
        summary["wer"] = {}
        for label in ("teacher", "student"):
            errors = []
            for index, row in enumerate(gen_rows[: args.wer_count]):
                hypothesis = transcribe(asr, synthesize(nano, outputs[label][index], device))
                errors.append(wer(normalize_for_wer(row["text"]), normalize_for_wer(hypothesis)))
            summary["wer"][label] = sum(errors) / len(errors)
        log("eval_wer", **summary["wer"])

    return summary


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    train_rows = load_jsonl(data / "train.jsonl")
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    eval_rows = load_jsonl(data / "eval.jsonl")
    if args.eval_count:
        eval_rows = eval_rows[: args.eval_count]
    refs = torch.load(data / "refs.pt", weights_only=True)

    nano = load_nano(args.device, args.model_dir)
    teacher = nano.t3.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    pristine = copy.deepcopy(teacher).cpu()

    student, candidate_names = make_student(pristine, device)
    student.eval()
    latents = freeze_except_latents(student)
    latent_modules = {name: module for name, module in student.named_modules() if isinstance(module, TernaryLinear)}
    optimizer = torch.optim.AdamW(latents, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    meta = {
        "model": "ResembleAI/chatterbox-nano",
        "backbone": "t3.tfmr",
        "candidates": candidate_names,
        "ramp_frac": args.ramp_frac,
        "total_steps": args.steps,
        "group_size": "per-output-row",
        "step": 0,
    }
    log(
        "setup",
        candidates=len(candidate_names),
        latent_params=sum(p.numel() for p in latents),
        frozen_params=sum(p.numel() for p in student.parameters() if not p.requires_grad),
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        refs=len(refs),
        device=str(device),
        amp=amp,
    )

    start_step = 0
    if args.resume:
        start_step = load_resume(args.resume, latent_modules, optimizer, scaler, rng)
        log("resumed", step=start_step, path=args.resume)

    clip_value = args.clip if args.clip > 0 else None
    loop_started = time.time()
    for step in range(start_step, args.steps):
        alpha = ramp_alpha(step, args.steps, args.ramp_frac)
        set_alpha(student, alpha)
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        term_totals = {}
        for _ in range(args.batch):
            row = rng.choice(train_rows)
            ref = rng.choice(refs)
            inputs = teacher_forced_inputs(nano, row, make_cond(ref["speaker_emb"], ref["prompt_tokens"], device), device)
            loss, terms = distill_loss(student, teacher, inputs, args.w_ce, args.w_kd, autocast)
            scaled = loss / args.batch
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            step_loss += float(loss.detach()) / args.batch
            for name, value in terms.items():
                term_totals[name] = term_totals.get(name, 0.0) + value / args.batch

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(latents, args.grad_clip))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        clip_latents(student, clip_value)
        optimizer.zero_grad(set_to_none=True)

        completed = step + 1
        if completed % args.log_every == 0 or completed == args.steps:
            fields = {
                "step": completed,
                "alpha": round(alpha, 4),
                "loss": round(step_loss, 5),
                "terms": {k: round(v, 5) for k, v in term_totals.items()},
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "seconds": round(time.time() - loop_started, 1),
            }
            if amp:
                fields["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            log("step", **fields)

        if completed % args.ckpt_every == 0 or completed == args.steps:
            meta["step"] = completed
            save_checkpoint(student, latent_modules, optimizer, scaler, rng, out_dir, completed, meta, args.keep_last)

    set_alpha(student, 1.0)
    meta["step"] = args.steps
    summary = {
        "args": vars(args),
        "candidates": len(candidate_names),
        "train_seconds": round(time.time() - started, 1),
    }
    summary.update(evaluate(args, nano, teacher, student, pristine, eval_rows, device, autocast, meta, out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log("done", summary_path=str(out_dir / "summary.json"), seconds=summary["train_seconds"])


if __name__ == "__main__":
    main()
