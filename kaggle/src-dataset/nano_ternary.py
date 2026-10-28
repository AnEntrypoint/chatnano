import copy
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from chatterbox.models.s3gen.const import S3GEN_SIL, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from ternary_qat import TernaryLinear, dequantize, export_metadata, export_packed, replace_linears, set_alpha

START_SPEECH = 6561
STOP_SPEECH = 6562
SPEECH_CODE_LIMIT = 6561
ASR_SAMPLE_RATE = 16_000

CANDIDATE_RE = re.compile(r"^h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$")


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str), flush=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_nano(device, model_dir=None):
    if model_dir:
        return ChatterboxTurboTTS.from_local(model_dir, device, nano=True)
    return ChatterboxTurboTTS.from_pretrained(device=device, nano=True)


def conv1d_to_linear(conv):
    weight = conv.weight.detach()
    linear = nn.Linear(
        weight.shape[0], weight.shape[1], bias=conv.bias is not None, device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        linear.weight.copy_(weight.t())
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)
    return linear


def linearize_conv1d(module):
    for name, child in list(module.named_modules()):
        if isinstance(child, Conv1D):
            parent_name, _, child_name = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, child_name, conv1d_to_linear(child))
    return module


def make_student(pristine, device):
    student = copy.deepcopy(pristine)
    linearize_conv1d(student.tfmr)
    names = replace_linears(student.tfmr, predicate=lambda name, _module: CANDIDATE_RE.match(name) is not None)
    student.to(device)
    return student, names


def freeze_except_latents(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    latents = [m for m in model.modules() if isinstance(m, TernaryLinear)]
    for module in latents:
        module.weight.requires_grad_(True)
    return [module.weight for module in latents]


def make_cond(speaker_emb, prompt_tokens, device):
    return T3Cond(
        speaker_emb=speaker_emb.to(device=device, dtype=torch.float32),
        cond_prompt_speech_tokens=prompt_tokens.to(device=device, dtype=torch.long),
    )


def builtin_cond(nano, device):
    return make_cond(nano.conds.t3.speaker_emb, nano.conds.t3.cond_prompt_speech_tokens, device)


def text_ids(nano, text, device):
    return nano.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids.to(device)


def teacher_forced_inputs(nano, row, cond, device):
    text = text_ids(nano, row["text"], device)
    speech = torch.tensor([[START_SPEECH, *row["speech_tokens"], STOP_SPEECH]], dtype=torch.long, device=device)
    return dict(
        t3_cond=cond,
        text_tokens=text,
        text_token_lens=torch.tensor([text.shape[1]], dtype=torch.long, device=device),
        speech_tokens=speech,
        speech_token_lens=torch.tensor([speech.shape[1]], dtype=torch.long, device=device),
    )


def speech_logits(model, inputs):
    embeds, len_cond = model.prepare_input_embeds(
        t3_cond=inputs["t3_cond"],
        text_tokens=inputs["text_tokens"],
        speech_tokens=inputs["speech_tokens"],
    )
    hidden = model.tfmr(inputs_embeds=embeds, use_cache=False).last_hidden_state
    start = len_cond + inputs["text_tokens"].shape[1]
    return model.speech_head(hidden[:, start : start + inputs["speech_tokens"].shape[1]])


def shifted_pairs(logits, speech_tokens):
    vocab = logits.shape[-1]
    return logits[:, :-1].float().reshape(-1, vocab), speech_tokens[:, 1:].reshape(-1)


def distill_loss(student, teacher, inputs, w_ce, w_kd, autocast):
    with torch.no_grad(), autocast:
        teacher_logits = speech_logits(teacher, inputs)
    with autocast:
        student_logits = speech_logits(student, inputs)
    student_flat, targets = shifted_pairs(student_logits, inputs["speech_tokens"])
    teacher_flat, _ = shifted_pairs(teacher_logits, inputs["speech_tokens"])
    ce = F.cross_entropy(student_flat, targets)
    kd = F.kl_div(
        F.log_softmax(student_flat, dim=-1),
        F.log_softmax(teacher_flat, dim=-1),
        log_target=True,
        reduction="batchmean",
    )
    return w_ce * ce + w_kd * kd, {"ce": float(ce.detach()), "kd": float(kd.detach())}


@torch.no_grad()
def teacher_forced_metrics(model, nano, rows, cond, device, autocast):
    nll_total, correct, count = 0.0, 0, 0
    for row in rows:
        inputs = teacher_forced_inputs(nano, row, cond, device)
        with autocast:
            logits = speech_logits(model, inputs)
        flat, targets = shifted_pairs(logits, inputs["speech_tokens"])
        nll_total += float(F.cross_entropy(flat, targets, reduction="sum"))
        correct += int((flat.argmax(dim=-1) == targets).sum())
        count += int(targets.numel())
    return {"nll": nll_total / count, "top1": correct / count, "tokens": count}


@torch.no_grad()
def generate_speech_tokens(model, nano, text, device, autocast, max_gen_len, seed):
    torch.manual_seed(seed)
    with autocast:
        tokens = model.inference_turbo(
            t3_cond=builtin_cond(nano, device),
            text_tokens=text_ids(nano, text, device),
            max_gen_len=max_gen_len,
        )
    return tokens[0]


def synthesize(nano, tokens, device):
    tokens = tokens[tokens < SPEECH_CODE_LIMIT]
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long, device=device)
    tokens = torch.cat([tokens, silence]).unsqueeze(0)
    wav, _ = nano.s3gen.inference(speech_tokens=tokens, ref_dict=nano.conds.gen, n_cfm_timesteps=2)
    return wav.squeeze(0).float().cpu().numpy()


def normalize_for_wer(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def transcribe(asr, wav):
    audio = librosa.resample(wav, orig_sr=S3GEN_SR, target_sr=ASR_SAMPLE_RATE)
    return asr({"raw": audio, "sampling_rate": ASR_SAMPLE_RATE})["text"]


def export_blob(student, meta, step):
    alpha = float(next(m.alpha for m in student.modules() if isinstance(m, TernaryLinear)))
    backbone = student.tfmr
    set_alpha(backbone, 1.0)
    try:
        packed = {name: (codes.cpu(), scales.cpu()) for name, (codes, scales) in export_packed(backbone).items()}
        layers = {}
        for name, info in export_metadata(backbone).items():
            bias = info["bias"]
            layers[name] = {**info, "bias": None if bias is None else bias.cpu()}
    finally:
        set_alpha(backbone, alpha)
    return {"packed": packed, "layers": layers, "meta": {**meta, "step": step}}


def load_packed_student(pristine, blob, device):
    model = copy.deepcopy(pristine)
    linearize_conv1d(model.tfmr)
    for name, (codes, scales) in blob["packed"].items():
        info = blob["layers"][name]
        parent_name, _, child_name = name.rpartition(".")
        parent = model.tfmr.get_submodule(parent_name) if parent_name else model.tfmr
        replacement = nn.Linear(info["in_features"], info["out_features"], bias=info["bias"] is not None)
        with torch.no_grad():
            replacement.weight.copy_(dequantize(codes, scales, info["group_size"]))
            if info["bias"] is not None:
                replacement.bias.copy_(info["bias"])
        setattr(parent, child_name, replacement)
    return model.to(device).eval()


@torch.no_grad()
def reload_agreement(student, reloaded, nano, row, cond, device):
    inputs = teacher_forced_inputs(nano, row, cond, device)
    expected = speech_logits(student, inputs).float()
    actual = speech_logits(reloaded, inputs).float()
    max_abs = float((expected - actual).abs().max())
    agree = float((expected.argmax(-1) == actual.argmax(-1)).float().mean())
    return {"max_abs_logit_diff": max_abs, "argmax_agreement": agree}
