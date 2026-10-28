import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import soundfile as sf
import torch

from chatterbox.models.s3tokenizer import S3_SR
from nano_ternary import load_nano, log

ENC_COND_LEN = 15 * S3_SR
PROMPT_TOKENS = 375
MIN_SECONDS = 1.0
MAX_SECONDS = 20.0
REF_MIN_SECONDS = 6.0

SMOKE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please keep the receipt for your records.",
    "Turn left at the second traffic light.",
    "The meeting has been moved to Thursday afternoon.",
    "Our train departs from platform four in ten minutes.",
    "She painted the fence a bright shade of blue.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The library closes early on public holidays.",
    "Add two cups of flour and stir gently.",
    "The forecast calls for light rain tomorrow morning.",
    "He finally finished reading the long novel last night.",
    "Each student must submit the form before Friday.",
]


def read_metadata(root):
    with open(root / "metadata.csv", encoding="utf-8", newline="") as handle:
        for record in csv.reader(handle, delimiter="|"):
            if len(record) < 2:
                continue
            text = record[2] if len(record) > 2 and record[2].strip() else record[1]
            yield record[0], text.strip()


def make_smoke_corpus(nano, root):
    (root / "wavs").mkdir(parents=True, exist_ok=True)
    lines = []
    for index, sentence in enumerate(SMOKE_SENTENCES):
        utt_id = f"smoke{index:03d}"
        wav = nano.generate(sentence)
        sf.write(root / "wavs" / f"{utt_id}.wav", wav.squeeze(0).float().cpu().numpy(), nano.sr)
        lines.append(f"{utt_id}|{sentence}|{sentence}")
        log("smoke_utterance", id=utt_id)
    (root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reference_condition(nano, wav):
    speaker = torch.from_numpy(nano.ve.embeds_from_wavs([wav], sample_rate=S3_SR)).mean(axis=0, keepdim=True)
    prompt, _ = nano.s3gen.tokenizer.forward([wav[:ENC_COND_LEN]], max_len=PROMPT_TOKENS)
    return {
        "speaker_emb": speaker.float().cpu(),
        "prompt_tokens": torch.atleast_2d(prompt).long().cpu(),
    }


def tokenize_items(nano, items, root, ref_target, refs):
    rows = []
    for utt_id, text in items:
        wav = librosa.load(root / "wavs" / f"{utt_id}.wav", sr=S3_SR)[0]
        seconds = len(wav) / S3_SR
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            continue
        tokens, _ = nano.s3gen.tokenizer.forward([wav])
        rows.append({"id": utt_id, "text": text, "seconds": round(seconds, 3), "speech_tokens": tokens[0].tolist()})
        if len(refs) < ref_target and seconds >= REF_MIN_SECONDS:
            refs.append(reference_condition(nano, wav))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="directory with wavs/ and metadata.csv (LJSpeech layout)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-count", type=int, default=200)
    parser.add_argument("--ref-count", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N utterances after shuffling (0 = all)")
    parser.add_argument("--make-smoke-corpus", action="store_true", help="synthesize SMOKE_SENTENCES with the teacher first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default=None, help="local Chatterbox Nano snapshot (skips the Hugging Face download)")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nano = load_nano(args.device, args.model_dir)
    if args.make_smoke_corpus:
        corpus.mkdir(parents=True, exist_ok=True)
        make_smoke_corpus(nano, corpus)

    items = list(read_metadata(corpus))
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    eval_items = items[: args.eval_count]
    train_items = items[args.eval_count :]

    refs = []
    train_rows = tokenize_items(nano, train_items, corpus, args.ref_count, refs)
    eval_rows = tokenize_items(nano, eval_items, corpus, 0, [])
    refs.append({
        "speaker_emb": nano.conds.t3.speaker_emb.detach().float().cpu(),
        "prompt_tokens": nano.conds.t3.cond_prompt_speech_tokens.detach().long().cpu(),
    })

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    torch.save(refs, out / "refs.pt")
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "refs": len(refs),
        "train_seconds": round(sum(r["seconds"] for r in train_rows), 1),
        "eval_seconds": round(sum(r["seconds"] for r in eval_rows), 1),
        "train_tokens": sum(len(r["speech_tokens"]) for r in train_rows),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("prepared", **summary)


if __name__ == "__main__":
    main()
