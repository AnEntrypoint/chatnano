"""Resident Chatterbox-Nano server.

The model loads and warms up once at startup (graphs compiled for the common shapes). Each line of the
text is a paragraph. A paragraph longer than 72 words is split at sentence ends into chunks of at most 72
words, and the chunks are rendered and joined by concatenation. Each render's audio is sent as float32
little-endian mono PCM at 24 kHz in 250 ms blocks as soon as it is ready, over HTTP chunked transfer, with
a short pause after each paragraph. Text with no line breaks and no long paragraph renders in one pass, as
before. encoding=int16 sends signed 16-bit samples instead, half the bytes.

    python serve.py --port 8765                 # one paragraph per line, sent as each is rendered
    curl -N "http://127.0.0.1:8765/tts?text=Hello%20there&seed=0" -o out.f32   # seed optional
    curl -N "http://127.0.0.1:8765/tts?text=Hello%20there&encoding=int16" -o out.i16

Requests share one GPU, so generation runs one at a time under a lock.
"""
import argparse
import os
import re
import sys
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# keep compiled kernels across restarts: the default inductor cache lives under /tmp, which
# is wiped, so each cold boot would pay the full compile again
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/chatnano/inductor"))

import torch

sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

WARMUP_TEXT = "Hello there, this warms the compiled decode step."
BLOCK_SAMPLES = 6000  # 250 ms at 24 kHz: send size for whole-reply streaming
MAX_WORDS = 72  # longest chunk the model completes (measured, see too_long)
PARAGRAPH_PAUSE_S = 0.4  # silence between paragraphs: a natural break, and no abrupt join
# the sentence split of the sentence chunking whose joins passed the listening check (4dbaed4)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# half-precision weight directory written by export_fp16.py; the hub download is the fallback
WEIGHTS_DIR = os.environ.get("CHATNANO_WEIGHTS", "/workspace/chatsrc/chatnano-fp16")


def paragraphs(text):
    # one paragraph per line; blank lines are dropped
    return [p.strip() for p in text.split("\n") if p.strip()]


def chunks(paragraph):
    # sentences grouped into chunks of at most MAX_WORDS words. A chunk is rendered as one unit and units are
    # joined by concatenation, as sentence chunks were. A sentence longer than MAX_WORDS stays one chunk.
    out, cur, words = [], [], 0
    for s in (s.strip() for s in SENTENCE_END.split(paragraph)):
        if not s:
            continue
        n = len(s.split())
        if cur and words + n > MAX_WORDS:
            out.append(" ".join(cur))
            cur, words = [], 0
        cur.append(s)
        words += n
    if cur:
        out.append(" ".join(cur))
    return out


def unit_seed(seed, n):
    # unit 0 uses the request's seed, so a one-chunk reply is unchanged; later units get their own seed from it.
    # Re-seeding with manual_seed (not restoring a saved state) keeps CUDA graph capture valid.
    return seed if n == 0 else (seed + n * 1_000_003) % (2 ** 63)


class Server:
    def __init__(self):
        if os.path.isdir(WEIGHTS_DIR):
            self.model = ChatterboxTurboTTS.from_local(WEIGHTS_DIR, device="cuda", nano=True)
        else:
            self.model = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
        self.fast = FastNano(self.model)
        self.lock = threading.RLock()  # warm-up calls pieces() while holding it
        with self.lock:  # warm-up: compile the decode step and the vocoder path before serving
            for _ in self.pieces(WARMUP_TEXT):
                pass
        self.sr = self.model.sr

    def too_long(self, text):
        # Nano ends a long render early: four sentences (72 words) measure 0.99 of the expected length,
        # 8 sentences (144 words) 0.84, 16 sentences 0.14 (nano_fast/bench/length_probe.py). Long paragraphs are
        # split into chunks of at most 72 words, so only a single sentence over the limit is refused.
        return any(len(c.split()) > MAX_WORDS for p in paragraphs(text) for c in chunks(p))

    def pieces(self, text, seed=None):
        # the lock covers each render only, never the socket writes: a client that stops reading must not
        # hold it and stall every other request. Each unit is seeded from the request seed and its index,
        # so a seeded reply is the same whatever other requests render in between. A pause follows each
        # paragraph break; the joins inside a paragraph are plain concatenation.
        paras = paragraphs(text) or [text]
        units = []
        for p_index, p in enumerate(paras):
            cs = chunks(p)
            for c_index, c in enumerate(cs):
                last = p_index < len(paras) - 1 and c_index == len(cs) - 1
                units.append((c, last))
        pause = np.zeros(int(PARAGRAPH_PAUSE_S * self.model.sr), dtype=np.float32)
        for n, (unit, pause_after) in enumerate(units):
            with self.lock:
                with torch.inference_mode():
                    if seed is not None:
                        torch.manual_seed(unit_seed(seed, n))  # reproducible, for comparison with a single-pass render
                    audio = self.fast.generate(unit).squeeze(0).float().cpu().numpy()
            for i in range(0, len(audio), BLOCK_SAMPLES):
                yield audio[i:i + BLOCK_SAMPLES]
            if pause_after:
                yield pause


def make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path != "/tts":
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(url.query)
            encoding = query.get("encoding", ["float32"])[0]
            if encoding not in ("float32", "int16"):
                self.send_error(400, "encoding must be float32 or int16")
                return
            text = query.get("text", [""])[0].strip()
            if not text:
                self.send_error(400, "missing text")
                return
            try:
                seed = int(query["seed"][0]) if "seed" in query else None
            except ValueError:
                self.send_error(400, "seed must be an integer")
                return
            # int() takes arbitrary precision; torch.manual_seed rejects anything outside int64, and
            # it would raise after the 200 headers are already flushed
            if seed is not None and not (-(2 ** 63) <= seed < 2 ** 63):
                self.send_error(400, "seed must fit in a signed 64-bit integer")
                return
            if server.too_long(text):
                self.send_error(413, "a sentence is longer than 72 words; the model ends longer sentences early. Add a full stop inside it")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(server.sr))
            self.send_header("X-Encoding", "int16le" if encoding == "int16" else "float32le")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for piece in server.pieces(text, seed):
                    if encoding == "int16":
                        data = (piece.clip(-1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
                    else:
                        data = piece.astype("<f4").tobytes()
                    if data:
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except BrokenPipeError:
                pass  # client went away after the reply was rendered; nothing else is left to send
            except Exception:
                # the 200 headers are already flushed, so the stream can only be closed off
                traceback.print_exc()

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = Server()
    # bind before announcing readiness: a port already in use must fail the start, not look ready
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"ready on {args.host}:{args.port} sr={server.sr}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
