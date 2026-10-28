# Live measurement of serve.py as shipped: cold-after-ready, warm requests, concurrent requests.
# Arm A: one server, two requests at once (the GPU lock should serialise them).
# Arm B: two servers, one request each at once (independent processes, shared GPU).
import http.client, json, os, signal, subprocess, sys, threading, time, urllib.parse

PY = "/data/cache/venvs/ml/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.join(os.path.dirname(HERE), "serve.py")
TEXT_A = "Hello there, how are you doing today?"
TEXT_B = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
          "Would you like me to get some prices for you?")

def start(port, log):
    f = open(log, "w")
    p = subprocess.Popen([PY, "-I", SERVE, "--port", str(port)], stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    t0 = time.perf_counter()
    while True:
        if "ready on" in open(log).read():
            return p, time.perf_counter() - t0
        if p.poll() is not None:
            raise SystemExit(f"RESULT: FAIL server exited on port {port}; see {log}")
        time.sleep(0.5)

def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    p.wait(timeout=60)

def request(port, text):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    t0 = time.perf_counter()
    conn.request("GET", "/tts?text=" + urllib.parse.quote(text))
    resp = conn.getresponse()
    first, n = None, 0
    while True:
        c = resp.read1(65536)
        if not c:
            break
        if first is None:
            first = time.perf_counter() - t0
        n += len(c)
    total = time.perf_counter() - t0
    return {"status": resp.status, "sr": resp.getheader("X-Sample-Rate"), "first": first,
            "total": total, "bytes": n, "seconds": n / 4 / 24000}

def concurrent(jobs):
    out = [None] * len(jobs)
    def run(i, port, text):
        out[i] = request(port, text)
    ts = [threading.Thread(target=run, args=(i, port, text)) for i, (port, text) in enumerate(jobs)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    return out, time.perf_counter() - t0

if __name__ == "__main__":
    results = {}
    ok = True
    procs = []
    try:
        # arm A: one server, two requests at once. The server is stopped before arm B starts
        # so that no more than two servers share the 6 GB card at any time.
        pA, startA = start(8790, "/tmp/bench_a.log"); procs.append(pA)
        results["startup_to_ready_s"] = round(startA, 2)
        first = request(8790, TEXT_A)  # cold-after-ready: first request after the warm-up
        results["cold_after_ready"] = first
        for _ in range(2):
            results.setdefault("warm", []).append(request(8790, TEXT_B))
        armA, wallA = concurrent([(8790, TEXT_A), (8790, TEXT_B)])
        results["armA_one_server_two_at_once"] = {"wall": round(wallA, 2), "reqs": armA}
        stop(pA); procs.remove(pA)
        # arm B: two servers, one request each at once
        pB1, _ = start(8791, "/tmp/bench_b1.log"); procs.append(pB1)
        pB2, _ = start(8793, "/tmp/bench_b2.log"); procs.append(pB2)
        armB, wallB = concurrent([(8791, TEXT_A), (8793, TEXT_B)])
        results["armB_two_servers_one_each"] = {"wall": round(wallB, 2), "reqs": armB}
    finally:
        for p in procs:
            stop(p)
    print(json.dumps(results, indent=1, default=str))
    # guards: every reply is 200 at 24 kHz with whole 32-bit samples; the stream carried audio
    reqs = [first] + results["warm"] + armA + armB
    for r in reqs:
        if r["status"] != 200 or r["sr"] != "24000" or r["bytes"] % 4 or r["bytes"] == 0:
            ok = False
    a_serial = max(r["total"] for r in armA) / min(r["total"] for r in armA)
    b_parallel = max(r["total"] for r in armB) / min(r["total"] for r in armB)
    print(f"serialisation ratio A (slower/faster finish) = {a_serial:.2f}; B = {b_parallel:.2f}")
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)
