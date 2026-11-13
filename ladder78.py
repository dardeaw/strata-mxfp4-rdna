#!/usr/bin/env python3
"""7800XT ladder (published-spec style) + 5-scenario decode."""
import json, sys, time, urllib.request, uuid, argparse
TIMEOUT = 1500
CTX = 262144
SIZES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 250000]
BASE = ("In distributed systems, consistency models trade latency for guarantees. "
        "Linearizability offers the strongest guarantee, while eventual consistency "
        "relaxes it for availability. Benchmarks must isolate prefill from decode "
        "to attribute cost correctly, because prompt processing is compute bound "
        "and token generation is memory bandwidth bound. ")
SCEN = [
 ("prose", 0.4, "Write a short story about a robot learning to paint.", 256),
 ("fib", 0.4, "Write a Python function to compute the nth Fibonacci number.", 256),
 ("json", 0.0, "Output ONLY a JSON object with keys name, age, city, hobbies describing a person.", 128),
 ("code", 0.0, "Write a Python function to merge two sorted lists into one sorted list.", 128),
 ("repeat", 0.0, "Repeat the following sentence verbatim over and over: the quick brown fox jumps over the lazy dog.", 256),
]
def post(url, key, model, prompt, mx, temp):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "temperature": temp}).encode()
    req = urllib.request.Request(url, data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        obj = json.loads(r.read().decode("utf-8", "ignore"))
    return obj, time.perf_counter() - t0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=["ladder", "decode", "all"])
    a = ap.parse_args()
    ratio = None
    if a.phase in ("ladder", "all"):
        f = open(a.out + "-ladder.jsonl", "w")
        for size in SIZES:
            if size >= CTX - 512:
                print("SKIP %d" % size, flush=True); continue
            nonce = uuid.uuid4().hex[:12]
            est = int(size * 4.2) if ratio is None else int(size / ratio)
            parts = ["Ladder %s target %d. " % (nonce, size)]; k = 0; n = len(parts[0])
            while n < est:
                q = "[%s-%05d] %s" % (nonce, k, BASE); parts.append(q); n += len(q); k += 1
            try:
                obj, wall = post(a.url, a.key, a.model, "".join(parts), 1, 0.0)
                u = obj.get("usage", {}); pt = u.get("prompt_tokens", -1)
                if pt and pt > 0: ratio = pt / sum(map(len, parts))
                rec = {"target": size, "actual": pt, "wall_s": round(wall, 1),
                       "wall_tps": round(pt / wall, 1) if pt > 0 else None}
                print("LADDER target=%d actual=%s wall=%.1fs -> %s tok/s" % (
                    size, pt, wall, rec["wall_tps"]), flush=True)
            except Exception as e:
                rec = {"target": size, "error": repr(e)[:200]}
                print("LADDER target=%d ERROR %r" % (size, e), flush=True)
            f.write(json.dumps(rec) + "\n"); f.flush()
            time.sleep(2)
        f.close()
    if a.phase in ("decode", "all"):
        f = open(a.out + "-decode.jsonl", "w")
        for name, temp, prompt_base, mx in SCEN:
            nonce = uuid.uuid4().hex[:12]
            try:
                obj, wall = post(a.url, a.key, a.model, prompt_base + "\n[session %s]" % nonce, mx, temp)
                u = obj.get("usage", {}); ct = u.get("completion_tokens", -1)
                ch = (((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                rec = {"scen": name, "temp": temp, "gen": ct, "wall_s": round(wall, 1),
                       "wall_tps": round(ct / wall, 2) if ct > 0 else None,
                       "finish": ((obj.get("choices") or [{}])[0].get("finish_reason")),
                       "preview": ch[:60]}
                print("DECODE %s gen=%s wall=%.1fs -> %s tok/s finish=%s" % (
                    name, ct, wall, rec["wall_tps"], rec["finish"]), flush=True)
            except Exception as e:
                rec = {"scen": name, "error": repr(e)[:200]}
                print("DECODE %s ERROR %r" % (name, e), flush=True)
            f.write(json.dumps(rec) + "\n"); f.flush()
            time.sleep(2)
        f.close()
    print("LADDER78_DONE", flush=True)
if __name__ == "__main__":
    main()