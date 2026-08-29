#!/usr/bin/env python3
"""bench_robust.py — statistically robust sweep for GLM-5.3-Flash on 2x DGX Spark.

Upgrades over bench_c1c6.py:
  - N waves per level (default 4) + full per-request capture
  - Per-level mean +/- stdev and 95% CI (t-distribution via sample stats)
  - TTFT measured per request via streaming (interchunk SSE) + TPOT + decode tok/s
  - P50/P90/max wall per level
  - Mixed prompt set: code, reasoning, prose (acceptance varies by type)
  - Interleaved level order to decorrelate thermal/scheduler drift
  - Spec-decode acceptance delta per level from /metrics (as before)

Usage: python3 bench_robust.py [--url http://localhost:8000] [--waves 4]
       [--max-tokens 400] [--levels 1,2,3,4,5,6] [--tag NAME]
"""
import argparse, json, math, random, statistics, threading, time, urllib.request

URL = "http://localhost:8000"

PROMPTS = [
    # code (high acceptance expected)
    ("code", "Write a Python function that parses an nginx access log line into a dict, with a regex, and explain each group."),
    ("code", "Implement a rate limiter class in Python using the token bucket algorithm, then show example usage."),
    ("code", "Implement binary search in Python, then walk through the trace on [2,5,8,12,17,23] searching for 17."),
    ("code", "Write a SQL query for the top 5 customers by 90-day revenue, then rewrite it as a window function version."),
    # reasoning
    ("reason", "A warehouse ships 340 orders/day growing 6% weekly. Model 8 weeks of volume in a Python list comprehension and explain."),
    ("reason", "Explain the difference between TCP slow start and congestion avoidance, then pseudocode both."),
    ("reason", "Design a URL shortener: data model, hash strategy, collision handling, and read/write scaling. Be concrete."),
    # prose (low acceptance expected)
    ("prose", "Write a short essay on why coastal cities flood more often than inland ones, and what that means for insurance markets."),
    ("prose", "Draft a polite but firm email declining a vendor proposal while leaving the door open for next year."),
    ("prose", "Explain quantum entanglement to a curious 12-year-old using two everyday analogies."),
]

def http_post_stream(url, payload, timeout=900):
    """POST with stream=True; returns (ttft_s, total_s, completion_tokens, content)."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(url + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; buf = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            try:
                delta = ev["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                delta = ""
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                buf.append(delta)
    total = time.time() - t0
    ct = usage["completion_tokens"] if usage else max(1, round(sum(len(b) for b in buf)) // 4)
    return ttft, total, ct, "".join(buf)

def metrics_cum(url):
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
        drafted = accepted = None
        for line in txt.splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total{"):
                drafted = float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
                accepted = float(line.split()[-1])
        return drafted, accepted
    except Exception:
        return None, None

def ci95(vals):
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals) * 1.96 / math.sqrt(len(vals))

def run_wave(url, c, max_tokens):
    out = [None] * c
    def one(i):
        kind, text = PROMPTS[(i * 3 + c * 2 + random.randint(0, 9)) % len(PROMPTS)]
        salt = f"[{random.randint(1, 10**12)}] "
        payload = {"model": "glm-5.3-flash",
                   "messages": [{"role": "user", "content": salt + text}],
                   "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
        t0 = time.time()
        try:
            ttft, total, ct, content = http_post_stream(url, payload)
            out[i] = {"ok": True, "kind": kind, "wall": total - t0 + t0 - t0 + total - (total - t0) if False else total,
                      "ttft": ttft or (total), "ct": ct,
                      "tps": ct / max(total, 1e-9), "len_ok": ct >= max_tokens * 0.9}
        except Exception as e:
            out[i] = {"ok": False, "err": str(e)[:80], "wall": time.time() - t0}
    ts = [threading.Thread(target=one, args=(i,)) for i in range(c)]
    t0 = time.time()
    [t.start() for t in ts]; [t.join() for t in ts]
    return out, time.time() - t0

def summarize(level_rows):
    walls = [r["wall"] for r in level_rows]
    ttfts = [r["ttft"] for r in level_rows]
    tpss  = [r["tps"] for r in level_rows]
    def pct(v, p):
        s = sorted(v); k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
        return s[k]
    return {
        "n": len(walls),
        "wall_mean": round(statistics.mean(walls), 2),
        "wall_stdev": round(statistics.stdev(walls), 2) if len(walls) > 1 else 0.0,
        "wall_p50": round(pct(walls, 50), 2),
        "wall_p90": round(pct(walls, 90), 2),
        "ttft_mean": round(statistics.mean(ttfts), 2),
        "ttft_p90": round(pct(ttfts, 90), 2),
        "tps_mean": round(statistics.mean(tpss), 2),
        "tps_ci95": round(ci95(tpss), 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(",")]
    # interleave: pass1 runs levels ascending, pass2 descending, etc.
    order = []
    for w in range(a.waves):
        chunk = levels if w % 2 == 0 else list(reversed(levels))
        order.extend((w, c) for c in chunk)

    # warmup
    http_post_stream(a.url, {"model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "Explain hash maps in 200 words."}],
        "max_tokens": 300, "temperature": 1.0, "top_p": 0.95})

    per_level = {c: [] for c in levels}
    d_prev, a_prev = metrics_cum(a.url)
    agg_by_wave = {}

    for w, c in order:
        out, wall = run_wave(a.url, c, a.max_tokens)
        ok = [r for r in out if r["ok"]]
        d_now, a_now = metrics_cum(a.url)
        acc = (a_now - a_prev) / (d_now - d_prev) if (d_now and d_prev and a_now and a_prev and d_now > d_prev) else None
        d_prev, a_prev = d_now, a_now
        toks = sum(r["ct"] for r in ok)
        agg = toks / wall if wall else 0
        agg_by_wave.setdefault(c, []).append(agg)
        per_level[c].extend(ok)
        accs = f"{acc:.3f}" if acc else "n/a"
        print(f"wave{w} C{c}: {len(ok)}/{c} ok  agg={agg:6.1f} tok/s  wall={wall:6.1f}s  accept={accs}", flush=True)

    print("\n=== SUMMARY (per-request stats, all waves pooled) ===")
    print(f"{'C':>2} {'n':>3} {'wall_mean':>9} {'wall_sd':>8} {'wall_p50':>8} {'wall_p90':>8} "
          f"{'ttft_mean':>9} {'ttft_p90':>8} {'tps_mean':>8} {'tps_ci95':>8}")
    final = {}
    for c in levels:
        s = summarize(per_level[c])
        final[c] = s
        print(f"{c:>2} {s['n']:>3} {s['wall_mean']:>9.2f} {s['wall_stdev']:>8.2f} {s['wall_p50']:>8.2f} {s['wall_p90']:>8.2f} "
              f"{s['ttft_mean']:>9.2f} {s['ttft_p90']:>8.2f} {s['tps_mean']:>8.2f} {s['tps_ci95']:>8.2f}")

    print("\n=== per-prompt-type decode tok/s (mean) ===")
    for kind in ("code", "reason", "prose"):
        v = [r["tps"] for c in levels for r in per_level[c] if r["kind"] == kind]
        if v:
            print(f"{kind:>7}: {statistics.mean(v):6.2f} tok/s  (n={len(v)})")

    peak_c = max(final, key=lambda c: statistics.mean([x for x in agg_by_wave[c]]))
    print(f"\npeak aggregate by mean wave agg: C{peak_c} = "
          f"{statistics.mean(agg_by_wave[peak_c]):.1f} tok/s")

    with open(f"/tmp/bench_robust_{a.tag}.json", "w") as f:
        json.dump({"tag": a.tag, "waves": a.waves, "max_tokens": a.max_tokens,
                   "summary": final, "wave_agg": agg_by_wave}, f, indent=1)
    print(f"saved -> /tmp/bench_robust_{a.tag}.json")

if __name__ == "__main__":
    main()
