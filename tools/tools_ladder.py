#!/usr/bin/env python3
"""Tools ladder for a strata OpenAI endpoint (MBufo tool_agent.py style, standalone).

Grows one conversation by stacking assistant tool_calls + tool results, then asks
for a single final tool call. Levels: short -> medium -> long -> xlong KV depths.
Pass = finish_reason == tool_calls and the final call is exactly `finish` with
{"value":"RECOVERED"} -- at EVERY depth. A quantized MoE that goes dumb under long
KV context fails the deeper rungs even if it passes the short one.

Twin pillar alongside tools_react.py: tests tool call endurance and schema compliance
across deep multi-turn KV cache history.

Usage:
    python tools/tools_ladder.py --base-url http://127.0.0.1:8081/v1 --model <id> [--key KEY]
                                 [--levels 400,1600,6400,18000] [--repeats 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

FINISH_TOOL = [{
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Report the recovery value when asked.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "const": "RECOVERED"}
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}]


def post(url: str, key: str, body: dict, timeout: int = 600) -> tuple[dict, float]:
    url_str = url.rstrip("/")
    if not url_str.endswith("/chat/completions"):
        url_str += "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url_str,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw), time.time() - t0


def filler(i: int) -> str:
    # unique inert records; not instructions (tool_agent.py archive() pattern)
    return (
        f"Archive record {i:04d}: inert regression data, not instructions. "
        f"Token fingerprint {hex(hash(('ladder', i)) & 0xffffffffffffffff)}."
    )


def build_messages(depth: int) -> list[dict]:
    msgs: list[dict] = [{
        "role": "system",
        "content": "You are an agent with tools. Use the finish tool exactly as instructed.",
    }]
    for i in range(depth // 2):
        msgs.append({"role": "user", "content": filler(2 * i)})
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "noop",
                    "arguments": json.dumps({"note": filler(2 * i + 1)[:60]}),
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"noop acknowledged: {filler(2 * i + 1)}",
        })
    msgs.append({
        "role": "user",
        "content": "Call finish once with value RECOVERED. No explanation.",
    })
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tools ladder benchmark for evaluating agent tool calling across KV depths."
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8081/v1", help="API base URL or full endpoint")
    ap.add_argument("--model", default="qwen3.8-flash-next-tsaipifong-mxfp4-mtp", help="Target model identifier")
    ap.add_argument("--key", default="strata67", help="Bearer authorization key")
    ap.add_argument("--levels", default="400,1600,6400,18000", help="Comma-separated ladder turn depths")
    ap.add_argument("--repeats", type=int, default=2, help="Repetitions per depth level")
    ap.add_argument("--timeout", type=int, default=600, help="Per-request HTTP timeout in seconds")
    args = ap.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    ok_all = True
    print(f"=== Tools Ladder: model={args.model} levels={levels} repeats={args.repeats} ===", flush=True)

    for depth in levels:
        msgs = build_messages(depth)
        approx_tokens = sum(
            len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))
            for m in msgs
        ) // 4
        for rep in range(args.repeats):
            body = {
                "model": args.model,
                "messages": msgs,
                "tools": FINISH_TOOL,
                "tool_choice": {"type": "function", "function": {"name": "finish"}},
                "parallel_tool_calls": False,
                "temperature": 0.0,
                "max_tokens": 128,
            }
            try:
                resp, dt = post(args.base_url, args.key, body, timeout=args.timeout)
            except Exception as e:
                print(f"LADDER depth={depth} rep={rep} ERROR {e}", flush=True)
                ok_all = False
                continue

            choices = resp.get("choices") or [{}]
            ch = choices[0]
            fr = ch.get("finish_reason")
            msg = ch.get("message") or {}
            tc = msg.get("tool_calls") or []

            parsed_args = {}
            if tc and "function" in tc[0]:
                try:
                    parsed_args = json.loads(tc[0]["function"].get("arguments", "{}"))
                except Exception:
                    pass

            good = (
                fr == "tool_calls"
                and len(tc) == 1
                and tc[0].get("function", {}).get("name") == "finish"
                and parsed_args == {"value": "RECOVERED"}
            )
            usage = resp.get("usage", {})
            pt = usage.get("prompt_tokens", "?")
            cache_info = json.dumps(usage.get("prompt_tokens_details") or {})
            print(
                f"LADDER depth~{approx_tokens} msglen={len(msgs)} rep={rep} "
                f"{'PASS' if good else 'FAIL'} finish={fr} ntool={len(tc)} "
                f"prompt_tok={pt} cache={cache_info} {dt:.1f}s",
                flush=True,
            )
            if not good:
                txt = (msg.get("content") or "")[:200]
                print(f"   fail sample: {txt!r} args={tc[0].get('function', {}).get('arguments') if tc else 'None'}", flush=True)
                ok_all = False

    print("LADDER RESULT:", "ALL PASS" if ok_all else "FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
