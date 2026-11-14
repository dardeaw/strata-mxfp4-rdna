#!/usr/bin/env python3
"""8-Task ReAct Tool-Calling Agent Benchmark Suite for Strata.

Twin pillar alongside tools_ladder.py: evaluates an OpenAI-compatible endpoint
across 8 diverse functional agent tool tasks under deterministic temperature=0 decoding.
No dummy tool execution needed; tests tool selection, parameter constraint adherence,
and decoding throughput.

Tasks:
  1. weather:   get_weather(city, unit)
  2. email:     send_email(to, subject, body)
  3. search:    search_docs(query, limit)
  4. calc:      calculate(expression)
  5. translate: translate_text(text, target_lang)
  6. read:      read_file(path)
  7. shell:     run_command(command)
  8. summary:   summarize(topic, points)

Usage:
    python tools/tools_react.py [--base-url http://127.0.0.1:8081/v1] [--model <id>] [--key KEY]
    python tools/tools_react.py http://127.0.0.1:8081/v1/chat/completions <model>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["C", "F"]},
                },
                "required": ["city", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the internal documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "limit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text to a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_lang": {"type": "string"},
                },
                "required": ["text", "target_lang"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize a document in N bullet points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "points": {"type": "integer"},
                },
                "required": ["topic", "points"],
            },
        },
    },
]

TASKS = [
    ("weather", "What is the weather in Taipei right now? Use Celsius.", "get_weather"),
    ("email", "Email ops@example.com about the disk being full on db-7. Keep it short.", "send_email"),
    ("search", "Search our docs for the KV cache eviction policy. Return five results.", "search_docs"),
    ("calc", "Calculate 17 multiplied by 24, then add 100. Show the expression you evaluate.", "calculate"),
    ("translate", "Translate 'Good morning' to French.", "translate_text"),
    ("read", "Read the file README.md from the repository root.", "read_file"),
    ("shell", "Check disk usage with the df command.", "run_command"),
    ("summary", "Summarize the quarterly report in 3 bullet points using the summarize tool.", "summarize"),
]


def post(url: str, key: str, body: dict, timeout: int = 600) -> tuple[dict, float]:
    url_str = url.rstrip("/")
    if not url_str.endswith("/chat/completions"):
        url_str += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url_str,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    dt = time.perf_counter() - t0
    return payload, dt


def call_task(url: str, key: str, model: str, prompt: str, timeout: int = 600) -> tuple[list[dict], float, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }
    payload, dt = post(url, key, body, timeout=timeout)
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    tc = msg.get("tool_calls") or []
    usage = payload.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    tok_per_sec = ct / dt if dt > 0 and ct else 0.0
    return tc, dt, tok_per_sec


def check_args(task: str, args: str | dict) -> tuple[bool, str]:
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except Exception:
        return False, "args not valid JSON"
    if not isinstance(parsed, dict):
        return False, "args not object"

    low = {str(k).lower(): v for k, v in parsed.items()}
    get = lambda k, d="": low.get(k, d)

    if task == "weather":
        ok = str(get("city", "")).lower() == "taipei" and str(get("unit", "")).upper() in ("C", "CELSIUS")
        return ok, "city/unit"
    if task == "email":
        ok = (
            "ops@example.com" in str(get("to", "")).lower()
            and "disk" in str(get("subject", "")).lower()
            and bool(str(get("body", "")).strip())
        )
        return ok, "to/subject/body"
    if task == "search":
        try:
            lim = int(get("limit", 0) or 0)
        except (TypeError, ValueError):
            return False, "limit invalid"
        ok = "eviction" in str(get("query", "")).lower() and lim == 5
        return ok, "query/limit"
    if task == "calc":
        return bool(str(get("expression", "")).strip()), "expression"
    if task == "translate":
        ok = bool(str(get("text", "")).strip()) and "french" in str(get("target_lang", "")).lower()
        return ok, "text/lang"
    if task == "read":
        ok = "readme" in str(get("path", "")).lower()
        return ok, "path"
    if task == "shell":
        ok = "df" in str(get("command", "")).lower()
        return ok, "command"
    if task == "summary":
        try:
            pts = int(get("points", 0) or 0)
        except (TypeError, ValueError):
            return False, "points invalid"
        return pts == 3, "points"

    return False, "unknown task"


def main() -> None:
    # Support both argparse flags and backward-compatible positional args: [url] [model]
    default_url = "http://127.0.0.1:8081/v1"
    default_model = "qwen3.8-flash-next-tsaipifong-mxfp4-mtp"

    pos_url = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    pos_model = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None

    ap = argparse.ArgumentParser(
        description="ReAct 8-task tool-calling agent benchmark suite for Strata."
    )
    ap.add_argument("--base-url", default=pos_url or default_url, help="OpenAI-compatible endpoint base URL")
    ap.add_argument("--model", default=pos_model or default_model, help="Target model identifier")
    ap.add_argument("--key", default="strata67", help="Authorization bearer token")
    ap.add_argument("--timeout", type=int, default=600, help="Per-request HTTP timeout in seconds")
    ap.add_argument("--prompt-summary-ambiguous", action="store_true", help="Use original unguided summary prompt")
    args, unknown = ap.parse_known_args()

    tasks = list(TASKS)
    if args.prompt_summary_ambiguous:
        tasks = [
            (t, "Summarize the quarterly report in 3 bullet points." if t == "summary" else p, w)
            for t, p, w in tasks
        ]

    npass = 0
    dtpss: list[float] = []
    print(f"=== ReAct 8-Task Benchmark: url={args.base_url} model={args.model} ===", flush=True)

    for name, prompt, want in tasks:
        try:
            tc, dt, dtps = call_task(args.base_url, args.key, args.model, prompt, timeout=args.timeout)
            if not tc:
                print(f"{name:<10} NO-CALL {dt:.1f}s", flush=True)
                continue
            fn = tc[0].get("function", {})
            fn_name = fn.get("name", "")
            ok, why = check_args(name, fn.get("arguments", ""))
            fnok = fn_name == want
            good = ok and fnok
            npass += 1 if good else 0
            if dtps > 0:
                dtpss.append(dtps)
            print(
                f"{name:<10} {'PASS' if good else 'FAIL'} tool={fn_name} want={want} ({why}) {dt:.1f}s {dtps:.1f} tok/s",
                flush=True,
            )
        except Exception as e:
            print(f"{name:<10} ERROR {e!r}", flush=True)

    med = statistics.median(dtpss) if dtpss else 0.0
    print(f"=== result: {npass}/{len(tasks)} pass, median decode {med:.1f} tok/s ===", flush=True)
    print("REACT-DONE", flush=True)
    sys.exit(0 if npass == len(tasks) else 1)


if __name__ == "__main__":
    main()
