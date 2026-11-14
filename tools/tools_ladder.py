#!/usr/bin/env python3
"""Tools Ladder: tool-call accuracy & degradation vs context length on Strata / MBufo endpoints.

Directly modeled on MBufo / concbench tool_ladder.py: evaluates whether an LLM
maintains tool-calling competence as the context length increases along a geometric ladder
(e.g. 0 -> 500 -> 2,000 -> 8,000 -> 16,000 -> 32,000 -> 64,000 -> 128,000 tokens).

Tests:
  1. Tool dispatch accuracy (correct tool invoked vs no-call / text-json hallucination)
  2. Schema argument parsing (proper JSON, matching expected keys and constraints)
  3. Prefill & decode performance across context depth
  4. Context-degradation breakdown by length rung and by task domain

Twin pillar alongside tools_react.py:
  - tools_ladder.py: Multi-rung context scaling endurance (MBufo style)
  - tools_react.py:  8-domain functional ReAct agent task suite

Usage:
    python tools/tools_ladder.py [--url http://127.0.0.1:8081/v1] [--model <id>] [--key KEY]
                                 [--lengths 0,500,2000,8000,16000,32000,64000,128000]
                                 [--tasks weather,email,search] [--reps 1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

BASE_FILLER = (
    "The quick brown fox jumps over the lazy dog. In distributed "
    "systems, consistency models trade latency for guarantees; "
    "linearizability offers the strongest guarantee, while eventual "
    "consistency relaxes it. Benchmarks must isolate prefill from decode "
    "to attribute cost correctly, because prompt processing is compute "
    "bound and token generation is memory bandwidth bound. Careful "
    "experiment design repeats each measurement three times and reports "
    "the median. "
)

SYSTEM_PROMPT = (
    "You are an operations assistant with tools. When a tool can answer "
    "the request, call it instead of writing the arguments as text."
)

ALL_TOOLS = [
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

TASK_REGISTRY = {
    "weather": (
        "weather",
        "What is the weather in Taipei right now? Use Celsius.",
        "get_weather",
    ),
    "email": (
        "email",
        "Email ops@example.com about the disk being full on db-7. Keep it short.",
        "send_email",
    ),
    "search": (
        "search",
        "Search our docs for the KV cache eviction policy. Return five results.",
        "search_docs",
    ),
    "calc": (
        "calc",
        "Calculate 17 multiplied by 24, then add 100. Show the expression you evaluate.",
        "calculate",
    ),
    "translate": (
        "translate",
        "Translate 'Good morning' to French.",
        "translate_text",
    ),
    "read": (
        "read",
        "Read the file README.md from the repository root.",
        "read_file",
    ),
    "shell": (
        "shell",
        "Check disk usage with the df command.",
        "run_command",
    ),
    "summary": (
        "summary",
        "Summarize the quarterly report in 3 bullet points using the summarize tool.",
        "summarize",
    ),
}

DEFAULT_TASKS = ["weather", "email", "search"]


def normalize_endpoint(url: str) -> str:
    cleaned = url.rstrip("/")
    if not cleaned.endswith("/chat/completions"):
        if cleaned.endswith("/v1"):
            cleaned += "/chat/completions"
        else:
            cleaned += "/v1/chat/completions"
    return cleaned


def post_request(
    url: str,
    key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
    max_tokens: int = 256,
    timeout: int = 1800,
) -> tuple[float, dict]:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", "ignore"))
    wall = time.perf_counter() - t0
    return wall, payload


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
        if str(get("city", "")).lower() != "taipei":
            return False, f"city={get('city')!r}"
        if str(get("unit", "")).upper() not in ("C", "CELSIUS"):
            return False, f"unit={get('unit')!r}"
        return True, "ok"

    if task == "email":
        if "ops@example.com" not in str(get("to", "")).lower():
            return False, f"to={get('to')!r}"
        if "disk" not in str(get("subject", "")).lower():
            return False, f"subject={get('subject')!r}"
        if not str(get("body", "")).strip():
            return False, "empty body"
        return True, "ok"

    if task == "search":
        try:
            limit = int(get("limit", 0) or 0)
        except (TypeError, ValueError):
            return False, f"limit={get('limit')!r}"
        if "eviction" not in str(get("query", "")).lower():
            return False, f"query={get('query')!r}"
        if limit != 5:
            return False, f"limit={limit}"
        return True, "ok"

    if task == "calc":
        if not str(get("expression", "")).strip():
            return False, "empty expression"
        return True, "ok"

    if task == "translate":
        if not str(get("text", "")).strip():
            return False, "empty text"
        if "french" not in str(get("target_lang", "")).lower():
            return False, f"target_lang={get('target_lang')!r}"
        return True, "ok"

    if task == "read":
        if "readme" not in str(get("path", "")).lower():
            return False, f"path={get('path')!r}"
        return True, "ok"

    if task == "shell":
        if "df" not in str(get("command", "")).lower():
            return False, f"command={get('command')!r}"
        return True, "ok"

    if task == "summary":
        try:
            points = int(get("points", 0) or 0)
        except (TypeError, ValueError):
            return False, f"points={get('points')!r}"
        if points != 3:
            return False, f"points={points}"
        return True, "ok"

    return False, "unknown task"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tools Ladder: Tool-call degradation vs context length on Strata/MBufo endpoints."
    )
    ap.add_argument(
        "--url",
        "--base-url",
        dest="url",
        default="http://127.0.0.1:8081/v1",
        help="API base URL or full /chat/completions endpoint (default: http://127.0.0.1:8081/v1)",
    )
    ap.add_argument(
        "--model",
        default="qwen3.8-flash-next-tsaipifong-mxfp4-mtp",
        help="Target model identifier (default: qwen3.8-flash-next-tsaipifong-mxfp4-mtp)",
    )
    ap.add_argument(
        "--key",
        default="strata67",
        help="Authorization bearer key (default: strata67)",
    )
    ap.add_argument(
        "--lengths",
        default="0,500,2000,8000,16000,32000,64000,128000",
        help="Comma-separated context length rungs (default: 0,500,2000,8000,16000,32000,64000,128000)",
    )
    ap.add_argument(
        "--tasks",
        default="weather,email,search",
        help="Comma-separated task names or 'all' (options: weather, email, search, calc, translate, read, shell, summary)",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Repetitions per length rung (default: 1)",
    )
    ap.add_argument(
        "--tool-choice",
        default="auto",
        help="tool_choice parameter value (default: auto)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="HTTP request timeout in seconds (default: 1800)",
    )
    args = ap.parse_args()

    endpoint = normalize_endpoint(args.url)
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    if args.tasks.strip().lower() == "all":
        selected_task_keys = list(TASK_REGISTRY.keys())
    else:
        selected_task_keys = [k.strip() for k in args.tasks.split(",") if k.strip()]

    tasks = [TASK_REGISTRY[k] for k in selected_task_keys if k in TASK_REGISTRY]
    if not tasks:
        print(f"Error: no valid tasks selected from {args.tasks!r}", file=sys.stderr)
        sys.exit(1)

    # Filter declared tools to those required for selected tasks
    wanted_fn_names = {expected for _, _, expected in tasks}
    tools = [t for t in ALL_TOOLS if t["function"]["name"] in wanted_fn_names]

    print(
        f"=== Tools Ladder Benchmark (MBufo style) ===\n"
        f"Endpoint: {endpoint}\n"
        f"Model:    {args.model}\n"
        f"Lengths:  {lengths}\n"
        f"Tasks:    {[t[0] for t in tasks]}\n"
        f"Reps:     {args.reps}\n",
        flush=True,
    )

    total = 0
    failures: list[tuple] = []
    by_length: dict[int, list[int]] = {}
    by_task: dict[str, list[int]] = {}

    for length in lengths:
        by_length.setdefault(length, [0, 0])
        for rep in range(args.reps):
            filler = (
                ""
                if length == 0
                else f"tooltest {length} r{rep} "
                + BASE_FILLER * max(1, round(length / 77))
            )
            for task, question, expected in tasks:
                total += 1
                by_task.setdefault(task, [0, 0])

                user_content = (filler + "\n\n" + question).strip()
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                try:
                    wall, payload = post_request(
                        endpoint,
                        args.key,
                        args.model,
                        messages,
                        tools,
                        tool_choice=args.tool_choice,
                        timeout=args.timeout,
                    )
                except Exception as error:  # noqa: BLE001
                    failures.append((length, rep, task, "ERROR", str(error)))
                    by_task[task][1] += 1
                    by_length[length][1] += 1
                    print(
                        f"{total:3d} len={length:<6d} rep={rep:02d} {task:<9s} FAIL error={error}",
                        flush=True,
                    )
                    continue

                choices = payload.get("choices") or [{}]
                choice = choices[0]
                message = choice.get("message") or {}
                calls = message.get("tool_calls") or []
                usage = payload.get("usage") or {}
                actual_prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                finish = choice.get("finish_reason")
                content = str(message.get("content") or "")
                tok_s = (completion_tokens / wall) if wall > 0 and completion_tokens else 0.0

                ok = False
                detail = ""
                if len(calls) == 1:
                    fn = calls[0].get("function") or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "")
                    try:
                        json.loads(raw_args)
                        json_ok = True
                    except Exception:
                        json_ok = False
                    args_ok, note = check_args(task, raw_args)
                    ok = (name == expected) and args_ok and json_ok
                    detail = "ok" if ok else f"{name} {note} {str(raw_args)[:70]}"
                else:
                    kind = (
                        "text-json"
                        if re.search(r"get_weather|send_email|search_docs|calculate|translate_text|read_file|run_command|summarize", content)
                        else "no-call"
                    )
                    detail = f"{kind} {content[:70]!r}"

                if ok:
                    by_task[task][0] += 1
                    by_length[length][0] += 1
                else:
                    by_task[task][1] += 1
                    by_length[length][1] += 1
                    failures.append((length, rep, task, finish, detail))

                print(
                    f"{total:3d} len={length:<6d} rep={rep:02d} {task:<9s} "
                    f"{'PASS' if ok else 'FAIL'} tok={actual_prompt_tokens:<7d} "
                    f"finish={finish:<10s} {wall:.1f}s {tok_s:4.1f} tok/s "
                    f"{'' if ok else detail}",
                    flush=True,
                )

    failed = len(failures)
    passed = total - failed
    pct = (failed / total * 100) if total > 0 else 0.0
    print(
        f"\n=== result: {passed}/{total} pass, {failed}/{total} fail ({pct:.1f}% fail) ===",
        flush=True,
    )

    print("\nBy Task:")
    for task_name, (good, bad) in sorted(by_task.items()):
        total_task = good + bad
        t_pct = (good / total_task * 100) if total_task > 0 else 0.0
        print(f"  {task_name:<10s} {good:3d}/{total_task:3d} pass ({t_pct:.0f}%)")

    print("\nBy Context Length:")
    for len_val, (good, bad) in sorted(by_length.items()):
        total_len = good + bad
        l_pct = (good / total_len * 100) if total_len > 0 else 0.0
        print(f"  len={len_val:<7d} {good:3d}/{total_len:3d} pass ({l_pct:.0f}%)")

    if failures:
        print("\nFailures:")
        for row in failures:
            print(" ", row)

    print("\nLADDER-DONE", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
