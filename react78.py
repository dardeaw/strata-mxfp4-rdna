#!/usr/bin/env python3
"""Reconstructed 8-task ReAct/tool-call suite (strata-0210 prod engine).
Pattern follows concbench/tool_ladder.py: the model must call the right tool
with valid args (temp=0, deterministic). No real tool execution.
Covers: weather, email, search (original 3) + calc, translate, read_file,
run_command, summarize (reconstructed to 8).
"""
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next-tsaipifong-mxfp4-mtp"

TOOLS = [
 {"type": "function", "function": {"name": "get_weather",
   "description": "Current weather for a city.",
   "parameters": {"type": "object", "properties": {
     "city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}},
     "required": ["city", "unit"]}}},
 {"type": "function", "function": {"name": "send_email",
   "description": "Send an email to a recipient.",
   "parameters": {"type": "object", "properties": {
     "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
     "required": ["to", "subject", "body"]}}},
 {"type": "function", "function": {"name": "search_docs",
   "description": "Search the internal documentation.",
   "parameters": {"type": "object", "properties": {
     "query": {"type": "string"}, "limit": {"type": "integer"}},
     "required": ["query", "limit"]}}},
 {"type": "function", "function": {"name": "calculate",
   "description": "Evaluate an arithmetic expression.",
   "parameters": {"type": "object", "properties": {
     "expression": {"type": "string"}}, "required": ["expression"]}}},
 {"type": "function", "function": {"name": "translate_text",
   "description": "Translate text to a target language.",
   "parameters": {"type": "object", "properties": {
     "text": {"type": "string"}, "target_lang": {"type": "string"}},
     "required": ["text", "target_lang"]}}},
 {"type": "function", "function": {"name": "read_file",
   "description": "Read a file from the repository.",
   "parameters": {"type": "object", "properties": {
     "path": {"type": "string"}}, "required": ["path"]}}},
 {"type": "function", "function": {"name": "run_command",
   "description": "Run a shell command and return output.",
   "parameters": {"type": "object", "properties": {
     "command": {"type": "string"}}, "required": ["command"]}}},
 {"type": "function", "function": {"name": "summarize",
   "description": "Summarize a document in N bullet points.",
   "parameters": {"type": "object", "properties": {
     "topic": {"type": "string"}, "points": {"type": "integer"}},
     "required": ["topic", "points"]}}},
]

TASKS = [
 ("weather", "What is the weather in Taipei right now? Use Celsius.", "get_weather"),
 ("email", "Email ops@example.com about the disk being full on db-7. Keep it short.", "send_email"),
 ("search", "Search our docs for the KV cache eviction policy. Return five results.", "search_docs"),
 ("calc", "Calculate 17 multiplied by 24, then add 100. Show the expression you evaluate.", "calculate"),
 ("translate", "Translate 'Good morning' to French.", "translate_text"),
 ("read", "Read the file README.md from the repository root.", "read_file"),
 ("shell", "Check disk usage with the df command.", "run_command"),
 ("summary", "Summarize the quarterly report in 3 bullet points.", "summarize"),
]

def call(task, prompt):
    body = json.dumps({"model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS, "tool_choice": "auto",
        "temperature": 0.0, "max_tokens": 256, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
        "Authorization": "Bearer strata67"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    dt = time.perf_counter() - t0
    tc = (((payload.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
    u = payload.get("usage", {})
    ct = u.get("completion_tokens", 0)
    return tc, dt, (ct / dt if dt > 0 and ct else 0.0)

def check(task, args):
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except Exception:
        return False, "args not JSON"
    if not isinstance(parsed, dict):
        return False, "args not object"
    low = {str(k).lower(): v for k, v in parsed.items()}
    get = lambda k, d="": low.get(k, d)
    if task == "weather":
        ok = str(get("city", "")).lower() == "taipei" and str(get("unit", "")).upper() in ("C", "CELSIUS")
        return ok, "city/unit"
    if task == "email":
        ok = "ops@example.com" in str(get("to", "")).lower() and "disk" in str(get("subject", "")).lower() and bool(str(get("body", "")).strip())
        return ok, "to/subject/body"
    if task == "search":
        try: lim = int(get("limit", 0) or 0)
        except (TypeError, ValueError): return False, "limit?"
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
        try: pts = int(get("points", 0) or 0)
        except (TypeError, ValueError): return False, "points?"
        return pts == 3, "points"
    return False, "unknown task"

def main():
    npass = 0
    dtpss = []
    for name, prompt, want in TASKS:
        try:
            tc, dt, dtps = call(name, prompt)
            if not tc:
                print("%-8s NO-CALL %.1fs" % (name, dt), flush=True)
                continue
            fn = tc[0].get("function", {})
            ok, why = check(name, fn.get("arguments", ""))
            fnok = fn.get("name", "") == want
            good = ok and fnok
            npass += 1 if good else 0
            dtpss.append(dtps)
            print("%-8s %s tool=%s want=%s (%s) %.1fs %.1f tok/s" % (
                name, "PASS" if good else "FAIL", fn.get("name", "?"), want, why, dt, dtps), flush=True)
        except Exception as e:
            print("%-8s ERROR %r" % (name, e), flush=True)
    import statistics
    med = statistics.median(dtpss) if dtpss else 0
    print("=== result: %d/%d pass, median decode %.1f tok/s ===" % (npass, len(TASKS), med))

if __name__ == "__main__":
    main()
