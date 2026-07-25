"""pass@1 code-correctness eval against a running llama-server, using a
contamination-reduced sample from LiveCodeBench (AtCoder, stdin/stdout
problems only) instead of HumanEval.

See README "Benchmark methodology" for why this reduces but does not
eliminate contamination risk: the newest available LiveCodeBench problems
(Jan-Apr 2025) still predate all three models' likely training cutoffs.

Usage:
    python livecodebench_bench.py <label>
"""

import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
HERE = Path(__file__).parent
DATA_PATH = HERE / "LiveCodeBench-AtCoder.jsonl.gz"
RESULTS_DIR = HERE / "results"

MAX_TOKENS = 4096  # same reasoning-model lesson as humaneval_bench.py
EXEC_TIMEOUT_S = 10

SYSTEM = (
    "Solve the given competitive programming problem. Write a complete "
    "Python program that reads input from stdin and writes the answer to "
    "stdout, exactly matching the expected output format. Respond with "
    "ONLY a single Python code block. No explanation."
)


def load_problems():
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def query(prompt):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    choice = body["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason", "")


def extract_code(response):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return blocks[0] if blocks else (response if "def " in response or "input(" in response or "sys.stdin" in response else None)


def run_program(code, stdin_text):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            input=stdin_text, capture_output=True, text=True, timeout=EXEC_TIMEOUT_S,
            encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout, (result.stderr or "").strip()[-300:]
    except subprocess.TimeoutExpired:
        return None, "", "timeout"
    finally:
        os.unlink(path)


def check_output(actual, expected):
    # Whitespace-insensitive comparison, matching typical judge behavior.
    return actual.strip().split() == expected.strip().split()


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python livecodebench_bench.py <label>")
    label = sys.argv[1]

    problems = load_problems()
    results = []
    passed = 0
    wall_start = time.time()

    for i, p in enumerate(problems, 1):
        qid, title = p["question_id"], p["question_title"]
        prompt = p["question_content"]
        tests = p["public_test_cases"]

        t0 = time.time()
        try:
            response, finish_reason = query(prompt)
        except Exception as exc:
            results.append({"question_id": qid, "title": title, "pass": False,
                            "error": f"request failed: {exc}", "seconds": time.time() - t0})
            print(f"[{i}/{len(problems)}] {qid} ({title}): FAIL (request error)")
            continue
        elapsed = time.time() - t0

        code = extract_code(response)
        if code is None:
            note = "empty response (truncated mid-reasoning?)" if finish_reason == "length" \
                else "no code extracted"
            results.append({"question_id": qid, "title": title, "pass": False,
                            "error": note, "seconds": elapsed})
            print(f"[{i}/{len(problems)}] {qid} ({title}): FAIL ({note}) [{elapsed:.1f}s]")
            continue

        all_pass, fail_detail = True, ""
        for t in tests:
            rc, stdout, stderr = run_program(code, t["input"])
            if rc != 0 or not check_output(stdout, t["output"]):
                all_pass = False
                fail_detail = stderr if rc != 0 else f"wrong output (got {stdout.strip()[:80]!r})"
                break

        results.append({"question_id": qid, "title": title, "pass": all_pass,
                        "error": fail_detail, "seconds": elapsed, "difficulty": p["difficulty"]})
        passed += all_pass
        print(f"[{i}/{len(problems)}] {qid} ({title}, {p['difficulty']}): "
              f"{'PASS' if all_pass else 'FAIL'} [{elapsed:.1f}s]")

    total = len(problems)
    wall = time.time() - wall_start
    summary = {
        "label": label,
        "source": "LiveCodeBench AtCoder subset (Jan-Apr 2025, stratified 10 easy/10 medium/10 hard)",
        "passed": passed,
        "total": total,
        "pass_at_1": round(passed / total, 4) if total else 0.0,
        "wall_clock_seconds": round(wall, 1),
        "generation_seconds": round(sum(r["seconds"] for r in results), 1),
        "failures": [r["question_id"] for r in results if not r["pass"]],
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"livecodebench_{label}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\npass@1: {passed}/{total} = {passed / total * 100:.1f}%")
    print(f"generation time: {summary['generation_seconds']}s "
          f"({summary['generation_seconds'] / total:.2f}s/problem)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
