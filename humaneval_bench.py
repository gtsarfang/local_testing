"""pass@1 code-correctness eval against a running llama-server.

Usage:
    python humaneval_bench.py <label>            # standard 80-problem sample
    python humaneval_bench.py <label> 0 40       # custom slice offsets

Sends HumanEval problems to http://127.0.0.1:8080, extracts the generated
function, and executes it against HumanEval's real unit tests in a subprocess.
Writes results/humaneval_<label>.json.
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
DATA_PATH = HERE / "HumanEval.jsonl.gz"
RESULTS_DIR = HERE / "results"

# Four spread slices of 20 rather than one contiguous block, so the sample
# isn't biased toward the (generally easier) start of the dataset.
DEFAULT_OFFSETS = [0, 40, 100, 140]
SLICE_SIZE = 20

# Generous budget: reasoning models (e.g. gpt-oss) emit chain-of-thought into a
# separate `reasoning_content` field first, and a tight limit truncates them
# mid-thought, leaving `content` empty — which looks like a wrong answer but is
# actually a starved one. See README "Benchmark methodology".
MAX_TOKENS = 4096
EXEC_TIMEOUT_S = 10

SYSTEM = (
    "Complete the given Python function. Respond with ONLY a single Python "
    "code block containing the complete function (signature + body, "
    "including the docstring). No explanation."
)


def load_problems(offsets, n=SLICE_SIZE):
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        all_problems = [json.loads(line) for line in f]
    picked = []
    for off in offsets:
        picked.extend(all_problems[off:off + n])
    return picked


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


def extract_code(response, entry_point):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = blocks[0] if blocks else response
    return code if f"def {entry_point}" in code else None


def preamble_of(prompt, entry_point):
    """Imports/helpers that precede the target function in the original prompt.

    A model asked to complete just the function won't repeat `from typing import
    List`, so the extracted code won't run standalone without this.
    """
    lines = []
    for line in prompt.splitlines():
        if line.startswith(f"def {entry_point}"):
            break
        lines.append(line)
    return "\n".join(lines)


def run_test(preamble, candidate, test_code, entry_point):
    source = f"{preamble}\n\n{candidate}\n\n{test_code}\n\ncheck({entry_point})\n"
    # utf-8 explicitly: Windows' default encoding chokes on characters models
    # sometimes emit, raising a misleading SyntaxError.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=EXEC_TIMEOUT_S,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or "").strip()[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python humaneval_bench.py <label> [offset ...]")
    label = sys.argv[1]
    offsets = [int(a) for a in sys.argv[2:]] or DEFAULT_OFFSETS

    problems = load_problems(offsets)
    results = []
    passed = 0
    wall_start = time.time()

    for i, p in enumerate(problems, 1):
        task_id, prompt = p["task_id"], p["prompt"]
        entry_point, test_code = p["entry_point"], p["test"]

        t0 = time.time()
        try:
            response, finish_reason = query(prompt)
        except Exception as exc:
            results.append({"task_id": task_id, "pass": False,
                            "error": f"request failed: {exc}", "seconds": time.time() - t0})
            print(f"[{i}/{len(problems)}] {task_id}: FAIL (request error)")
            continue
        elapsed = time.time() - t0

        code = extract_code(response, entry_point)
        if code is None:
            note = "empty response (truncated mid-reasoning?)" if finish_reason == "length" \
                else "no valid code block extracted"
            results.append({"task_id": task_id, "pass": False, "error": note, "seconds": elapsed})
            print(f"[{i}/{len(problems)}] {task_id}: FAIL ({note}) [{elapsed:.1f}s]")
            continue

        ok, err = run_test(preamble_of(prompt, entry_point), code, test_code, entry_point)
        results.append({"task_id": task_id, "pass": ok, "error": err, "seconds": elapsed})
        passed += ok
        print(f"[{i}/{len(problems)}] {task_id}: {'PASS' if ok else 'FAIL'} [{elapsed:.1f}s]")

    total = len(problems)
    wall = time.time() - wall_start
    summary = {
        "label": label,
        "offsets": offsets,
        "slice_size": SLICE_SIZE,
        "passed": passed,
        "total": total,
        "pass_at_1": round(passed / total, 4) if total else 0.0,
        "wall_clock_seconds": round(wall, 1),
        "generation_seconds": round(sum(r["seconds"] for r in results), 1),
        "failures": [r["task_id"] for r in results if not r["pass"]],
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"humaneval_{label}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\npass@1: {passed}/{total} = {passed / total * 100:.1f}%")
    print(f"generation time: {summary['generation_seconds']}s "
          f"({summary['generation_seconds'] / total:.2f}s/problem)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
