import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
DATA_PATH = Path(__file__).parent / "HumanEval.jsonl.gz"
N_PROBLEMS = 20
OFFSET = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TIMEOUT_S = 10

SYSTEM = (
    "Complete the given Python function. Respond with ONLY a single Python "
    "code block containing the complete function (signature + body, "
    "including the docstring). No explanation."
)


def load_problems(n, offset=0):
    with gzip.open(DATA_PATH, "rt") as f:
        lines = [json.loads(l) for l in f]
    return lines[offset:offset + n]


def query(prompt, max_tokens=512):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def extract_code(response, entry_point):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = m[0] if m else response
    if f"def {entry_point}" not in code:
        return None
    return code


def run_test(preamble, candidate_code, test_code, entry_point):
    full = preamble + "\n\n" + candidate_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT_S,
            encoding="utf-8", errors="replace",
        )
        return result.returncode == 0, result.stderr[-300:] if result.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        import os
        os.unlink(path)


def main():
    problems = load_problems(N_PROBLEMS, OFFSET)
    passed = 0
    results = []
    for i, p in enumerate(problems):
        task_id = p["task_id"]
        prompt = p["prompt"]
        entry_point = p["entry_point"]
        test_code = p["test"]

        t0 = time.time()
        try:
            response = query(prompt)
        except Exception as e:
            results.append((task_id, False, f"request failed: {e}"))
            print(f"[{i+1}/{len(problems)}] {task_id}: FAIL (request error)")
            continue
        elapsed = time.time() - t0

        code = extract_code(response, entry_point)
        if code is None:
            results.append((task_id, False, "no valid code block extracted"))
            print(f"[{i+1}/{len(problems)}] {task_id}: FAIL (bad extraction) [{elapsed:.1f}s]")
            continue

        preamble_lines = []
        for line in prompt.splitlines():
            if line.startswith(f"def {entry_point}"):
                break
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines)

        ok, err = run_test(preamble, code, test_code, entry_point)
        results.append((task_id, ok, err))
        status = "PASS" if ok else f"FAIL ({err[:60]})"
        print(f"[{i+1}/{len(problems)}] {task_id}: {status} [{elapsed:.1f}s]")
        if ok:
            passed += 1

    print(f"\npass@1: {passed}/{len(problems)} = {passed/len(problems)*100:.1f}%")
    with open(f"humaneval_results_offset{OFFSET}.json", "w") as f:
        json.dump({"pass_at_1": passed / len(problems), "passed": passed, "total": len(problems),
                    "results": [{"task_id": t, "pass": ok, "error": e} for t, ok, e in results]}, f, indent=2)


if __name__ == "__main__":
    main()
