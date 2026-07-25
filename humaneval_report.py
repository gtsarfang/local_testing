"""Read results/humaneval_*.json and print the README-ready markdown table.

Usage: python humaneval_report.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Display order/names for the README table.
LABELS = [
    ("qwen3-coder-30b-a3b", "**Qwen3-Coder-30B-A3B (default)**"),
    ("qwen3-coder-next", "Qwen3-Coder-Next"),
    ("gpt-oss-20b", "gpt-oss-20b"),
    ("gemma-4-26b-a4b", "Gemma 4 26B A4B QAT"),
]


def main():
    rows = []
    for label, display in LABELS:
        path = RESULTS_DIR / f"humaneval_{label}.json"
        if not path.exists():
            print(f"missing: {path}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append((display, d["passed"], d["total"], d["generation_seconds"], d["failures"]))

    print("| Model | HumanEval (pass@1) | Generation time |")
    print("|---|---|---|")
    for display, passed, total, secs, _ in rows:
        pct = passed / total * 100 if total else 0
        print(f"| {display} | {passed}/{total} ({pct:.1f}%) | {secs:.1f}s ({secs/total:.2f}s/problem) |")

    print()
    all_failures = [set(f) for *_, f in rows]
    if all_failures and all(f == all_failures[0] for f in all_failures):
        print(f"All models failed the identical set: {sorted(all_failures[0])}")
    else:
        for (display, *_rest, failures) in rows:
            print(f"{display} failures: {sorted(failures)}")


if __name__ == "__main__":
    main()
