import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

# ~2000 token prompt, fixed across runs for a fair comparison
_lines = "".join(f"def example_function_{i}():\n    return {i}\n\n" for i in range(400))
LONG_PROMPT = "Summarize what this code does in one sentence:\n\n" + _lines

SHORT_PROMPT = "Write a Python function that checks if a number is prime. Keep it concise."


def request(prompt, max_tokens):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    wall = time.time() - t0
    return body["timings"], wall


def avg(vals):
    return sum(vals) / len(vals)


def main():
    pp_runs = []
    tg_runs = []

    # prompt-processing heavy run (long prompt, short output)
    # prefix a unique marker each time so the server's prompt cache can't
    # shortcut the second run and skew the average
    for i in range(2):
        t, _ = request(f"[run {i}]\n" + LONG_PROMPT, 50)
        pp_runs.append(t["prompt_per_second"])

    # generation heavy run (short prompt, longer output)
    for i in range(2):
        t, _ = request(f"[run {i}]\n" + SHORT_PROMPT, 300)
        tg_runs.append(t["predicted_per_second"])

    result = {
        "prompt_tok_per_sec": round(avg(pp_runs), 1),
        "gen_tok_per_sec": round(avg(tg_runs), 1),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
