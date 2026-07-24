# Qwen3-Coder-30B-A3B, locally, on a 12GB GPU

A local coding LLM wired into [opencode](https://opencode.ai) via
[llama.cpp](https://github.com/ggml-org/llama.cpp) — 30B total parameters
running entirely on consumer hardware, no cloud API involved.

**Contents:** [Overview](#overview) · [Hardware](#hardware) ·
[Quick start](#quick-start) · [Configuration reference](#configuration-reference) ·
[Benchmarks](#benchmarks) · [Quant comparison](#quant-comparison-intelligence-vs-speed) ·
[Model comparison](#model-comparison-qwen3-coder-30b-a3b-vs-qwen3-coder-next) ·
[Code correctness: HumanEval](#code-correctness-humaneval) ·
[Tuning](#tuning-ncmoe-and-kv-cache) ·
[Getting real numbers](#getting-real-numbers-llama-bench--llama-perplexity) ·
[Known issues](#known-issues) ·
[Troubleshooting](#troubleshooting-quick-reference) ·
[Files in this repo](#files-in-this-repo)

## Overview

[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
is a Mixture-of-Experts model: 30B total parameters, but only ~3.3B active
per token (8 of 128 experts fire per forward pass). That's what makes it
viable on a 12GB card — a dense 30B model needs every parameter touched on
every token, but an MoE model only needs the active experts moved through
compute. Most of the model can sit in system RAM while only the active
slice does GPU work, as long as the inference engine can split GPU/CPU
placement at the expert level.

It scores ~50% on SWE-bench Verified at Q4 and is purpose-built for
agentic/tool-calling coding workflows, which is what matters here — the
target is an agent loop (opencode), not a chatbot.

**llama.cpp over Ollama:** both were available, but llama.cpp's
`--n-cpu-moe` (`-ncmoe`) flag — which pins the MoE expert weights of the
first N layers to CPU RAM while keeping attention/shared layers and the KV
cache on GPU — is a tunable knob for exactly this hardware constraint.
Ollama can load the same GGUF, but its GPU/CPU split is a heuristic, not
something you dial in by hand.

## Hardware

| | |
|---|---|
| CPU | Intel i9, 12th gen |
| GPU | NVIDIA RTX 3080 Ti (12GB VRAM) |
| RAM | 64GB DDR4 |
| OS | Windows 10 |

## Quick start

1. **Get the model** — [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
   (17.7GB), Unsloth's dynamic quant:

   ```bash
   pip install -U huggingface_hub
   hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
     Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
     --local-dir C:\path\to\models
   ```

   Don't grab the smallest quant a downloader UI suggests by default — that
   recommendation is usually based on "does this fit entirely in VRAM,"
   which doesn't account for MoE expert offload to system RAM. `UD-Q4_K_XL`
   is the real sweet spot for a 12GB + 64GB combo.

2. **Get a CUDA-enabled llama.cpp.** This setup used a prebuilt one that
   [Unsloth Studio](https://unsloth.ai) had already installed
   (`llama-server.exe` alongside a `ggml-cuda.dll` under
   `llama.cpp/build/bin/Release`) — check if you have the same. Otherwise,
   build from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
   with `-DGGML_CUDA=ON`; see their
   [build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
   for the CUDA toolkit / cmake / MSVC prerequisites.

3. **Launch it** — [`start-qwen-coder.ps1`](./start-qwen-coder.ps1) has the
   tuned launch command (edit the two paths at the top for your machine):

   ```bash
   powershell -File start-qwen-coder.ps1
   ```

4. **Wire it into opencode** — copy [`opencode.example.jsonc`](./opencode.example.jsonc)
   into your opencode config (`~/.config/opencode/opencode.jsonc`), then:

   ```bash
   opencode run -m llama.cpp/qwen3-coder-30b-a3b "your prompt here"
   ```

5. **Verify tool-calling works** — a plain chat reply isn't proof the agent
   loop works; confirm it can actually drive tools:

   ```bash
   opencode run -m llama.cpp/qwen3-coder-30b-a3b \
     "List the files in the current directory using your tools, then tell me how many .gguf files are there."
   ```

   If it runs `ls` itself and answers from the real output, tool calling is
   wired correctly. For a fuller smoke test, ask it to build a small CLI
   tool with a test suite and have it run + fix the tests itself — that
   exercises the full loop: planning, multi-file writes, running commands,
   reading output, self-correcting.

## Configuration reference

**Launch flags** (see [`start-qwen-coder.ps1`](./start-qwen-coder.ps1) for
the full command):

| Flag | Why |
|---|---|
| `-ngl 999` | offload all non-MoE layers to GPU |
| `-ncmoe 27` | keep MoE expert weights of the first 27 layers on CPU RAM (tuned for this hardware — see [Tuning](#tuning-ncmoe-and-kv-cache)) |
| `-ctk q8_0 -ctv q8_0` | quantized KV cache — frees VRAM, no measured quality cost (see below) |
| `-fa on` | flash attention |
| `--jinja` | use the model's embedded chat template — required for tool-calling to format correctly |
| `--no-mmap` | minor perf win once tensor placement is being overridden |

**opencode provider** ([`opencode.example.jsonc`](./opencode.example.jsonc)):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3-coder-30b-a3b": {
          "name": "Qwen3-Coder-30B-A3B (local)",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

## Benchmarks

Standardized numbers via `llama-bench` (`pp512`/`tg128`, 3+ repetitions with
warmup — see [Getting real numbers](#getting-real-numbers-llama-bench--llama-perplexity)
for why this isn't `bench.py`):

| config | pp512 (tok/s) | tg128 (tok/s) | VRAM free (idle) |
|---|---|---|---|
| `-ncmoe 30`, KV f16 | 366.18 ± 14.60 | 28.09 ± 0.99 | 643MB |
| `-ncmoe 34`, KV f16 | 340.17 ± 25.09 | 26.18 ± 1.54 | 1805MB |
| `-ncmoe 30`, KV q8_0 | 409.67 ± 21.45 | 29.18 ± 1.30 | 1518MB |
| **`-ncmoe 27`, KV q8_0 (current default)** | **456.77 ± 37.24** | **31.87 ± 1.73** | **1161MB** |

**Quality (perplexity):** 8.8606 ± 0.23686 for `UD-Q4_K_XL`, measured
against 50 chunks (512 tokens each) of the standard
[wikitext-2-raw](https://huggingface.co/datasets/ggml-org/ci) corpus.
Perplexity depends only on the quant, not `-ncmoe` (which just changes
*where* weights are computed, not the weights themselves).

Reproduce with `python bench.py` against a running server for a quick,
noisier read, or `llama-bench.exe` for the rigorous version (command in
[Getting real numbers](#getting-real-numbers-llama-bench--llama-perplexity)).

## Quant comparison: intelligence vs. speed

Same methodology, three [Unsloth Dynamic](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
quants, each with `-ncmoe` independently tuned for safe VRAM headroom
(KV cache q8_0 throughout — see [Tuning](#tuning-ncmoe-and-kv-cache) for
what "safe" means here):

| Quant | Size | `-ncmoe` | VRAM free | pp512 | tg128 | Perplexity (lower = better) |
|---|---|---|---|---|---|---|
| `UD-Q3_K_XL` | 13.8GB | 24 | 1739MB | 548.07 ± 37.49 | 33.12 ± 3.35 | 8.9187 ± 0.237 |
| **`UD-Q4_K_XL` (default)** | **17.7GB** | **27** | **1161MB** | **456.77 ± 37.24** | **31.87 ± 1.73** | **8.8606 ± 0.237** |
| `UD-Q5_K_XL` | 21.7GB | 32 | 1011MB | 248.78 ± 27.39 | 20.52 ± 1.73 | 8.7457 ± 0.233 |

The curve isn't linear. Q3 → Q4 costs a small amount of speed (33.1 → 31.9
tok/s gen) for a real quality gain. Q4 → Q5 costs a *much* bigger speed hit
(31.9 → 20.5 tok/s gen, ~36% slower) for a similar-sized quality gain — the
returns drop off past Q4. That's the actual reason Q4 is the default here:
not just Unsloth's own recommendation, but a real head-to-head on this
hardware showing it sits at the knee of the curve. Q5 is worth it if
quality matters more than interactivity for a given task; Q3 is worth it if
you want the fastest possible loop and can tolerate a small quality drop.

## Model comparison: Qwen3-Coder-30B-A3B vs. Qwen3-Coder-Next

Same `UD-Q4_K_XL` quant tier, same tuning methodology, two different base
models. [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
(80B total, still only 3B active) claims 70.6% on SWE-bench Verified versus
this baseline's ~50%. Worth checking whether that translates into something
usable on this hardware, not just a leaderboard number:

| Model | Total / Active | Size | `-ncmoe` | VRAM free | pp512 | tg128 | Perplexity |
|---|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B (default)** | 30.5B / 3.3B | 17.7GB | 27 | 1161MB | **456.77 ± 37.24** | **31.87 ± 1.73** | 8.8606 ± 0.237 |
| Qwen3-Coder-Next | 80B / 3B | 49.6GB | 41 | 863MB | 74.45 ± 23.85 | 22.12 ± 0.59 | **6.6825 ± 0.172** |

The quality gap is real, not just a paper claim — PPL 6.68 vs. 8.86 is a
large, unambiguous improvement, consistent with the SWE-bench delta. But
it isn't free, and the cost shows up somewhere non-obvious: **generation
speed only drops ~31% (32 → 22 tok/s), but prompt processing drops ~6x**
(457 → 74 tok/s). Both models activate roughly the same ~3B params per
token, so token-by-token generation cost stays close. Prompt processing is
different — it touches the full CPU-resident expert set across the whole
batch, and Next has far more of its 49.6GB sitting in system RAM at any
given offload setting than the 30B model's 17.7GB does. More weight moving
through RAM per prompt means prompt processing takes the brunt of the
slowdown, generation barely notices.

That distinction matters specifically for opencode: agentic coding sends
large, repeated context (file contents, tool output, diffs) on every turn,
which is a prompt-processing-heavy workload, not a generation-heavy one. A
6x prompt-processing regression will be felt on every turn of a real
session far more than the tg128 number alone suggests. On a long
(~5800-token) stress prompt, Next's own throughput improves to ~179 tok/s
(batching amortizes the RAM transfer better over a longer prompt) but is
still ~9.6x slower than the 30B model's ~1712 tok/s on the same prompt —
batching helps Next's own efficiency, it doesn't close the gap.

**Verdict:** not a clean win, a real tradeoff. Qwen3-Coder-30B-A3B stays
the default for fast iterative loops. Qwen3-Coder-Next is worth switching
to per-task when a problem is hard enough that the quality gap matters more
than turnaround time — not as a wholesale replacement. See
[Code correctness](#code-correctness-humaneval) below for where that gap
actually does and doesn't show up.

## Code correctness: HumanEval

Perplexity and SWE-bench are useful, but neither is "does this model write
code that actually runs and passes its tests." Neither is opencode
tool-calling — that proves the *plumbing* works, not that the *code* is
correct. [`humaneval_bench.py`](./humaneval_bench.py) closes that gap: it
sends real [HumanEval](https://github.com/openai/human-eval) problems
(bundled here as [`HumanEval.jsonl.gz`](./HumanEval.jsonl.gz), 164 official
problems) to a running server, extracts the generated function, and
executes it against HumanEval's actual unit tests in a sandboxed subprocess
— `pass@1`, temperature 0 for determinism.

```bash
python humaneval_bench.py        # first 20 problems
python humaneval_bench.py 140    # 20 problems starting at index 140
```

Tested 40 problems total per model (problems 0-19, then 140-159 — the
second slice specifically to check whether an "easy" subset was hiding a
real gap):

| Model | Problems 0-19 | Problems 140-159 | Total |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (default) | 20/20 | 19/20 | **39/40 (97.5%)** |
| Qwen3-Coder-Next | 20/20 | 19/20 | **39/40 (97.5%)** |

Identical. Both models failed the exact same single problem
(`HumanEval/145`, a digit-sum sorting task with a genuinely ambiguous sign
convention in its spec — a known-tricky problem, not a random miss).

**This is the actual finding, and it matters:** the quality gap that shows
up clearly in perplexity (6.68 vs. 8.86) and is claimed on SWE-bench (70.6%
vs. ~50%) does *not* show up on isolated, well-specified, single-function
coding tasks — both models are already saturated there. That's consistent
with what SWE-bench actually measures versus what HumanEval measures:
SWE-bench is repo-scale, ambiguous, multi-file, real-world bug fixing;
HumanEval is a clean function signature and docstring. The gap between
these two models is real, but it lives in the harder, more open-ended,
more context-heavy end of coding tasks — not in "can it implement a
clearly-specified function," which the faster/cheaper model already
handles just as well.

Two real harness bugs got caught and fixed during this: extracted code
needs the original prompt's import preamble prepended before execution (a
model completing just the function body doesn't repeat `from typing import
List`), and the temp file needs explicit UTF-8 encoding on Windows or
certain generated characters break execution with an unrelated-looking
`SyntaxError`. Both looked like model failures at first and were actually
harness bugs — worth being suspicious of your own eval code before
concluding the model failed.

## Tuning: `-ncmoe` and KV cache

Tuning is empirical — pick values, load the model, check headroom under
**real generation** (not just idle), adjust. [`bench.py`](./bench.py) hits
a running server's `/v1/chat/completions` endpoint and reports the
server's own `prompt tok/s` / `gen tok/s`.

First pass, KV cache at default `f16`:

| `-ncmoe` | VRAM free (idle) | Result |
|---|---|---|
| 20 | 358MB | **hung, never returned** |
| 24 | 216MB | **hung, never returned** |
| 30 | 643MB | stable, 27.1 tok/s |
| 34 | 1805MB | stable, 21.6 tok/s (slower) |

Values below ~30 didn't just run slower — they **silently hung** on real
generation requests, sometimes after correctly answering a health check.
This is a Windows/WDDM-specific failure mode: when a CUDA allocation
doesn't fit in remaining VRAM, WDDM can quietly page to system memory
instead of returning a clean out-of-memory error, so the request just never
completes. `nvidia-smi` shows the process still holding GPU memory the
whole time — nothing crashes, it just never comes back. (On Linux this
would likely fail loudly with a CUDA OOM instead — arguably easier to
debug than a silent hang.) `30` shipped as the first working config, stable
under a ~4800-token stress prompt with VRAM barely moving (the KV cache is
pre-allocated at startup for the full context size, so it doesn't grow with
usage).

**Second pass:** `-ncmoe` isn't the only lever. Quantizing the KV cache
(`-ctk q8_0 -ctv q8_0`, roughly halving its footprint) was tried expecting
a speed/quality tradeoff and got neither — it was faster *and* freed more
VRAM at the same `-ncmoe`:

| config | VRAM free (idle) | gen tok/s |
|---|---|---|
| `-ncmoe 30`, KV f16 | 643MB | 27.1 |
| `-ncmoe 30`, KV q8_0 | 1518MB | 29.1 |

That headroom allowed `-ncmoe` to drop again:

| config | VRAM free (idle) | gen tok/s |
|---|---|---|
| `-ncmoe 24`, KV q8_0 | 210MB | 33.4 (completed, but see below) |
| `-ncmoe 27`, KV q8_0 | 1161MB | 31.5 |

`-ncmoe 24` completed a benchmark run, but 210MB free is the same range
that hung above — one successful run isn't proof of stability, it's a coin
flip that happened to land right. `-ncmoe 27` is the trustworthy one:
re-verified with the same stress prompt, VRAM held steady, response under
4 seconds. That's the config now in use — faster than every earlier
attempt, with nearly double the headroom of the original.

**Takeaways:**
- Don't just check idle VRAM after model load — test with an actual
  generation request, and leave more headroom than looks strictly
  necessary.
- Don't tune one lever in isolation. KV cache quantization and MoE offload
  interact; freeing memory on one axis buys room to push the other
  further, and the combined optimum wasn't reachable by tuning `-ncmoe`
  alone.

## Getting real numbers: `llama-bench` & `llama-perplexity`

<details>
<summary>The prebuilt install from step 2 only shipped <code>llama-server.exe</code> — here's how the other tools got added without a build toolchain</summary>

`llama-bench` and `llama-perplexity` existed as internal `-impl.dll` files
with no matching `.exe` wrapper, and no `cmake` was installed on this
machine to build them from source. A metadata file dropped alongside the
build (`UNSLOTH_PREBUILT_INFO.json`) recorded exactly which upstream
release it came from (`unslothai/llama.cpp`, tag `b10069-mix-fb3d4ca`).
Downloading that same release's zip from GitHub and pulling the two `.exe`
files out of it gives tools that are guaranteed DLL/ABI-compatible with the
CUDA backend already installed — no build required, no version mismatch
risk. If you built from source in step 2 instead, you'll have both tools
already and can skip this.

</details>

```bash
llama-bench.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -ngl 999 -ncmoe 27 -fa on -ctk q8_0 -ctv q8_0

llama-perplexity.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -f wiki.test.raw -ngl 999 -ncmoe 30 -fa on -c 512 --chunks 50
```

(50 chunks is a subset of the full ~550-chunk wikitext-2 test set — a full
run takes ~12+ minutes; this is a faster representative sample.)

## Known issues

<details>
<summary><b>CUDA silently falls back to CPU</b> — server loads and responds, but VRAM usage never moves</summary>

The server started fine, loaded the model, and responded to requests — but
the whole 17.7GB sat in system RAM instead of VRAM. No error was printed.

Diagnosis:

```bash
llama-server.exe --list-devices
# Available devices:
#   (nothing listed)
```

No CUDA device was detected at all, despite `ggml-cuda.dll` being present.
Testing `LoadLibrary` on that DLL directly (a small PowerShell P/Invoke
snippet) returned **Win32 error 126** — `ERROR_MOD_NOT_FOUND`, meaning the
DLL itself loaded but one of *its* dependencies couldn't be found. Walking
the import table (`pefile` in Python) showed the real culprits:
`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` — the Visual C++
Redistributable, not installed on this machine.

**Fix without a system-wide installer:** Ollama ships its own copies of
these runtime DLLs (plus matching `cublas`/`cudart`) alongside its bundled
llama.cpp backend. Copying them from Ollama's install directory into the
llama.cpp build folder fixed it immediately — no admin install, no reboot:

```
<ollama install dir>\lib\ollama\cuda_v13\cublas64_13.dll
<ollama install dir>\lib\ollama\cuda_v13\cublasLt64_13.dll
<ollama install dir>\lib\ollama\cuda_v13\msvcp140.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140_1.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140_threads.dll
```
→ copied into `llama.cpp\build\bin\Release\` (same folder as
`llama-server.exe`). After that, `--list-devices` showed the 3080 Ti
correctly.

If you hit this and don't have Ollama installed, install the
[Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
directly.

</details>

<details>
<summary><b>Requests hang forever</b> — <code>/health</code> still returns 200, generation never completes</summary>

See [Tuning](#tuning-ncmoe-and-kv-cache) above — this happens when VRAM
headroom drops under ~500MB. It's a Windows/WDDM-specific failure mode:
CUDA allocations that don't fit get silently paged to system memory instead
of erroring, so the request just never returns. Fix: raise `-ncmoe`, or
quantize the KV cache to free headroom without raising `-ncmoe` at all —
then re-test with `bench.py` against a real generation request, not just
idle VRAM after load.

</details>

<details>
<summary><b>"Hang" that's actually just a slow model</b> — happened while tuning Qwen3-Coder-Next, worth checking before assuming WDDM paging</summary>

While tuning a much larger model (80B total vs. 30.5B), several `-ncmoe`
values that had *plenty* of VRAM headroom (as much as 2.8GB free — nowhere
near the ~500MB danger zone above) still appeared to hang on `bench.py`.
Turned out they weren't hung at all: `bench.py`'s long-prompt test alone
was taking 30-40 seconds per request on this bigger, more CPU-offloaded
model, and the benchmark script's own timeout was tuned for a model 6-9x
faster. Confirmed by re-running the exact same request with a much longer
timeout — it completed cleanly with real numbers.

**Lesson:** before concluding a config is unstable, check whether the
timeout you're using actually fits the model's expected speed. A genuinely
larger, more CPU-offloaded model can legitimately take much longer per
request without anything being wrong.

</details>

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `--list-devices` shows nothing | CUDA backend DLL failed to load | Missing VC++ Redistributable DLLs — see Known issues |
| VRAM barely used, huge RAM usage, model "works" but slow | Silently running CPU-only despite `-ngl` | Same as above — confirm with `--list-devices` before assuming offload is active |
| Request hangs forever, `/health` still returns 200 | `-ncmoe` too low, VRAM headroom under ~500MB | Raise `-ncmoe`, or quantize the KV cache to free headroom instead |
| `llama-bench`/`llama-perplexity` missing, only `-impl.dll` files present | Prebuilt install only shipped the tools it uses directly | Pull matching `.exe` files from the release named in `UNSLOTH_PREBUILT_INFO.json` (or build from source, which includes everything) |
| opencode can't call tools correctly | Chat template mismatch | Make sure `--jinja` is enabled |

## Files in this repo

| File | Purpose |
|---|---|
| [`start-qwen-coder.ps1`](./start-qwen-coder.ps1) | launch script, Qwen3-Coder-30B-A3B (default, fast) |
| [`start-qwen-coder-next.ps1`](./start-qwen-coder-next.ps1) | launch script, Qwen3-Coder-Next (slower, smarter) |
| [`opencode.example.jsonc`](./opencode.example.jsonc) | opencode provider config to copy in, both models |
| [`bench.py`](./bench.py) | quick HTTP-based tok/s benchmark against a running server |
| [`humaneval_bench.py`](./humaneval_bench.py) | pass@1 code-correctness eval against a running server |
| [`HumanEval.jsonl.gz`](./HumanEval.jsonl.gz) | official HumanEval dataset (164 problems), bundled for reproducibility |
