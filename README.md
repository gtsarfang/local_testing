# Local coding LLMs on a 12GB GPU

Setting up and comparing local coding models on constrained consumer
hardware, wired into [opencode](https://opencode.ai) via
[llama.cpp](https://github.com/ggml-org/llama.cpp) — no cloud API involved.

*Benchmarked 2026-07-24 · llama.cpp `b10069` (`72051453a`) · NVIDIA driver
610.74 / CUDA 13.3 · Windows 10. Windows/WDDM-specific behavior is called
out explicitly where it matters — see [Known issues](#known-issues).*

## TL;DR

| Model | pp512 | tg128 | Perplexity | HumanEval (n=80) | Wall-clock (80 problems) |
|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B (default)** | 456.77 | 31.87 | 8.86 | 77/80 (96.2%) | **455.6s** |
| Qwen3-Coder-Next | 74.45 | 22.12 | **6.68** | **79/80 (98.8%)** | 1245.4s |
| gpt-oss-20b | **1393.10** | **71.43** | n/a¹ | 75/80 (93.8%) | 774.8s |

*(full table with error margins, VRAM, and config in
[Model comparison](#model-comparison).)*

**Which one do I use:**
- Fast iterative loop, most day-to-day tasks → **Qwen3-Coder-30B-A3B** (the default)
- A problem hard enough that quality matters more than turnaround time → **Qwen3-Coder-Next**
- Raw tok/s matters more than wall-clock time-to-answer → **gpt-oss-20b** (reasoning model — read the caveat before picking it for interactive use)

**Four things worth knowing before you read further:**
1. **A bigger sample changed the finding, and that's the point.** At n=40,
   all three models scored identically on HumanEval. At n=80, a real (if
   modest) gap appeared: Next 98.8%, default 96.2%, gpt-oss 93.8%. The
   smaller sample wasn't wrong, it just wasn't enough — see
   [Findings](#findings) for what that gap actually looks like next to the
   70.6% SWE-bench claim.
2. **One problem breaks every model tested, at both sample sizes.**
   `HumanEval/145` is the *only* failure all three models share — strong,
   repeated evidence that its spec is genuinely ambiguous, not that these
   models are weak.
3. **tok/s ≠ wall-clock time.** gpt-oss-20b is the fastest model here by a
   wide margin, but it's a reasoning model that spends tokens thinking
   before answering — it took 1.7x longer than the default to finish the
   same 80 problems, and Next (despite similar tok/s to the default) took
   2.7x longer.
4. **Every number here got sanity-checked before being trusted**, including
   this section's own — the "identical at n=40" finding didn't survive
   a bigger sample, and one harness fix (reasoning-model token budget)
   turned out to be a partial fix, not a complete one, once tested at
   scale. See [Benchmark methodology](#benchmark-methodology).

¹ perplexity is a documented-broken metric for gpt-oss specifically, not a
real quality number — see [Known issues](#known-issues).

## Contents

[TL;DR](#tldr) · [What makes this work](#what-makes-this-work) ·
[Hardware & software](#hardware--software) · [Quick start](#quick-start) ·
[Model comparison](#model-comparison) ·
[Quant comparison](#quant-comparison-intelligence-vs-speed) ·
[Tuning](#tuning--ncmoe-and-kv-cache) ·
[Benchmark methodology](#benchmark-methodology) ·
[Limitations](#limitations) · [Known issues](#known-issues) ·
[Troubleshooting](#troubleshooting-quick-reference) ·
[Files in this repo](#files-in-this-repo)

## What makes this work

Every model compared here is a **Mixture-of-Experts** model: tens of
billions of total parameters, but only a few billion active per token (a
handful of experts fire per forward pass, out of many more available).
That's what makes any of them viable on a 12GB card — a dense model needs
every parameter touched on every token, but an MoE model only needs the
active experts moved through compute. Most of the model can sit in system
RAM while only the active slice does GPU work, as long as the inference
engine can split GPU/CPU placement at the expert level.

**llama.cpp over Ollama:** both were available, but llama.cpp's
`--n-cpu-moe` (`-ncmoe`) flag — which pins the MoE expert weights of the
first N layers to CPU RAM while keeping attention/shared layers and the KV
cache on GPU — is a tunable knob for exactly this hardware constraint.
Ollama can load the same GGUF, but its GPU/CPU split is a heuristic, not
something you dial in by hand.

The default, [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
(30B total, ~3.3B active), scores ~50% on SWE-bench Verified at Q4 and is
purpose-built for agentic/tool-calling coding workflows — which is what
matters here, since the target is an agent loop (opencode), not a chatbot.
See [Model comparison](#model-comparison) for how the alternatives stack up.

## Hardware & software

| | |
|---|---|
| CPU | Intel i9, 12th gen |
| GPU | NVIDIA RTX 3080 Ti (12GB VRAM) |
| RAM | 64GB DDR4 |
| OS | Windows 10 |
| llama.cpp | build `b10069` (`72051453a`) |
| NVIDIA driver | 610.74 / CUDA 13.3 |

Several of the [Known issues](#known-issues) below are **Windows/WDDM
-specific failure modes** (silent CUDA fallback, silent hangs under low
VRAM headroom) — on Linux, the same misconfigurations would likely fail
loudly instead of silently, which is easier to debug but means those
specific sections may not transfer directly.

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
   into your opencode config (`~/.config/opencode/opencode.jsonc`) — it
   defines all three models compared here as a single `llama.cpp` provider,
   pointing at `http://127.0.0.1:8080/v1`. Then:

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

## Model comparison

Every model here is measured the same three ways: **speed** (`llama-bench`
pp512/tg128), **quality** (perplexity on wikitext-2-raw), and
**correctness** (`pass@1` on real [HumanEval](https://github.com/openai/human-eval)
problems, executed in a sandboxed subprocess — see
[methodology](#benchmark-methodology)). Same quant tier where possible,
same tuning approach (`-ncmoe` + KV cache individually tuned per model for
safe VRAM headroom, per [Tuning](#tuning--ncmoe-and-kv-cache)).

**Results:**

| Model | pp512 | tg128 | Perplexity | HumanEval | Wall-clock |
|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B (default)** | 456.77 ± 37.24 | 31.87 ± 1.73 | 8.8606 ± 0.237 | 77/80 (96.2%) | **455.6s** |
| [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | 74.45 ± 23.85 | 22.12 ± 0.59 | **6.6825 ± 0.172** | **79/80 (98.8%)** | 1245.4s |
| [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) | **1393.10 ± 70.51** | **71.43 ± 0.31** | not comparable¹ | 75/80 (93.8%) | 774.8s |

**Tuned config:**

| Model | Total / Active | Size | `-ncmoe` | VRAM free |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 30.5B / 3.3B | 17.7GB | 27 | 1161MB |
| Qwen3-Coder-Next | 80B / 3B | 49.6GB | 41 | 863MB |
| gpt-oss-20b | 21B / 3.6B | 11.9GB | 4 | 1343MB |

¹ gpt-oss-20b's perplexity is a known-broken measurement for this model
family — see [Known issues](#known-issues).

### Findings

**A bigger sample changed the finding — that's the headline, not a footnote.**
At n=40 (two 20-problem slices), all three models scored identically:
39/40, same single failure, every one of them. That looked like a clean
"HumanEval is saturated for all three" result and got written up as one.
Doubling to n=80 (four spread slices) broke that: Qwen3-Coder-Next actually
pulled ahead (79/80, 98.8%), the default held at 77/80 (96.2%), and
gpt-oss-20b came in at 75/80 (93.8%). The smaller sample wasn't
*measured wrong* — it was just too small to see a real, if modest, gap
that was there the whole time. Nothing later in this doc is exempt from
the same risk: n=80 of 164 is bigger, not definitive (see
[Limitations](#limitations)).

**The gap is real, but nowhere near SWE-bench-sized.** Next going 77→79
correct (out of 80) versus Qwen3-Coder-30B-A3B is a real, directly-measured
edge — first actual correctness evidence for the perplexity gap (6.68 vs.
8.86), not just a proxy. But it's a 2.6-point gap on well-specified
function-level problems, next to an 70.6%-vs-~50% SWE-bench Verified claim
between the same two models — over an order of magnitude larger. Taken
together, the honest read is: Next is somewhat better even at the easy
end, and *dramatically* better at repo-scale, ambiguous, multi-file
work (which is what SWE-bench actually measures and this repo has not
independently verified). Don't round "somewhat better here" up to "as much
better as the leaderboard claims."

**One problem breaks every model, at both sample sizes.** `HumanEval/145`
(a digit-sum-sorting task with a genuinely ambiguous sign convention) is
the *only* failure Qwen3-Coder-Next has at all, and it's shared by every
other model tested, at n=40 and again at n=80. Three different
vendors/architectures repeatedly landing on the identical failure is much
stronger evidence the spec itself is ambiguous than any one model's miss
would be.

**Even where Next wins, it's expensive to get there.** At n=80,
Qwen3-Coder-30B-A3B finished in 455.6s wall-clock (5.70s/problem avg);
Next took 1245.4s (15.57s/problem avg) — **2.73x as long** for a 2-point
correctness edge. That ratio grew from 2.02x at n=40, because the larger,
more varied sample includes problems where Next visibly reasons/plans much
longer (several individual responses ran 30-40s). This is the number that
most directly answers "how long do I actually wait," and it's the
practical cost of that correctness edge on this hardware.

**The speed cost lands somewhere non-obvious.** Generation speed only
drops ~31% (32 → 22 tok/s) going to Next, but prompt processing drops ~6x
(457 → 74 tok/s). Both models activate roughly the same ~3B params per
token, so token-by-token generation cost stays close — but prompt
processing touches the *full* CPU-resident expert set across the whole
batch, and Next has far more of its 49.6GB sitting in system RAM at any
given offload setting than the 30B model's 17.7GB does. That matters
specifically for opencode: agentic coding sends large, repeated context
(file contents, tool output, diffs) every turn — a prompt-processing-heavy
workload, not a generation-heavy one — so a 6x prompt-processing
regression is felt far more than the tg128 number alone suggests. On a
long (~5800-token) stress prompt Next's own throughput improves to ~179
tok/s (batching amortizes the RAM transfer better over a longer prompt),
but that's still ~9.6x slower than the 30B model's ~1712 tok/s on the same
prompt — batching helps Next's own efficiency, it doesn't close the gap to
the smaller model. **Lesson for comparing any future model here:** check
pp512 and tg128 separately, they can diverge sharply for models of very
different total size, and pick whichever one matches your actual workload
(agentic/context-heavy vs. short back-and-forth).

**Raw tok/s isn't wall-clock time, if the model reasons before answering.**
gpt-oss-20b posted the fastest tok/s of any model here by a wide margin
(1393 pp512, 71 tg128 — roughly 3x the default's numbers) but at n=80 took
**1.70x longer than the default** to actually finish the same problem set
(774.8s vs. 455.6s, 9.68s/problem avg). It's a reasoning model: llama.cpp
surfaces its chain-of-thought in a separate `reasoning_content` field, and
it can spend a large number of tokens thinking before it ever writes the
answer — one problem took 57.3 seconds despite generating at 71 tok/s the
whole time, because nearly all of that time was reasoning, not answering.
**Lesson:** tok/s tells you how fast the model can generate, not how long
you'll actually wait for a finished answer — those are the same thing for
a non-reasoning model and can diverge a lot for one that isn't.

**The reasoning-budget fix from n=40 was partial, not complete.** Raising
`max_tokens` to 4096 took gpt-oss-20b from 17/20 to 20/20 on the original
small sample (see [Benchmark methodology](#benchmark-methodology)) — but
at n=80, `HumanEval/1` *still* came back with an empty `content` field
after 59 seconds of reasoning, the same truncation symptom, at the same
4096-token budget. Some problems apparently need more than 4096 reasoning
tokens for this model. The other three gpt-oss-20b failures
(`/10`, `/47`, `/113`) were genuine wrong answers, not truncation — worth
distinguishing, since only one of the four is a harness/budget limitation
and the other three are real capability misses. **Lesson:** a fix
validated on 20 problems isn't guaranteed to generalize to 80 — verify
harness fixes at the same scale you're about to trust the results at.

## Quant comparison: intelligence vs. speed

Same methodology, three [Unsloth Dynamic](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
quants of the default model, each with `-ncmoe` independently tuned for
safe VRAM headroom (KV cache q8_0 throughout — see
[Tuning](#tuning--ncmoe-and-kv-cache) for what "safe" means here):

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

## Tuning: `-ncmoe` and KV cache

Tuning is empirical — pick values, load the model, check headroom under
**real generation** (not just idle), adjust. [`bench.py`](./bench.py) hits
a running server's `/v1/chat/completions` endpoint and reports the
server's own `prompt tok/s` / `gen tok/s`; final numbers are validated with
`llama-bench` (see [Benchmark methodology](#benchmark-methodology)).

**First pass**, KV cache at default `f16`:

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

**Final validated numbers** for the shipped config, via `llama-bench`
(`pp512`/`tg128`, 3+ repetitions with warmup):

| config | pp512 (tok/s) | tg128 (tok/s) | VRAM free (idle) |
|---|---|---|---|
| `-ncmoe 30`, KV f16 | 366.18 ± 14.60 | 28.09 ± 0.99 | 643MB |
| `-ncmoe 34`, KV f16 | 340.17 ± 25.09 | 26.18 ± 1.54 | 1805MB |
| `-ncmoe 30`, KV q8_0 | 409.67 ± 21.45 | 29.18 ± 1.30 | 1518MB |
| **`-ncmoe 27`, KV q8_0 (shipped default)** | **456.77 ± 37.24** | **31.87 ± 1.73** | **1161MB** |

Perplexity (8.8606 ± 0.23686 for `UD-Q4_K_XL`) depends only on the quant,
not `-ncmoe` (which just changes *where* weights are computed, not the
weights themselves).

**Takeaways:**
- Don't just check idle VRAM after model load — test with an actual
  generation request, and leave more headroom than looks strictly
  necessary.
- Don't tune one lever in isolation. KV cache quantization and MoE offload
  interact; freeing memory on one axis buys room to push the other
  further, and the combined optimum wasn't reachable by tuning `-ncmoe`
  alone.

## Benchmark methodology

**Getting `llama-bench` and `llama-perplexity`:**

<details>
<summary>The prebuilt install from Quick start step 2 only shipped <code>llama-server.exe</code> — here's how the other tools got added without a build toolchain</summary>

`llama-bench` and `llama-perplexity` existed as internal `-impl.dll` files
with no matching `.exe` wrapper, and no `cmake` was installed on this
machine to build them from source. A metadata file dropped alongside the
build (`UNSLOTH_PREBUILT_INFO.json`) recorded exactly which upstream
release it came from (`unslothai/llama.cpp`, tag `b10069-mix-fb3d4ca`).
Downloading that same release's zip from GitHub and pulling the two `.exe`
files out of it gives tools that are guaranteed DLL/ABI-compatible with the
CUDA backend already installed — no build required, no version mismatch
risk. If you built from source instead, you'll have both tools already and
can skip this.

</details>

```bash
llama-bench.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -ngl 999 -ncmoe 27 -fa on -ctk q8_0 -ctv q8_0

llama-perplexity.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -f wiki.test.raw -ngl 999 -ncmoe 30 -fa on -c 512 --chunks 50
```

Perplexity uses 50 chunks (512 tokens each) of the standard
[wikitext-2-raw](https://huggingface.co/datasets/ggml-org/ci) corpus — a
subset of the full ~550-chunk test set (a full run takes ~12+ minutes;
this is a faster representative sample).

**HumanEval (code correctness):**

```bash
python humaneval_bench.py <label>              # standard 80-problem sample (4 slices of 20)
python humaneval_bench.py <label> 0 40         # custom slice offsets
powershell -File run_eval_all.ps1              # runs all three models unattended
python humaneval_report.py                     # summarize results/*.json into a table
```

Sends real HumanEval problems (bundled as
[`HumanEval.jsonl.gz`](./HumanEval.jsonl.gz), 164 official problems) to a
running server, extracts the generated function, executes it against
HumanEval's actual unit tests in a sandboxed subprocess, `pass@1` at
temperature 0 for determinism. Results are written to `results/*.json` and
committed, so the numbers in this README are independently checkable
against raw output, not just hand-transcribed.

Three real harness bugs got caught and fixed while building this — all
worth knowing before trusting any pass@1 number you didn't verify by hand:

1. Extracted code needs the original prompt's import preamble prepended
   before execution (a model completing just the function body doesn't
   repeat `from typing import List`).
2. The temp file needs explicit UTF-8 encoding on Windows, or certain
   generated characters break execution with an unrelated-looking
   `SyntaxError`.
3. **Reasoning models need a much larger token budget — and even a large
   one isn't guaranteed enough.** gpt-oss-20b initially scored 17/20 (at
   n=20) with `max_tokens=512` — not because it got problems wrong, but
   because `finish_reason: "length"` hit while it was still mid-reasoning
   in `reasoning_content`, leaving `content` (the actual answer field)
   empty. Raising the budget to 4096 fixed that sample (20/20). But at
   n=80, the identical symptom reappeared on a different problem
   (`HumanEval/1`, 59 seconds of reasoning, still empty) — the fix reduced
   the failure rate, it didn't eliminate the underlying limitation. A
   model that appears to fail by returning nothing may just be a model
   that wasn't given room to finish thinking, and "enough room" isn't a
   constant.

All three looked like model failures at first and were actually harness
bugs — worth being suspicious of your own eval code before concluding the
model failed. (The corollary, from testing this same fix at a larger
scale: verifying a fix works isn't the same as verifying it always works.)

## Limitations

Honest caveats on everything above, not just the parts that look good:

- **Sample sizes are modest, not definitive.** HumanEval is 80 of 164
  official problems; perplexity is 50 of ~550 wikitext-2 chunks;
  `llama-bench` uses 3+ repetitions, not dozens. These are large enough to
  see real signal, not large enough to rule out a different result on a
  different sample.
- **HumanEval contamination is a real, unaddressed risk.** It's been
  public since 2021 and is one of the most-used eval sets in the field —
  models may have partially memorized it during training. If so, both the
  n=40 "identical" result and the n=80 "small real gap" result partly
  reflect memorization overlap, not purely reasoning ability. Nothing here
  rules that out for either finding. A contamination-resistant eval (fresh
  problems, e.g. LiveCodeBench-style) would be the actual fix — not done
  yet.
- **The evidence is still asymmetric, just less than it was.** At n=80,
  this repo directly measured a real correctness edge for Next (79/80 vs.
  77/80) — that part is no longer purely a proxy claim. But the *size* of
  that edge (2.6 points) versus the SWE-bench claim (70.6% vs. ~50%, over
  an order of magnitude larger) is not something independently verified
  here — the large claimed gap still rests on Next's own model card, not
  a benchmark run in this repo. Don't round "measurably better here" up to
  "as much better as the leaderboard claims."
- **None of these benchmarks measure the actual target workload.** The
  goal is agentic coding through opencode — multi-file, tool-calling,
  large repeated context. HumanEval is isolated single-function
  correctness; perplexity is generic text; `llama-bench` is synthetic
  prompt/generation throughput. All are useful proxies. None is a
  repo-scale agentic eval.
- **Single machine, single session, one driver/DLL combination.** Every
  number here is specific to this exact 3080 Ti + this exact CUDA/driver
  stack (see [Hardware & software](#hardware--software)). Results may not
  transfer even to a different 3080 Ti with different drivers, let alone
  different hardware.
- **Perplexity is invalid for gpt-oss specifically** — a known upstream
  issue, not a finding about the model's real quality. See
  [Known issues](#known-issues).

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

See [Tuning](#tuning--ncmoe-and-kv-cache) above — this happens when VRAM
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

<details>
<summary><b>gpt-oss-20b perplexity is ~20x worse than everything else here</b> — and it's not a real quality signal</summary>

`llama-perplexity` reported PPL 160 for gpt-oss-20b, versus 6.68-8.86 for
every other model tested — an implausibly large gap for a model that
otherwise generates coherent, correct text (it scored 39/40 on HumanEval
at n=40, identical to everything else). This is a known, documented issue,
not a bug in this repo's setup: see
[ggml-org/llama.cpp#15155](https://github.com/ggml-org/llama.cpp/issues/15155)
(~195 PPL reported independently) and a parallel
[huggingface/transformers issue](https://github.com/huggingface/transformers/issues/40990)
(~394 PPL). The likely cause: gpt-oss was trained so heavily around its
special "harmony" chat/reasoning format that raw next-token prediction on
plain prose (no chat formatting, which is what a standard perplexity test
does) is genuinely miscalibrated for this model family, even though it
behaves completely normally through its actual chat interface.

**Lesson:** perplexity is only a valid quality proxy for models evaluated
the way they were trained to be used. Before trusting a wildly outlying
number, check whether the model has documented quirks with the specific
evaluation method — don't just report the number.

</details>

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `--list-devices` shows nothing | CUDA backend DLL failed to load | Missing VC++ Redistributable DLLs — see Known issues |
| VRAM barely used, huge RAM usage, model "works" but slow | Silently running CPU-only despite `-ngl` | Same as above — confirm with `--list-devices` before assuming offload is active |
| Request hangs forever, `/health` still returns 200 | `-ncmoe` too low, VRAM headroom under ~500MB | Raise `-ncmoe`, or quantize the KV cache to free headroom instead |
| `llama-bench`/`llama-perplexity` missing, only `-impl.dll` files present | Prebuilt install only shipped the tools it uses directly | Pull matching `.exe` files from the release named in `UNSLOTH_PREBUILT_INFO.json` (or build from source, which includes everything) |
| opencode can't call tools correctly | Chat template mismatch | Make sure `--jinja` is enabled |
| pass@1 unexpectedly low on a reasoning model | Token budget too small — cut off mid-thought | Raise `max_tokens` significantly (see Benchmark methodology) |

## Files in this repo

| File | Purpose |
|---|---|
| [`start-qwen-coder.ps1`](./start-qwen-coder.ps1) | launch script, Qwen3-Coder-30B-A3B (default, fast) |
| [`start-qwen-coder-next.ps1`](./start-qwen-coder-next.ps1) | launch script, Qwen3-Coder-Next (slower, smarter) |
| [`start-gpt-oss-20b.ps1`](./start-gpt-oss-20b.ps1) | launch script, gpt-oss-20b (fastest tok/s, reasoning model) |
| [`opencode.example.jsonc`](./opencode.example.jsonc) | opencode provider config to copy in, all three models |
| [`bench.py`](./bench.py) | quick HTTP-based tok/s benchmark against a running server |
| [`humaneval_bench.py`](./humaneval_bench.py) | pass@1 code-correctness eval against a running server |
| [`run_eval_all.ps1`](./run_eval_all.ps1) | runs the HumanEval eval against all three models unattended |
| [`humaneval_report.py`](./humaneval_report.py) | summarizes `results/*.json` into a markdown table |
| [`HumanEval.jsonl.gz`](./HumanEval.jsonl.gz) | official HumanEval dataset (164 problems), bundled for reproducibility |
| [`results/`](./results) | raw per-problem HumanEval output, committed so the numbers above are checkable |
| [`LICENSE`](./LICENSE) | MIT |
