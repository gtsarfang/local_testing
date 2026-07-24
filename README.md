# Running Qwen3-Coder-30B-A3B Locally on a 12GB GPU

A walkthrough of setting up a local coding LLM that actually fits — 30B total
parameters, 12GB of VRAM, and a real agentic coding loop through
[opencode](https://opencode.ai), no cloud API involved.

## Hardware

- CPU: Intel i9, 12th gen
- GPU: NVIDIA RTX 3080 Ti (12GB VRAM)
- RAM: 64GB DDR4
- OS: Windows 10

## Why this model

[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
is a **Mixture-of-Experts** model: 30B total parameters, but only ~3.3B are
active per token (8 of 128 experts fire per forward pass). That distinction is
what makes it viable on a 12GB card — a dense 30B model would need every
parameter touched on every token, but an MoE model only needs to move the
active experts through compute. That means you can park most of the model in
system RAM and still get good throughput, as long as your inference engine
can split GPU/CPU placement at the *expert* level.

At Q4 quantization it scores ~50% on SWE-bench Verified, purpose-built for
agentic/tool-calling coding workflows — which matters more here than raw
size, since the target is an agent loop (opencode), not a chatbot.

Quant used: **`UD-Q4_K_XL`** (17.7GB), Unsloth's dynamic quant — smaller and
higher quality than the plain `Q4_K_M` because it allocates precision
non-uniformly across layers instead of a flat 4 bits everywhere.

## The tool: llama.cpp, not Ollama

Both were available, but llama.cpp was the right call here specifically
because of `--n-cpu-moe` (`-ncmoe`): a flag that pins the MoE expert weights
of the first N layers to CPU RAM while keeping attention/shared layers and
the KV cache on GPU. This is a tunable knob for exactly this hardware
constraint. Ollama can load the same GGUF, but its GPU/CPU split is a
heuristic, not something you dial in by hand.

## Step 1 — Get the model

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
  --local-dir C:\path\to\models
```

Don't grab the smallest quant a downloader UI suggests by default — tools
that recommend based on "does this fit entirely in VRAM" don't know your
engine can split MoE experts across GPU+RAM. `UD-Q4_K_XL` is the real sweet
spot for a 12GB + 64GB combo, not `IQ1_*`.

## Step 2 — Build/get a CUDA-enabled llama.cpp

If you have Unsloth's tooling installed, it may have already built one for
you (check for `llama-server.exe` alongside a `ggml-cuda.dll` in its
`llama.cpp/build/bin/Release` folder). Otherwise, build from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with
`-DGGML_CUDA=ON`.

## Step 3 — The gotcha: CUDA silently falls back to CPU

This one cost the most time. The server started fine, loaded the model, and
responded to requests — but VRAM usage never moved, and the whole 17.7GB sat
in system RAM instead. No error was printed.

Diagnosis:

```bash
llama-server.exe --list-devices
# Available devices:
#   (nothing listed)
```

No CUDA device was detected at all, despite `ggml-cuda.dll` being present.
Testing `LoadLibrary` on that DLL directly (via a small PowerShell P/Invoke
snippet) returned **Win32 error 126** — `ERROR_MOD_NOT_FOUND`, meaning the
DLL itself loaded but one of *its* dependencies couldn't be found. Walking
the import table (`pefile` in Python) showed the real culprits:
`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` — the Visual C++
Redistributable. It wasn't installed on this machine.

**Fix, without a system-wide installer:** Ollama ships its own copies of
these runtime DLLs (plus matching `cublas`/`cudart`) alongside its bundled
llama.cpp backend, since it's a self-contained deployment. Copying them from
Ollama's install directory into the llama.cpp build folder fixed it
immediately — no admin install, no reboot:

```
<ollama install dir>\lib\ollama\cuda_v13\cublas64_13.dll
<ollama install dir>\lib\ollama\cuda_v13\cublasLt64_13.dll
<ollama install dir>\lib\ollama\cuda_v13\msvcp140.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140_1.dll
<ollama install dir>\lib\ollama\cuda_v13\vcruntime140_threads.dll
```
→ copied into `llama.cpp\build\bin\Release\` (same folder as `llama-server.exe`).

After that, `--list-devices` showed the 3080 Ti correctly.

If you hit the same silent-CPU-fallback symptom and don't have Ollama
installed, the general fix is: install the
[Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Step 4 — Tune `--n-cpu-moe` for your VRAM budget

This is empirical: pick a value, load the model, check headroom under real
generation (not just at idle), adjust. [`bench.py`](./bench.py) is a small
script for this — it hits the running server's `/v1/chat/completions`
endpoint with a long prompt-processing-heavy request and a short
generation-heavy request, and reports back the server's own reported
`prompt tok/s` and `gen tok/s`.

| `-ncmoe` | VRAM free (idle) | prompt tok/s | gen tok/s |
|---|---|---|---|
| 20 | 358MB | — | **hung, never returned** |
| 24 | 216MB | — | **hung, never returned** |
| **30** | **643MB** | **1034** | **27.1** |
| 34 | 1805MB | 999 | 21.6 |

The important finding here isn't the speed numbers — it's that **`-ncmoe`
values below ~30 didn't just run slower, they silently hung** on real
generation requests, sometimes after correctly answering a health check.
This is a Windows/WDDM-specific failure mode: when a CUDA allocation doesn't
fit in the remaining VRAM, WDDM can quietly page to system memory instead of
returning a clean out-of-memory error, so the request just never completes
instead of failing fast. `nvidia-smi` will still show the process using GPU
memory the whole time — nothing crashes, it just never comes back. (On
Linux this class of config would probably fail loudly with a CUDA OOM
instead, which is arguably easier to debug than a silent hang.)

Going from `-ncmoe 30` to `-ncmoe 34` bought ~1.2GB more headroom (643MB →
1805MB free) at the cost of ~20% generation speed (27.1 → 21.6 tok/s) — a
real, measurable tradeoff, not just "more offload = worse" in a vague sense.

Also verified stable with a ~4800-token prompt at `-ncmoe 30`: VRAM barely
moved, because the KV cache is pre-allocated at server startup for the full
context size — it doesn't grow with usage the way you'd expect on a naive
setup.

**Takeaway:** don't just check idle VRAM after model load and call it
tuned — test with an actual generation request, and give yourself more
headroom than looks strictly necessary. "It loaded fine" is not the same as
"it will actually respond." `-ncmoe 30` shipped as the first working config
— but see below, it wasn't the final one.

## Step 4b — Quantized KV cache: a free win that changed the answer

`-ncmoe` isn't the only lever. The KV cache itself defaults to `f16`, and
`-ctk q8_0 -ctv q8_0` quantizes it to 8-bit — roughly halving its VRAM
footprint. Tried this at `-ncmoe 30` expecting a speed/quality tradeoff and
got neither: it was **faster and freed more VRAM**, not a tradeoff at all.

| config | VRAM free (idle) | prompt tok/s | gen tok/s |
|---|---|---|---|
| `-ncmoe 30`, KV f16 (original) | 643MB | 1034 | 27.1 |
| `-ncmoe 30`, KV q8_0 | **1518MB** | 1190 | 29.1 |

With that much headroom back, `-ncmoe` could drop again — this time landing
somewhere genuinely fast *and* safely clear of the hang zone found earlier
(anything under ~500MB free is the danger zone; this isn't):

| config | VRAM free (idle) | prompt tok/s | gen tok/s |
|---|---|---|---|
| `-ncmoe 24`, KV q8_0 | 210MB | 1320 | 33.4 |
| **`-ncmoe 27`, KV q8_0** | **1161MB** | **1291** | **31.5** |

`-ncmoe 24` technically completed a benchmark run, but 210MB free is the
same range that silently hung earlier in this doc — one successful run
isn't proof of stability, it's a coin flip that happened to land right.
`-ncmoe 27` is the one that's actually trustworthy: re-verified with the
same ~4800-token stress prompt, VRAM held steady, response in under 4
seconds. **This is the config now in use** — faster than every earlier
attempt, including the original `-ncmoe 30`/f16 baseline, with nearly
double its headroom.

**Takeaway:** don't just tune one lever. KV cache quantization and MoE
offload interact — freeing memory on one axis buys room to push the other
further, and the combined optimum wasn't reachable by tuning `-ncmoe` alone.

## Step 5 — Launch command

[`start-qwen-coder.ps1`](./start-qwen-coder.ps1) has the tuned launch
command — edit the two paths at the top for your machine, then:

```bash
powershell -File start-qwen-coder.ps1
```

Equivalent to:

```powershell
llama-server.exe `
  -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf `
  --host 127.0.0.1 --port 8080 `
  -ngl 999 -ncmoe 27 `
  -c 32768 -fa on `
  -ctk q8_0 -ctv q8_0 `
  --no-mmap `
  --jinja `
  -np 1
```

`--jinja` matters — it uses the model's embedded chat template, which is
what makes tool-calling format correctly for an agent like opencode.
`--no-mmap` is a minor perf win once you're overriding tensor placement
(llama.cpp prints a warning suggesting it).

## Step 6 — Wire it into opencode

[`opencode.example.jsonc`](./opencode.example.jsonc) — copy this into your
opencode config directory (`~/.config/opencode/opencode.jsonc` on this
setup):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "qwen3-coder-30b-a3b": {
          "name": "Qwen3-Coder-30B-A3B (local)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

```bash
opencode run -m llama.cpp/qwen3-coder-30b-a3b "your prompt here"
```

## Step 7 — Verify tool-calling actually works

A plain chat response isn't proof the agent loop works — opencode needs the
model to correctly emit tool calls (bash, file read/write, etc.) in the
format its chat template expects. Quick check:

```bash
opencode run -m llama.cpp/qwen3-coder-30b-a3b \
  "List the files in the current directory using your tools, then tell me how many .gguf files are there."
```

If it runs `ls` itself and answers from the real output (not a guess), tool
calling is wired correctly.

As a fuller smoke test, try a multi-file build: ask it to write a small CLI
tool with a couple of source files and a test suite, then have it run the
tests itself and fix anything that fails. That exercises the full agent
loop — planning, multi-file writes, running commands, reading the output,
self-correcting — not just a single tool call.

## Reproducing the benchmark

With the server running, [`bench.py`](./bench.py) reports prompt-processing
and generation tok/s using the server's own timing data — quick, but noisy
(HTTP overhead, no warmup, small sample):

```bash
python bench.py
# {"prompt_tok_per_sec": 1291.0, "gen_tok_per_sec": 31.5}
```

Useful for comparing configs on your own hardware — just restart the server
with different flags between runs, and give it a few seconds after process
exit for VRAM to actually release before launching the next one.

## Getting real numbers: `llama-bench` and `llama-perplexity`

The Unsloth-installed build only shipped `llama-server.exe` — the
`llama-bench` and `llama-perplexity` tools existed as internal `-impl.dll`
files but had no matching `.exe` wrapper. Rather than setting up a whole
build toolchain (no `cmake` was even installed on this machine), the fix was
simpler: `UNSLOTH_PREBUILT_INFO.json` in the llama.cpp source tree records
exactly which upstream release this build came from
(`unslothai/llama.cpp`, tag `b10069-mix-fb3d4ca`). Downloading that same
release's zip from GitHub and pulling `llama-bench.exe` /
`llama-perplexity.exe` out of it gives tools that are guaranteed
DLL/ABI-compatible with the CUDA backend already installed — no build
required, no version mismatch risk.

`llama-bench` runs the standard `pp512`/`tg128` test (512-token prompt
processing, 128-token generation, 3+ repetitions with warmup) — the same
methodology used across the llama.cpp community, so these numbers are
directly comparable to numbers other people post for other hardware:

```bash
llama-bench.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -ngl 999 -ncmoe 27 -fa on -ctk q8_0 -ctv q8_0
```

| config | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| `-ncmoe 30`, KV f16 | 366.18 ± 14.60 | 28.09 ± 0.99 |
| `-ncmoe 34`, KV f16 | 340.17 ± 25.09 | 26.18 ± 1.54 |
| `-ncmoe 30`, KV q8_0 | 409.67 ± 21.45 | 29.18 ± 1.30 |
| **`-ncmoe 27`, KV q8_0** | **456.77 ± 37.24** | **31.87 ± 1.73** |

## Quality check: perplexity

Speed numbers don't say anything about whether the model is still *good* at
this quantization level. Perplexity does — lower is better, and it's the
standard proxy metric for quantization quality loss. Tested against
[wikitext-2-raw](https://huggingface.co/datasets/ggml-org/ci) (the
standard corpus for this — llama.cpp's own `scripts/get-wikitext-2.sh` uses
it), 50 chunks of 512 tokens each (a subset, not the full ~550-chunk test
set — full run would take ~12+ minutes, this is a faster representative
sample):

```bash
llama-perplexity.exe -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -f wiki.test.raw -ngl 999 -ncmoe 30 -fa on -c 512 --chunks 50
# Final estimate: PPL = 8.8606 +/- 0.23686
```

`-ncmoe` only changes *where* weights are computed (GPU vs CPU), not the
weights themselves, so this number is unaffected by which `-ncmoe` value is
used — it's purely a function of the quant (`UD-Q4_K_XL`).

There's no other quant of this model benchmarked here to compare against,
so treat this as a baseline number for the `UD-Q4_K_XL` quant specifically,
useful if you want to test whether a different quant or config changes
quality meaningfully — not a verdict on its own.

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `--list-devices` shows nothing | CUDA backend DLL failed to load | Check for missing VC++ Redistributable DLLs (see Step 3) |
| VRAM barely used, huge RAM usage, model "works" but slow | Silently running CPU-only despite `-ngl` | Same as above — confirm with `--list-devices` before assuming offload is active |
| Request hangs forever, `/health` still returns 200 | `-ncmoe` too low, VRAM headroom under ~500MB — WDDM pages silently instead of erroring | Raise `-ncmoe` a few steps, or quantize the KV cache (`-ctk q8_0 -ctv q8_0`) to free headroom without raising `-ncmoe` at all — then re-test with `bench.py` against a real generation request, not just idle VRAM after load |
| `llama-bench`/`llama-perplexity` missing, only `-impl.dll` files present | Unsloth's installer only ships the tools it uses directly | Pull the matching `.exe` files from the same release zip named in `UNSLOTH_PREBUILT_INFO.json` — guaranteed ABI-compatible, no build needed |
| opencode can't call tools correctly | Chat template mismatch | Make sure `--jinja` is enabled so the model's own template is used |
