# Local Inference Backend (llama.cpp, cross-platform)

Polar Llama can run inference against [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server` -- an OpenAI-compatible local server that runs GGUF models on
**Linux, Windows, and macOS, with or without a GPU** -- using the same
`engine="server"` code path documented in
[`docs/local_mlx_backend.md`](local_mlx_backend.md).

**This document is the cross-platform counterpart of `local_mlx_backend.md`.**
`local_mlx_backend.md` is scoped to Apple Silicon because `mlx`/`mlx-lm` only
run there; this page is for everyone else (Linux, Windows, Intel/AMD, CUDA,
ROCm, or a Mac where you'd rather use llama.cpp than MLX). Both pages
describe the *same* `engine="server"` adapter
(`polar_llama/local/server_backend.py`) -- there is no llama.cpp-specific
code in Polar Llama. This document is just install/run instructions for a
different local server implementation, plus the caveats that are specific to
it.

> **Front and center: sampling parameters are NOT forwarded by the client.**
> `inference_local(..., engine="server")` accepts `max_tokens`, `temperature`,
> `top_p`, and `stop` for API parity with `engine="in_process"`, but the
> existing Rust `OpenAIClient` (`src/model_client/openai.rs::format_request_body`)
> deliberately does not put sampling parameters in the request body (some
> hosted models, e.g. OpenAI's o-series/gpt-5, reject them), so **none of
> these reach your local server's request**. Passing a non-default value logs
> a one-time Python warning but otherwise has no effect. **Set sampling via
> `llama-server`'s own CLI flags when you start it** -- `--temp`, `-n`
> (max tokens), `--top-p`, `--top-k`, etc. -- not via the Python call. See
> [Sampling parameters](#sampling-parameters-set-them-on-the-server-not-in-python)
> below for the flag mapping.

## 1. Install `llama-server`

`llama-server` ships inside every llama.cpp release as a prebuilt binary --
no compiler needed for the common case.

### Linux (prebuilt release binary)

```bash
# Pick a release tag from https://github.com/ggml-org/llama.cpp/releases
LLAMACPP_TAG=b10004
curl -fL -o llamacpp.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_TAG}/llama-${LLAMACPP_TAG}-bin-ubuntu-x64.tar.gz"
mkdir -p llamacpp && tar xzf llamacpp.tar.gz -C llamacpp --strip-components=1

# llama-server links libgomp (OpenMP) for its CPU backend
sudo apt-get update && sudo apt-get install -y libgomp1

LD_LIBRARY_PATH="$PWD/llamacpp" ./llamacpp/llama-server --version
```

CUDA builds are separate release assets (e.g.
`llama-<tag>-bin-ubuntu-cuda-<version>-x64.tar.gz` for the GPU-accelerated
binary, plus `cudart-llama-bin-*` for the CUDA runtime libraries if you don't
already have a CUDA toolkit installed) -- swap the asset name above for the
CUDA variant matching your driver, and add `-ngl <n>` (number of layers to
offload to GPU; `-ngl 999` offloads everything that fits) when you start the
server. ROCm and Vulkan builds are published the same way for AMD/other GPUs.

### Windows (prebuilt release binary)

```powershell
# Pick a release tag from https://github.com/ggml-org/llama.cpp/releases
$Tag = "b10004"
Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/llama-$Tag-bin-win-cpu-x64.zip" -OutFile llamacpp.zip
Expand-Archive llamacpp.zip -DestinationPath llamacpp
.\llamacpp\llama-server.exe --version
```

CUDA builds for Windows follow the same `llama-<tag>-bin-win-cuda-<version>-x64.zip`
naming as Linux, plus a matching `cudart-llama-bin-win-cuda-<version>-x64.zip`
runtime package if you need the CUDA runtime DLLs alongside it.

### macOS (Homebrew)

Apple Silicon Macs are already covered by MLX
(`docs/local_mlx_backend.md`), but llama.cpp also runs well there (including
GPU offload via Metal) if you'd rather standardize on one server across your
whole fleet:

```bash
brew install llama.cpp
llama-server --version
```

## 2. Download a GGUF model

Any GGUF-format model works. For a small model to get started:

```bash
curl -fL -o model.gguf \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
```

Browse [huggingface.co/models?library=gguf](https://huggingface.co/models?library=gguf)
for larger/better models (Llama, Qwen, Gemma, Mistral, ... all publish GGUF
builds).

## 3. Start the server

```bash
llama-server -m model.gguf --port 8080
```

`llama-server` exposes `POST http://localhost:8080/v1/chat/completions` --
the same OpenAI-compatible endpoint `mlx_lm.server` and `vllm-mlx` expose, so
everything in `local_mlx_backend.md`'s "Two engines" and "`engine="server"`"
sections applies unchanged. `/health` returns `200` once the model has
finished loading and the server is ready to accept requests -- useful for
scripted startup (see the CI job in `.github/workflows/CI.yml`'s
`llamacpp_server_test` for a working poll loop).

## 4. Point Polar Llama at it

Identical to the MLX server path -- this is the *same* adapter, just a
different base URL:

```python
import polars as pl
import polar_llama  # registers the .llama namespace

df = pl.DataFrame({"prompt": ["Summarize photosynthesis in one sentence."]})

df = df.with_columns(
    answer=pl.col("prompt").llama.inference_local(
        "model.gguf",           # any non-empty string -- llama-server ignores
                                 # the request's `model` field, it always
                                 # answers with whatever GGUF it was started with
        engine="server",
        base_url="http://localhost:8080",
    )
)
```

`model=` is accepted for API parity with every other `inference_*` call but
is not used to select anything server-side: a single `llama-server` process
serves exactly one loaded GGUF, chosen at startup by `-m`. Passing the wrong
model name here does not raise or route incorrectly -- it just gets echoed
back in the (unused) response `model` field.

## Sampling parameters: set them on the server, not in Python

Because `max_tokens`/`temperature`/`top_p`/`stop` are not forwarded (see the
caveat at the top of this document), configure sampling with `llama-server`
CLI flags at startup instead:

| Python kwarg (not forwarded) | `llama-server` flag | Notes |
|---|---|---|
| `max_tokens` | `-n <N>` / `--n-predict <N>` | Max tokens to generate per request. `-1` = until EOS or context limit. |
| `temperature` | `--temp <T>` | `--temp 0` for deterministic/greedy output. |
| `top_p` | `--top-p <P>` | |
| `stop` | *(no server-wide flag)* | `llama-server` supports `stop` in the **request body**, but the Rust client does not send it (same limitation as the others). If you need per-row stop sequences today, call the lower-level `inference_async(provider=Provider.OPENAI, model=..., ...)` yourself after `polar_llama.local.server_backend.set_local_endpoint(base_url)` and extend the request -- or open a follow-up issue; this is exactly the kind of gap Tier 2 (see below) would close by forwarding these fields in the Rust request body. |

Example:

```bash
llama-server -m model.gguf --port 8080 --temp 0 -n 512 --top-p 0.95 -c 4096
```

`-c <N>` sets the context window (`--parallel <N>` splits it into `N`
concurrent request slots -- raise this if you're sending many rows
concurrently through Polar Llama's own `POLAR_LLAMA_MAX_CONCURRENCY`, or
requests will queue server-side instead of running in parallel).

## Feature matrix

Accurate as of this document, for `pl.col(...).llama.inference_local(...)`
and the lower-level functions it wraps. "Server" columns describe what the
*existing, unchanged* `engine="server"` adapter gets for free from each
server's OpenAI-compatibility; "in-process" describes the Apple-Silicon-only
`engine="in_process"` MLX engine (`docs/local_mlx_backend.md`).

| Feature | `server` + `llama-server` | `server` + `mlx_lm.server` | `in_process` (mlx-lm) |
|---|---|---|---|
| Platform | Linux, Windows, macOS; CPU or CUDA/ROCm/Vulkan/Metal GPU | Apple Silicon only | Apple Silicon only |
| Structured outputs (`response_model`) | Not wired through `inference_local()` (no `response_model` param on it); reachable via `inference_async(provider=Provider.OPENAI, ...)` directly after `set_local_endpoint()` -- `llama-server` supports JSON-schema-constrained grammars server-side | Same as `llama-server` column -- same adapter, same gap; `mlx_lm.server`'s structured-output support is more limited | Not implemented -- the `map_batches` UDF returns plain strings only |
| Sampling-param forwarding (`max_tokens`/`temperature`/`top_p`/`stop`) | **Not forwarded** -- set via server CLI flags (see table above) | **Not forwarded** -- same limitation, same server CLI-flag workaround | Forwarded directly -- these are real Python kwargs consumed by `mlx-lm`'s sampler |
| Batching / concurrency | Server-side request slots (`--parallel`) + Rust `futures::buffered` fan-out (`POLAR_LLAMA_MAX_CONCURRENCY`) | Same mechanism -- Rust fan-out, `mlx_lm.server`'s own scheduler | `mlx-lm`'s in-process continuous-batching `BatchGenerator` |
| Collapsed prefill (`POLAR_LLAMA_LOCAL_COLLAPSE`) | N/A -- that's an in-process-engine-only optimization; `llama-server` has its own independent prompt-caching (`--slot-save-path`, automatic prefix reuse across requests to the same slot) | N/A, same reasoning | Supported (`docs/collapsed_prefill.md`) |
| Batched quantized KV cache (`POLAR_LLAMA_LOCAL_KV_BITS`) | N/A -- `llama-server` has its own `--cache-type-k`/`--cache-type-v` flags for quantized KV, set at server startup, independent of Polar Llama | N/A, same reasoning | Supported (`docs/batch_quantized_kv.md`) |
| `usage=True` | **Real usage block** -- `llama-server` returns `usage.prompt_tokens`/`completion_tokens`/`prompt_tokens_details.cached_tokens`, decoded by the existing `src/model_client/openai.rs::parse_usage` exactly like a hosted OpenAI response; `cost_usd` resolves to `null` for a local GGUF name unless you register one via `price_table=` | Same mechanism; whether `mlx_lm.server` populates a real `usage` block depends on its version -- verify against yours | Reports tokens (whitespace-based) + latency; `cost_usd` is always `0.0` (no network cost) |
| Streaming (`on_token`) | Reachable -- `inference_stream(provider=Provider.OPENAI, ...)` after `set_local_endpoint()` streams real SSE deltas from `llama-server`; not exposed through the `inference_local()` wrapper itself | Same reachability, same caveat | Not implemented -- `engine="in_process"` has no token-level callback |

Rows marked "N/A" for the in-process engine mean the feature doesn't apply
there conceptually (e.g. collapsed prefill is meaningless without an
in-process KV cache to collapse); rows marked "N/A" for the server engines
mean the *equivalent* capability exists but lives entirely on the server side
via its own flags, not through anything Polar Llama's Python/Rust code
touches.

## What this document does NOT cover (Tier 2, deferred)

This is the **Tier 1** scope: `engine="server"` pointed at a real
`llama-server` process you run yourself, which required zero library-code
changes because the existing OpenAI-compatible adapter already handles it.
A Tier 2 **in-process** llama.cpp binding (analogous to `engine="in_process"`
for MLX, e.g. via `llama-cpp-python` or an ONNX Runtime path, giving
non-Apple machines a no-server-process option with sampling parameters that
*are* forwarded) is intentionally **out of scope here** and tracked as a
separate follow-up issue.

## References

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) -- releases,
  `llama-server` docs, and supported GGUF quantizations.
- [`docs/local_mlx_backend.md`](local_mlx_backend.md) -- the Apple-Silicon
  MLX-specific engine (`engine="in_process"`) and the shared `engine="server"`
  adapter design this document reuses.
- `.github/workflows/CI.yml`, job `llamacpp_server_test` -- a working,
  pinned, CI-verified example of downloading `llama-server`, downloading a
  GGUF, starting the server, and polling `/health`.
- `tests/test_llamacpp_server_live.py` -- the gated test this CI job runs,
  asserting response shape (row count, non-null/non-empty strings, no
  in-band error envelopes, a `system=` case, and a real `usage=True` block)
  against a live `llama-server`.
