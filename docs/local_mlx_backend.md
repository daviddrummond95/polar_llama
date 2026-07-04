# Local Inference Backend (MLX)

Polar Llama can run inference against a **local model on your own machine**
instead of a hosted API, using the same `.llama` expression API you already
use for OpenAI/Anthropic/Gemini/Groq/Bedrock:

```python
pl.col("prompt").llama.inference_local(
    "mlx-community/gemma-4-e2b-it-4bit",
    system=None,
    engine="server",
    base_url=None,
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    stop=None,
)
```

The result column is `String` completions, **in the original row order**, the
same contract as every other `inference_*` function in this library.

`system=...` is a separate, immutable argument (not concatenated into the
prompt string) because it is the prefix that a local server or in-process
KV cache uses to decide what can be reused across rows. Keep it identical
across a batch if you want caching to help.

This document is scoped to **Apple Silicon** (M-series) Macs, which is the
only platform MLX runs on. On any other machine, use a hosted provider or
point `engine="server"` at a non-Mac local server (e.g. vLLM on Linux/CUDA)
that speaks the OpenAI chat-completions API.

## Two engines, two risk profiles

`inference_local` has two backends, selected with `engine=`:

| | `engine="server"` (default) | `engine="in_process"` |
|---|---|---|
| How it runs | Polar Llama's existing async Rust fan-out talks HTTP to a local OpenAI-compatible server you started yourself | A Python `map_batches` UDF drives `mlx-lm` directly in the same process |
| Requires | Any local server that implements `/v1/chat/completions` (`mlx_lm.server`, `vllm-mlx`, `llama.cpp` server, LM Studio, etc.) | `pip install polar-llama[local]` (installs `mlx` + `mlx-lm`) and an Apple Silicon Mac |
| Concurrency | Rust `futures::buffered`, same knob as hosted providers (`POLAR_LLAMA_MAX_CONCURRENCY`, default 64) | `mlx-lm`'s own continuous-batching scheduler (`BatchGenerator`) |
| Maturity | Low risk — no new code path, just a different base URL | Newer, more moving parts; treat as **experimental** |
| When to reach for it | You already run (or are fine running) a local server process, want the simplest integration, or want to batch across multiple local *and* hosted providers uniformly | You want the lowest latency / highest throughput for a single Mac with no server process to manage, and are comfortable with MLX's current rough edges (see [Known limitations](#known-limitations-in-process-engine)) |

If you're not sure which to pick: start with `engine="server"`. It's the
same code path used for every hosted provider, just retargeted at
`localhost`, so it inherits all of that path's error handling and
concurrency behavior for free.

## `engine="server"`: point Polar Llama at a local server

### 1. Start a local OpenAI-compatible server

**`mlx_lm.server`** (ships with `mlx-lm`, Apple Silicon only):

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/gemma-4-e2b-it-4bit --port 8080
```

**`vllm-mlx`** ([waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx),
Apache-2.0) — a vLLM-style continuous-batching server targeting MLX:

```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/gemma-4-e2b-it-4bit --port 8080
```

Either one exposes `POST http://localhost:8080/v1/chat/completions`.

### 2. Point Polar Llama at it

The Rust `OpenAIClient` already reads `OPENAI_BASE_URL` at request time
(`src/model_client/openai.rs`), so no Rust changes are needed to route
through a local server — `inference_local(..., engine="server")` sets this
for you for the duration of the call. Two equivalent ways to configure it:

```python
import polars as pl
from polar_llama import Provider

df = pl.DataFrame({"prompt": ["Summarize photosynthesis in one sentence."]})

# Option A: pass base_url explicitly
df = df.with_columns(
    answer=pl.col("prompt").llama.inference_local(
        "mlx-community/gemma-4-e2b-it-4bit",
        engine="server",
        base_url="http://localhost:8080",
    )
)

# Option B: set OPENAI_BASE_URL yourself (e.g. in your shell/CI env) and
# omit base_url — inference_local will pick it up the same way any other
# OpenAI-routed call does.
```

```bash
export OPENAI_BASE_URL="http://localhost:8080"
```

Most local servers accept any non-empty string as the API key; you do not
need a real OpenAI key. If your server enforces one, set `OPENAI_API_KEY` as
usual.

Everything else — concurrency, retries, error handling, structured-output
validation — is whatever the existing OpenAI-provider code path already does.
There is nothing local-specific about it beyond the base URL.

## `engine="in_process"`: in-process MLX via `mlx-lm`

### Install

```bash
pip install "polar-llama[local]"
```

This pulls in `mlx` and `mlx-lm`. It is an **optional extra** — the base
`polar-llama` install has no MLX dependency, and importing `polar_llama` on
Linux/Windows/Intel Mac continues to work with zero MLX-related import
errors. The import is only attempted, and only guarded/raises a helpful
`ImportError`, when you actually call `engine="in_process"`.

### Usage

```python
import polars as pl

df = pl.DataFrame({"prompt": [
    "Explain the sliding-window trick in one sentence.",
    "What is a KV cache?",
]})

df = df.with_columns(
    answer=pl.col("prompt").llama.inference_local(
        "mlx-community/gemma-4-e2b-it-4bit",
        engine="in_process",
        max_tokens=256,
    )
)
```

Under the hood this is a pure-Python `map_batches` UDF (mirroring the
`register_plugin(...).map_batches(...)` pattern already used by the hosted
`inference()` path) that calls into `mlx-lm`'s own **continuous batching**
(`BatchGenerator`) and **batched, left-padded KV cache**
(`BatchKVCache` / `BatchRotatingKVCache`) machinery. Polar Llama does not
reimplement batching or KV management — `mlx-lm` 0.31+ already ships both,
and this backend is a thin wrapper over them.

The loaded model is held in a process-global singleton behind a lock, keyed
by `(model, engine)`, so repeated calls (and repeated Polars morsels under
the streaming engine) reuse the same weights and cache instead of reloading
per batch.

### Known limitations (in-process engine)

This backend is newer and more GPU-dependent than `engine="server"`. Be
aware of the following before relying on it in production:

- **Apple Silicon + GPU only.** `mlx` has a CPU fallback, but it is not a
  supported target for this backend — expect it to be slow and it is not
  exercised in CI. There is no CUDA/CPU equivalent; use `engine="server"`
  with a non-Mac server (e.g. vLLM) if you need that.
- **Speculative decoding is not available.** `mlx-lm`'s speculative decoding
  is single-sequence only and is not integrated into `BatchGenerator`; the
  in-process engine does not use it.
- **No quantized KV cache for rotating/sliding-window caches.**
  `RotatingKVCache.to_quantized` currently raises `NotImplementedError`
  upstream, so hybrid architectures with sliding-window layers (e.g. Gemma 4)
  run their KV in full precision even when weights are 4-bit quantized.
- **Gemma 3n batched generation is broken in mlx-lm <= 0.31.3**
  ([mlx-lm #1384](https://github.com/ml-explore/mlx-lm/issues/1384)): the
  batched cache path (used by both `batch_generate` and `mlx_lm.server`)
  crashes or silently emits garbage on Gemma 3n's KV-shared hybrid
  architecture. Polar Llama ships a guarded, idempotent runtime workaround —
  call `polar_llama.local._mlx_patches.apply_gemma3n_batched_shared_kv_patch()`
  before loading a Gemma 3n model (it is a no-op once upstream is fixed).
  Root cause, upstream patch, and the parity validation are documented in
  [`mlx_lm_1384_fix.md`](mlx_lm_1384_fix.md).
- **Hybrid-model prefix caching has a known rough edge.** Trimming a shared
  prefix cache on hybrid (sliding + full attention) models has an open
  upstream issue (see [References](#references)); if you see stale or
  incorrect completions when reusing a prefix cache across very different
  follow-up prompts on a hybrid model, disable prefix-cache reuse for that
  workload as a workaround.
- **CI does not exercise real MLX.** All in-process-engine *logic* (batching
  semantics, error payload shape, ordering) is tested on Linux against a
  fake, deterministic engine with no GPU and no `mlx` import. The real
  `mlx-lm` code path is exercised manually on Apple Silicon, not in CI.

## Memory and batch-size expectations

Figures below are **measured 4-bit-quantized weight footprints**, not
theoretical/effective-parameter estimates:

| Model | 4-bit weights | Notes |
|---|---|---|
| Gemma 4 E2B | ~2.51 GB | |
| Gemma 4 E4B | ~5.82 GB | |
| Ornith-1.0-9B | ~5.2 GB | MIT-licensed, post-trained on Gemma 4 / Qwen 3.5 |

KV cache on top of weights is what actually limits batch size, and it's
where Gemma 4's architecture helps a lot: Gemma 4 is a **hybrid** model where
4 of every 5 layers use **sliding-window attention capped at 512 tokens**,
and a subset of later layers **share KV** with earlier ones instead of
allocating their own. In practice this means Gemma-4 KV grows far more
slowly with sequence length and batch size than a plain full-attention model
of the same size — most of the "long context" KV cost that would otherwise
dominate memory at high batch sizes never materializes.

Honest batch-size expectations on **this class of hardware (Apple M4 Pro,
24 GB unified memory)**:

- **E2B (4-bit, ~2.5 GB weights):** comfortably supports batch sizes in the
  tens of concurrent sequences at moderate context lengths (low hundreds of
  tokens) before KV growth becomes the constraint, thanks to the 512-token
  sliding window on most layers.
- **E4B (4-bit, ~5.8 GB weights):** noticeably less headroom for batching
  and long contexts on 24 GB than E2B — expect single-digit-to-low-teens
  concurrent sequences to be a safer starting point, tuning down as context
  length grows.
- These are **starting points for experimentation, not guarantees.** Actual
  ceilings depend on prompt length, `max_tokens`, other processes competing
  for unified memory, and macOS's own memory pressure behavior. Always
  benchmark your own workload rather than assuming these numbers transfer
  directly — the throughput multipliers cited below were measured on a
  machine with roughly double this one's memory bandwidth (see below).
- `BatchGenerator` sets its own Metal "wired" memory limit internally; you
  generally do not need to tune MLX memory environment variables yourself
  before trying a given batch size.

## References

- Gim, I., Chen, G., Lee, S.-S., Sarda, N., Khandelwal, A., & Zhong, L.
  (2023). *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*.
  [arXiv:2311.04934](https://arxiv.org/abs/2311.04934).
- [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) (Apache-2.0)
  — vLLM-style continuous-batching server for MLX. Its reported 4.3x/5.8x
  throughput gains over `llama.cpp` were measured on an **M4 Max** (roughly
  2x this machine's memory bandwidth); treat those numbers as ceilings for
  Apple Silicon in general, not targets for an M4 Pro specifically. The
  broader "MLX leads llama.cpp by 20-87%" comparison is from the same
  project's benchmarking, arXiv:2601.19139.
- [lablup/mlxcel](https://github.com/lablup/mlxcel) — MLX serving/tooling
  project referenced for MLX ecosystem context.
- [mlx-lm issue #980](https://github.com/ml-explore/mlx-lm/issues/980) —
  hybrid-model prefix-cache trim path is broken; closed as "outdated" after
  a server-side mitigation, but no trim fix has been merged upstream. This
  is the basis for the hybrid prefix-caching caveat above.
- [Gemma 4](https://ai.google.dev/gemma) — hybrid sliding/full attention
  (5:1 ratio), PLE, released in E2B/E4B sizes.
- [Ornith-1.0-9B](https://huggingface.co/mlx-community) — MIT-licensed model
  post-trained on Gemma 4 / Qwen 3.5.
