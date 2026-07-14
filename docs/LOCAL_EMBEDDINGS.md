# Local (Offline) Embeddings

`embedding_local` is the fully-offline counterpart of `embedding_async`
(issue #83): it runs an embedding model **in this process**, via
[`mlx_embeddings`](https://pypi.org/project/mlx-embeddings/), instead of
calling a hosted provider API (OpenAI/Gemini/Bedrock). No API key, no
network call, no per-token cost.

```python
import polars as pl
from polar_llama import embedding_local

df = pl.DataFrame({
    "text": ["Hello world", "Machine learning is fun"]
})

df = df.with_columns(
    embeddings=embedding_local(pl.col("text"))
)
```

**Using the `.llama` namespace:**
```python
df = df.with_columns(
    embeddings=pl.col("text").llama.embedding_local()
)
```

The output dtype is `List[Float64]` -- **identical to `embedding_async`** --
so a column produced by either function is a drop-in input to
`cosine_similarity`, `dot_product`, `euclidean_distance`, `knn_hnsw`,
`HnswIndex`, and `cluster_embeddings`. You can mix and match: embed a large
offline corpus locally, then compare it against a query embedded through a
hosted provider (or vice versa) -- the only thing that matters downstream is
the vector, not which function produced it.

## Signature

```python
embedding_local(
    expr,
    *,
    model: str = "mlx-community/bge-small-en-v1.5-bf16",
    engine: str = "in_process",
    batch_size: int = 64,
    normalize: bool = True,
) -> pl.Expr
```

| Parameter | Default | Meaning |
|---|---|---|
| `model` | `mlx-community/bge-small-en-v1.5-bf16` | An `mlx-community` (or other `mlx_embeddings`-compatible) model repo id. |
| `engine` | `"in_process"` | `"in_process"` (alias `"mlx"`) uses the real `mlx_embeddings` model. `"fake"` forces the dependency-free deterministic engine (see "CI / testing" below). |
| `batch_size` | `64` | Number of texts handed to the model per call; larger columns are processed in row-aligned chunks of this size, bounding peak memory. |
| `normalize` | `True` | L2-normalize every output vector. |

Null handling matches `embedding_async`'s Rust plugin exactly: a `None`
input row produces a `None` output row; an **empty string `""` is embedded
like any other text**, not nulled (verified against
`src/expressions.rs::embedding_async`, which only skips `None` values). A
per-row engine failure (e.g. a pathological input) also degrades to a
`None` output row rather than raising -- one bad row never aborts the rest
of the batch or column.

## Requirements and first-use download

`engine="in_process"` (the default) requires the optional `[local]` extra,
and only runs on Apple Silicon (the `mlx` family is Metal-only):

```bash
pip install "polar-llama[local]"
```

This installs `mlx`, `mlx-lm` (used by `inference_local`/`Predict`'s
on-device backend), and `mlx-embeddings` (used here). **Merely importing
`polar_llama` or `polar_llama.local` never imports any of these** -- they
are only imported the first time you actually call `embedding_local(...,
engine="in_process")` (or `"mlx"`). `import polar_llama` and
`import polar_llama.local` succeed on any machine, including Linux CI
runners with none of `mlx`/`mlx-lm`/`mlx-embeddings` installed.

The first call to `embedding_local` with a given `model` downloads that
model's weights from the Hugging Face Hub -- for the default
`mlx-community/bge-small-en-v1.5-bf16` (384 dimensions), that's about
**65 MB**. The download is cached under `~/.cache/huggingface` (the
standard HF cache directory); every subsequent call, including in a new
process, reuses the cached weights with **zero network traffic**. If you
want to assert that, set `HF_HUB_OFFLINE=1` after the first successful call
in your environment -- any cache miss will then raise instead of silently
hitting the network.

A process-global singleton registry (keyed by `(model, engine)`, mirroring
the completion-side `inference_local` engine) ensures a given model is
loaded into memory **at most once per process**, no matter how many times
`embedding_local(...)` appears in a pipeline or how many `map_batches`
morsels the Polars streaming engine dispatches.

## `normalize`

`mlx-community/bge-small-en-v1.5-bf16` (and BGE-style models generally)
already return L2-normalized pooled vectors from `mlx_embeddings.generate`.
`normalize=True` (the default) re-normalizes defensively, so the unit-norm
guarantee holds regardless of which model you point `model=` at -- useful
because `cosine_similarity` on unit vectors is numerically identical to
`dot_product`, which is cheaper. Pass `normalize=False` to get the model's
raw pooled output untouched.

## CI / testing: the fake engine seam

Every local-backend feature in this package (completions and embeddings)
shares the same testing seam: a deterministic, dependency-free fake engine
that never imports `mlx`, plus one environment variable that forces it:

```bash
POLAR_LLAMA_LOCAL_ENGINE=fake
```

Setting this env var (or passing `engine="fake"` explicitly) routes
`embedding_local` through `FakeEmbeddingEngine` instead of
`MlxEmbeddingEngine`. Each text's vector is derived from a SHA-256 hash of
that text: identical text always produces an identical vector (so a
duplicated document is provably its own nearest neighbor in a test), and
different text produces a different vector, all without downloading or
running a real model. This is what the package's own CI-safe test suite
(`tests/test_local_embeddings.py`, marker `local`) uses -- it runs on any
machine, including Linux, with zero `mlx` install.

For advanced test setups, you can also inject a specific engine instance
directly into the registry, bypassing the environment variable:

```python
from polar_llama.local.embed import FakeEmbeddingEngine, register_embedding_engine

register_embedding_engine("my-model", FakeEmbeddingEngine("my-model"), engine="in_process")
```

The real (`local_gpu`-marked) end-to-end test --
`tests/test_local_embeddings_gpu.py` -- embeds a small offline document set
with the real `mlx-community/bge-small-en-v1.5-bf16` model, builds an
`HnswIndex`, embeds a query, and asserts the semantically matching document
ranks first. It is excluded from CI by the same `-m "not local_gpu"` filter
used everywhere else in this repo's local-backend test suite, and is meant
to be run by hand on an Apple-Silicon Mac.

## Throughput

Measured on an Apple-Silicon Mac (M-series), warmed up (model already
loaded, so load time is excluded), embedding 256 short documents
(`batch_size=64`, default model, `normalize=True`):

```
256 docs in 0.193s -> ~1,330 docs/sec
```

For comparison, a hosted-API embedding call (e.g. OpenAI
`text-embedding-3-small` via `embedding_async`) is bound by network
round-trip latency and provider rate limits rather than local compute --
`embedding_local` trades a larger, higher-dimensional embedding space
(1536 dims for `text-embedding-3-small` vs. 384 for `bge-small`) for zero
network latency, zero API cost, and offline availability. Choose
`embedding_async` when embedding quality/dimensionality on a specific
provider's model matters most; choose `embedding_local` when you need to
run fully offline, at zero marginal cost, or without shipping text to a
third party.

## See also

- [`docs/local_mlx_backend.md`](local_mlx_backend.md) -- the completion-side
  local backend (`inference_local`), which this feature's engine/registry
  design directly mirrors.
- [`docs/VECTOR_SIMILARITY_AND_ANN.md`](VECTOR_SIMILARITY_AND_ANN.md) --
  `cosine_similarity`, `knn_hnsw`, and `HnswIndex`, all of which accept
  `embedding_local`'s output unchanged.
