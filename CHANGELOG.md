# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2026-07-05

### Added
- **Local prompt-tuning bridge** — `polar_llama.local.make_local_inference_fn(model, ...)` returns an `inference_fn` that drives `polar_llama.optimize`'s DSPy-style `Predict` / `BootstrapFewShot` / `InstructionOptimizer` against on-device gemma-3n (mlx-lm), with no cloud/Rust path. It reuses the singleton-loaded weights and applies the mlx-lm #1384 batched fix automatically.
- **Collapsed-prefill opt-in for `inference_local(engine="in_process")`** via `POLAR_LLAMA_LOCAL_COLLAPSE=1` — shares the common prompt prefix across rows (also the bridge's default). Measured **~2.8× faster on a full prompt-tuning schedule** (3.4× on a demo-laden eval) at identical, parity-verified output; the dominant speedup whenever rows share a long prefix (a shared `system` prompt, or few-shot demos during tuning). Mutually exclusive with `POLAR_LLAMA_LOCAL_KV_BITS` (that path takes precedence).
- `MlxBatchEngine.get_model_and_tokenizer()` to reuse the singleton-loaded weights.

### Fixed
- `InstructionOptimizer` no longer crashes with `TypeError: the truth value of a Series is ambiguous` when the proposer model returns the `instructions` field as a JSON array instead of a newline-delimited string — list/Series values are flattened to newline-delimited text (`polar_llama/optimize.py`). This surfaces with small local models that emit `{"instructions": [...]}`.

## [0.5.0] - 2026-07-04

### Added
- **Local MLX inference backend** — `col(...).llama.inference_local(...)` for on-device batched generation on Apple Silicon, with two engines:
  - `engine="server"` (default): points the existing async fan-out at a local OpenAI-compatible endpoint (`mlx_lm.server` / vllm-mlx) via `OPENAI_BASE_URL` — no Rust changes.
  - `engine="in_process"`: a `map_batches` UDF wrapping `mlx_lm`'s `BatchGenerator`, behind a `LocalEngine` protocol with a `FakeEngine` seam so the batching/ordering/error-isolation logic is testable on CPU/CI without a GPU. Optional extra: `pip install polar-llama[local]`.
- **Collapsed prefix prefill** (`polar_llama/local/collapsed_prefill.py`): computes a shared prompt prefix once instead of re-prefilling it per row (token-level longest-common-prefix + suffix-only batching). Measured **10.36× vs sequential** on gemma-3n E4B (32 rows, 5 KB shared prompt) at 32/32 exact greedy parity.
- **Batched quantized KV cache** (`BatchQuantizedKVCache`; opt-in via `POLAR_LLAMA_LOCAL_KV_BITS=4`): closes mlx-lm's "quantized KV × batching" gap. ~47% KV-memory cut (Q4) at parity with fp16 batched output, roughly doubling the batch/context that fits in 24 GB (memory/capacity win; not a throughput speedup with the current unfused attention path).
- **mlx-lm #1384 fix** (runtime monkeypatch, `polar_llama/local/_mlx_patches.py`): corrects a RoPE offset-aliasing bug that garbled *batched* generation on hybrid Gemma 3n / Gemma 4 models; verified token-identical to sequential. Ready-to-post upstream PR in `patches/PR_1384.md`.
- Build-vs-buy gate benchmark (`benchmarks/local_mlx_gate.py`), parity/throughput validators, and design/decision docs under `docs/`.

### Notes
- The local backend and its Apple-GPU tests are opt-in; CI runs the CPU-safe logic tests (`-m "not local_gpu"`, FakeEngine, no mlx). The `[local]` extra requires Python ≥ 3.10.

## [0.3.0] - 2026-06-10

### Added
- **Tool use / MCP integration** (`tools_to_response_model`, `mcp_tools`, `execute_tool_calls`, `tool_results_to_message`, and `.llama` namespace methods): LLMs emit tool calls as structured output and `execute_tool_calls` runs every call of every row batch-parallel against an MCP server (`tools/call`) or a Python `executor` callable; per-call failures are data. Guide: `docs/TOOL_USE.md`; example: `examples/tool_use_calorie_tracker.py`.
- **Provider-native prompt caching** (`cache=True` / `CacheConfig`): shares a cached system prefix across rows via Anthropic `cache_control` content blocks, with 5-minute and 1-hour (`ttl="1h"`, `extended-cache-ttl` beta) TTLs; `inference_messages` now also accepts List(Struct) input in addition to JSON strings.
- **DSPy-style prompt optimization engine** (`polar_llama.optimize`):
  - `Signature` — declarative task specs (`"question -> answer"` shorthand or explicit `InputField`/`OutputField` with types and descriptions)
  - `Predict` — executable LLM module that runs one parallel, batched inference per DataFrame and returns `pred_<field>` columns
  - `evaluate` — metric-based scoring of a module against a labeled DataFrame
  - `BootstrapFewShot` — mines few-shot demonstrations from training rows the module already answers correctly (akin to `dspy.BootstrapFewShot`)
  - `InstructionOptimizer` — COPRO-style instruction search: an LLM proposes instruction rewrites, every candidate is evaluated on the trainset, best one wins
  - Fully testable offline via an injectable `inference_fn` backend
- `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` environment overrides for proxies and gateways
- `POLAR_LLAMA_MAX_CONCURRENCY` environment variable to bound concurrent in-flight requests per batch (default 64; previously unbounded)
- Gemini: native `system_instruction` support and native JSON-schema structured outputs (`response_json_schema`)
- Pricing data for current models (GPT-5/4.1/o-series, Claude 4.x/Fable 5, Gemini 2.5, Bedrock Claude 4.5) and o200k tokenizer detection for GPT-4.1/GPT-5/o3/o4

### Changed
- **Updated default models** (previous defaults were retired/decommissioned):
  - OpenAI: `gpt-4-turbo` → `gpt-4o-mini`
  - Anthropic: `claude-3-opus-20240229` (retired) → `claude-opus-4-8`
  - Gemini: `gemini-1.5-pro` → `gemini-2.5-flash`
  - Groq: `llama3-70b-8192` (decommissioned) → `llama-3.3-70b-versatile`
  - Bedrock: `anthropic.claude-3-haiku-20240307-v1:0` → `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- OpenAI/Groq requests no longer hardcode `temperature`/`max_tokens` (newer models such as the o-series and GPT-5 reject those parameters)
- Bedrock region now respects `AWS_REGION`/`AWS_DEFAULT_REGION` before falling back to `us-east-1`
- Removed import-time debug printing from the Python package and the native module
- Minimum supported Python version is now 3.9 (abi3-py39 wheels)
- Removed deprecated `new()`/`with_model()` Rust constructors (deprecated since 0.2.0); use `new_with_model()`

### Fixed
- **Gemini structured outputs** previously sent OpenAI-style Bearer auth and always failed; Gemini now authenticates via the `x-goog-api-key` header on all paths
- **Bedrock structured outputs** previously attempted a raw HTTP POST to the Bedrock endpoint; they now route through the AWS SDK like plain requests
- Bedrock now works from the synchronous `inference` expression (previously returned an error)
- TLS verification is no longer disabled (`danger_accept_invalid_certs` removed); the shared client uses rustls with the OS certificate store
- **Bedrock prompt caching** now actually emits a `CachePoint` block on the Converse request when `cache_control` is set (previously the marker was injected but never sent). Bedrock supports only the default ~5-minute cache type.
- **Prompt-cache grouping**: rows with no cacheable system prefix are no longer lumped into a single serial cache group (they are processed as individual rows).
- **Windows wheels build again**: the AWS SDK now uses the modern `ring` rustls provider (`aws-smithy-http-client/rustls-ring`) instead of `aws-lc-rs`, whose `aws-lc-sys` native build fails under MSVC ("C atomics require C11 or later"). This also drops the legacy rustls 0.21 path.
- Bumped vulnerable transitive crates flagged by `cargo audit` (bytes, quinn-proto, rustls-webpki, time, anyhow, memmap2). The remaining pyo3 < 0.29 advisories are pinned by pyo3-polars 0.26 and are explicitly ignored in CI until the bridge supports pyo3 0.29.

### Performance
- `inference_messages` now accepts `List(Struct{role, content})` input natively in Rust, so the default path no longer wraps every call in a Python `map_batches` UDF (keeps queries lazy/streaming)
- Single shared HTTP client with connection pooling (previously a new client per batch)
- Bounded request concurrency via buffered streams instead of unbounded `join_all`
- JSON schemas are compiled once per batch instead of once per row
- AWS Bedrock client/credential chain is cached per region instead of being rebuilt per request
- Vector similarity ops (`cosine_similarity`, `dot_product`, `euclidean_distance`) use a contiguous-slice fast path
- Tokenizer/pricing lookups hoisted out of per-row loops in cost expressions
- Removed redundant per-row clones of schemas, models, and message batches in the expression layer (~400 lines of duplicated dispatch removed)

### Security
- **Fixed RUSTSEC-2025-0020** (pyo3 buffer overflow): upgraded pyo3 0.23 → 0.27 via pyo3-polars 0.26
- Removed `ureq` 2.x and `once_cell` dependencies (sync path now reuses the async clients; `std::sync::LazyLock` replaces `once_cell`)
- Updated polars 0.46 → 0.53, jsonschema 0.28 → 0.46, tiktoken-rs 0.6 → 0.12, aws-sdk-bedrockruntime to latest

## [0.2.2] - 2025-12-17

### Added
- **LLM cost calculation** with tiktoken tokenization for accurate token counting and cost estimation
- **Vector embeddings** (`embedding_async`) - Parallelized, memory-efficient embedding generation
  - Support for OpenAI, Gemini, and AWS Bedrock embedding models
  - Streaming approach for minimal memory footprint
- **Vector similarity functions** for high-performance vector operations:
  - `cosine_similarity` - Measure angle between vectors
  - `dot_product` - Calculate dot product
  - `euclidean_distance` - Calculate straight-line distance
- **Approximate Nearest Neighbor (ANN) search** via HNSW algorithm (`knn_hnsw`)
  - Sub-linear O(log N) search time
  - High recall rates (>95% typical)
  - Scalable to millions of vectors
- Comprehensive repository grading rubric and documentation improvements
- CODE_OF_CONDUCT.md for community guidelines
- SECURITY.md for vulnerability reporting
- Complete API documentation for all Polars expressions
- Architecture diagram in documentation
- Dependency scanning with Dependabot and cargo-audit
- Code coverage reporting in CI pipeline
- `.cargo/audit.toml` configuration for documented security exceptions

### Fixed
- Struct schema inference for Pydantic models with Optional fields
- Mermaid flowchart rendering in documentation

### Security
- **Fixed RUSTSEC-2025-0024**: Updated crossbeam-channel from 0.5.14 to 0.5.15 (double free on Drop)
- **Fixed RUSTSEC-2024-0421**: Updated idna dependency via url crate upgrade (Punycode label issue)
- **Fixed RUSTSEC-2025-0009**: Updated ring from 0.17.11 to 0.17.14 (AES panic issue)
- Updated ureq from 0.11 to 2.x to fix rustls 0.16 vulnerabilities and webpki issues (RUSTSEC-2024-0336, RUSTSEC-2023-0052)
- Updated tokio from 1.37 to 1.48 to fix unsound broadcast channel issue (RUSTSEC-2025-0023)
- Updated reqwest from 0.11 to 0.12 to get newer rustls versions
- Updated futures from 0.3.30 to 0.3.31 to avoid yanked version
- **Documented RUSTSEC-2025-0020** (pyo3 buffer overflow): Cannot be fixed yet as pyo3-polars 0.20.0 requires pyo3 0.23
  - The vulnerability is in PyString::from_object which this codebase doesn't directly use
  - Risk assessed as LOW for our use case
  - Will update when polars 0.52+ stabilizes and pyo3-polars supports pyo3 0.24+
  - Documented in audit.toml with justification
- **3 out of 4 critical vulnerabilities resolved**, 1 documented and accepted with risk assessment

## [0.2.1] - 2025-11-19

### Added
- Link-Time Optimization (LTO) for release builds to improve performance
- Single codegen unit for release builds

### Changed
- Added `llama` namespace for better Python package organization
- Performance optimizations in release configuration

### Fixed
- Additional error handling improvements

## [0.2.0] - 2025-11-12

### Added
- **Taxonomy-based tagging feature** with detailed reasoning and confidence scores
  - Support for hierarchical taxonomies
  - Multi-category classification
  - Confidence scoring for each tag
  - Reasoning explanation for tag assignments
  - Comprehensive documentation in `docs/TAXONOMY_TAGGING.md`
- **Structured outputs with Pydantic integration**
  - Support for Pydantic models as response schemas
  - Automatic validation of LLM responses
  - Polars struct-based output for structured data
  - Parallel validation for batch operations
- Structured output validation tests
- Comprehensive test suite for taxonomy tagging
- Python 3.8 compatibility for structured outputs

### Changed
- Updated documentation for Pydantic structured outputs feature
- Enhanced error handling for API responses

### Fixed
- Clippy `needless_question_mark` lint warning
- Dead code warning in `AnthropicContent` struct
- Python 3.8 compatibility issues in test suite

## [0.1.6] - 2024-03-08

### Added
- Comprehensive test suite for LLM inference interfaces
- Support for all providers in synchronous inference
- GitHub workflow dispatch for manual CI triggers
- Expanded test coverage

### Changed
- Refactored test organization and structure
- Professionalized README and PyPI configuration
- Cleaned up expression declarations

### Fixed
- Synchronous inference now supports all providers (not just async)
- AI inference error handling
- Import paths and Python layer structure

## [0.1.5] - 2024-03-08

### Added
- AWS Bedrock provider support
- Message history support for multi-turn conversations
- PyPI package publishing support

### Changed
- Improved import structure and Python abstraction layer
- Cleaned up module exports

### Fixed
- Polars expression registration issues
- Import errors in tests
- Non-existent feature flags removed

## [0.1.0] - 2024-03-08

### Added
- Initial release of Polar Llama
- OpenAI provider support
- Anthropic (Claude) provider support
- Google Gemini provider support
- Groq provider support
- Parallel asynchronous inference via Polars expressions
- Multi-message conversation support
- PyO3-based Python bindings
- Tokio async runtime integration
- Basic test suite
- MIT License
- Initial documentation

### Features
- `inference()` - Synchronous LLM inference
- `inference_async()` - Parallel asynchronous inference
- `inference_messages()` - Multi-message conversations
- `string_to_message()` - Message formatting helper
- `combine_messages()` - Message array handling
- Provider abstraction via ModelClient trait

[Unreleased]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/daviddrummond95/polar_llama/releases/tag/v0.1.0
