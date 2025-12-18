# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
