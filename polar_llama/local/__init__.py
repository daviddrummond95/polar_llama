"""Local in-process LLM backend (MLX on Apple Silicon).

This package lets `pl.col(...).llama.inference_local(...)` run completions
against a local model instead of a remote provider API. Two engines are
supported: an OpenAI-compatible "server" engine (routes through the existing
Rust async fan-out) and an "in_process" engine backed by mlx-lm's
BatchGenerator/batch_generate.

IMPORTANT: `mlx` / `mlx-lm` are an OPTIONAL extra (`pip install
polar-llama[local]`) and are never imported at module import time. Importing
`polar_llama.local` must always succeed, even on machines without MLX
installed (e.g. Linux CI runners).

See docs/local_mlx_backend.md for the full design and usage guide.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HAS_MLX",
    "is_mlx_available",
    "require_mlx",
    # Lazily re-exported from sibling modules (see __getattr__ below):
    "LocalEngine",
    "FakeEngine",
    "MlxBatchEngine",
    "inference_local",
    "PrefixCache",
    "start_server_backend",
]

# Names lazily re-exported from sibling modules, keyed by the module that
# defines them. Sibling modules are only imported the first time one of
# their attributes is actually accessed, so `import polar_llama.local` never
# imports mlx/mlx_lm (or anything else heavy) eagerly.
_LAZY_ATTRS = {
    "LocalEngine": "engine",
    "FakeEngine": "engine",
    "MlxBatchEngine": "engine",
    "inference_local": "expr",
    "PrefixCache": "prefix_cache",
    "start_server_backend": "server_backend",
}


def is_mlx_available() -> bool:
    """Return True if `mlx_lm` can be imported in this environment.

    This performs the import lazily (only when called) so that merely
    importing `polar_llama.local` never triggers an mlx import.
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def require_mlx() -> Any:
    """Import and return the `mlx_lm` module, or raise a helpful error.

    Use this at the top of any function that actually needs mlx/mlx-lm
    (e.g. inside `MlxBatchEngine`), rather than importing mlx_lm at module
    scope.
    """
    try:
        import mlx_lm
    except ImportError as exc:
        raise ImportError(
            "the local in-process backend needs mlx-lm; install with: "
            "pip install polar-llama[local]"
        ) from exc
    return mlx_lm


# `HAS_MLX` is exposed as a module-level name for convenience, but computed
# lazily via module `__getattr__` below so that simply importing this
# package does not eagerly probe for mlx either.
def __getattr__(name: str) -> Any:
    if name == "HAS_MLX":
        return is_mlx_available()

    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
