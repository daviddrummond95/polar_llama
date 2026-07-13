"""CI-safe regression tests for issue #70 mlx-lm workarounds.

No mlx imports: the patch functions are guarded no-ops without mlx, and the
engine wiring is exercised with fake mlx_lm modules injected via sys.modules.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import polars as pl
import pytest

import polar_llama.local._mlx_patches as patches
import polar_llama.local.engine as engine_mod
from polar_llama.local.engine import (
    FakeEngine,
    MlxBatchEngine,
    clear_registry,
    register_engine,
)
from polar_llama.local.expr import inference_local

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global registry between tests."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# 1. Both patches run, in order, BEFORE load; and load-once holds.
# ---------------------------------------------------------------------------
def test_ensure_loaded_applies_both_patches_before_load(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "_require_mlx", lambda: calls.append("require"))
    monkeypatch.setattr(
        patches,
        "apply_gemma3n_batched_shared_kv_patch",
        lambda: (calls.append("gemma3n"), True)[1],
    )
    monkeypatch.setattr(
        patches,
        "apply_batchgen_stats_zerodiv_patch",
        lambda: (calls.append("stats"), True)[1],
    )

    fake_mlx = types.ModuleType("mlx_lm")

    def load(model, tokenizer_config=None, model_config=None):
        calls.append("load")
        return ("MODEL", "TOK")

    fake_mlx.load = load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)

    eng = MlxBatchEngine("dummy")
    assert eng.get_model_and_tokenizer() == ("MODEL", "TOK")
    assert calls == ["require", "gemma3n", "stats", "load"]

    # load-once: no further patch/load calls on second access.
    eng.get_model_and_tokenizer()
    assert calls == ["require", "gemma3n", "stats", "load"]


# ---------------------------------------------------------------------------
# 1b. The helper itself never raises even if a patch blows up.
# ---------------------------------------------------------------------------
def test_apply_mlx_patches_swallows_patch_errors(monkeypatch):
    def boom():
        raise RuntimeError("patch exploded")

    monkeypatch.setattr(patches, "apply_gemma3n_batched_shared_kv_patch", boom)
    monkeypatch.setattr(patches, "apply_batchgen_stats_zerodiv_patch", boom)
    engine_mod._apply_mlx_patches()  # must not raise


# ---------------------------------------------------------------------------
# 2. FakeEngine end-to-end never touches the patch functions.
# ---------------------------------------------------------------------------
def test_fake_engine_path_never_touches_patches(monkeypatch):
    monkeypatch.setattr(
        patches,
        "apply_gemma3n_batched_shared_kv_patch",
        lambda: pytest.fail("patch called on FakeEngine path"),
    )
    monkeypatch.setattr(
        patches,
        "apply_batchgen_stats_zerodiv_patch",
        lambda: pytest.fail("patch called on FakeEngine path"),
    )
    register_engine("dummy-70", FakeEngine("dummy-70"), engine="in_process")
    df = pl.DataFrame({"prompt": ["a", "b"]})
    out = df.with_columns(
        answer=inference_local(pl.col("prompt"), model="dummy-70", engine="in_process")
    )
    assert out["answer"].to_list() == ["echo:a", "echo:b"]


# ---------------------------------------------------------------------------
# 3. The stats guard: unmasks, guards clean-zero exit, is idempotent.
# ---------------------------------------------------------------------------
def _install_fake_generate(monkeypatch, prompt_time):
    """Inject a fake mlx_lm.generate whose stats() replicates the unguarded
    upstream 0.31.3 finally block (source-sniffable via inspect.getsource)."""

    class BatchStats:
        def __init__(self):
            self.prompt_tokens = 0
            self.prompt_time = 0.0
            self.prompt_tps = 0.0
            self.generation_tokens = 0
            self.generation_time = 0.0
            self.generation_tps = 0.0

    class BatchGenerator:
        _pt = prompt_time

        @contextmanager
        def stats(self, stats=None):
            stats = stats or BatchStats()
            try:
                yield stats
            finally:
                stats.prompt_tokens += 5
                stats.prompt_time += self._pt
                stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
                stats.generation_tokens += 3
                stats.generation_time += 1.0
                stats.generation_tps = stats.generation_tokens / stats.generation_time

    mod = types.ModuleType("mlx_lm.generate")
    mod.BatchStats = BatchStats
    mod.BatchGenerator = BatchGenerator
    pkg = types.ModuleType("mlx_lm")
    monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", mod)
    return mod


def test_stats_zerodiv_patch_unmasks_and_guards(monkeypatch):
    mod = _install_fake_generate(monkeypatch, prompt_time=0.0)
    assert patches.apply_batchgen_stats_zerodiv_patch() is True

    gen = mod.BatchGenerator()

    # Unmasking: the body's error propagates, not ZeroDivisionError.
    with pytest.raises(ValueError, match="real error"):
        with gen.stats():
            raise ValueError("real error")

    # Clean zero-work exit: no crash, tps guarded to 0.0.
    with gen.stats() as st:
        pass
    assert st.prompt_tps == 0.0
    assert st.generation_tps == 0.0

    # Idempotent: identity unchanged, still True.
    first = mod.BatchGenerator.stats
    assert patches.apply_batchgen_stats_zerodiv_patch() is True
    assert mod.BatchGenerator.stats is first


def test_stats_zerodiv_patch_normal_path_unchanged(monkeypatch):
    mod = _install_fake_generate(monkeypatch, prompt_time=2.0)
    assert patches.apply_batchgen_stats_zerodiv_patch() is True
    gen = mod.BatchGenerator()
    with gen.stats() as st:
        pass
    assert st.prompt_tps == 2.5  # 5 / 2.0, unchanged behavior
    assert st.generation_tps == 3.0  # 3 / 1.0
    assert st.generation_tokens == 3  # post-division accumulation ran


# ---------------------------------------------------------------------------
# 4. No-op paths: mlx absent, and upstream-already-guarded.
# ---------------------------------------------------------------------------
def test_stats_zerodiv_patch_noop_when_module_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", None)  # forces ImportError
    assert patches.apply_batchgen_stats_zerodiv_patch() is False


def test_stats_zerodiv_patch_noop_when_already_guarded(monkeypatch):
    class BatchGenerator:
        @contextmanager
        def stats(self, stats=None):
            # Guarded division: the divisor is bound to a local before use, so
            # the unguarded upstream substring never appears in this source,
            # and the source-sniff gates the patch off.
            try:
                yield stats
            finally:
                pt = stats.prompt_time if stats is not None else 0.0
                if pt:
                    stats.prompt_tps = stats.prompt_tokens / pt

    mod = types.ModuleType("mlx_lm.generate")
    mod.BatchGenerator = BatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", mod)

    before = BatchGenerator.stats
    assert patches.apply_batchgen_stats_zerodiv_patch() is False
    assert BatchGenerator.stats is before  # untouched
