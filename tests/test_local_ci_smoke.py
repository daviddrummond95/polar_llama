"""CI smoke test for the local backend on a machine WITHOUT mlx.

This is the canary the Linux CI job runs (``-m "local and not local_gpu"``):
it proves the local-backend package imports and its ``in_process`` logic runs
end-to-end using the dependency-free ``FakeEngine`` -- no ``mlx``/``mlx-lm``,
no GPU, no network. If any of these fail, the ``[local]`` extra has leaked a
mandatory mlx import into the base package.

The real ``MlxBatchEngine`` path is deliberately NOT exercised here; it lives
behind the optional ``[local]`` extra and requires Apple-Silicon GPU hardware
(marked ``local_gpu`` elsewhere and excluded from CI).
"""

from __future__ import annotations

import sys

import polars as pl
import pytest

pytestmark = pytest.mark.local


def test_import_local_does_not_import_mlx():
    """Importing ``polar_llama.local`` must never eagerly import mlx/mlx-lm."""
    import polar_llama.local as local

    assert local is not None
    # Merely importing the package must not have pulled in the optional deps.
    assert "mlx" not in sys.modules
    assert "mlx_lm" not in sys.modules


def test_is_mlx_available_false_in_ci():
    """On CI (and any machine without the [local] extra) mlx is unavailable."""
    from polar_llama.local import HAS_MLX, is_mlx_available

    assert is_mlx_available() is False
    assert HAS_MLX is False


def test_require_mlx_raises_helpful_error():
    """``require_mlx`` should point users at the optional extra, not crash raw."""
    from polar_llama.local import require_mlx

    with pytest.raises(ImportError, match=r"polar-llama\[local\]"):
        require_mlx()


def test_inference_local_in_process_fake_row_order(monkeypatch):
    """FakeEngine path of ``inference_local(engine="in_process")`` runs on CPU.

    Forces the dependency-free FakeEngine via the ``POLAR_LLAMA_LOCAL_ENGINE``
    override so the whole ``in_process`` map_batches UDF is exercised without
    importing mlx. Asserts a String column returned in ORIGINAL ROW ORDER.
    """
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    from polar_llama.local.engine import clear_registry
    from polar_llama.local.expr import inference_local

    clear_registry()
    try:
        prompts = [f"prompt-{i}" for i in range(25)]
        df = pl.DataFrame({"prompt": prompts})
        result = df.with_columns(
            answer=inference_local(
                pl.col("prompt"),
                model="ci-smoke-model",
                engine="in_process",
                max_tokens=512,
            )
        )

        # Deterministic FakeEngine output, one row per input, original order.
        assert result.schema["answer"] == pl.String
        assert result["answer"].to_list() == [f"echo:{p}" for p in prompts]
    finally:
        clear_registry()
