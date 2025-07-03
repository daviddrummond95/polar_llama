import sys
import types

# We only create a stub if the real package is missing.

try:
    import polars_mcp_runtime  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover – CI fallback
    import polars as pl  # type: ignore

    module = types.ModuleType("polars_mcp_runtime")

    def submit_df(plan):  # type: ignore[override]
        """Collect lazy plans or return DataFrames unchanged (local stub)."""

        if hasattr(plan, "collect"):
            return plan.collect()
        return plan

    module.submit_df = submit_df  # type: ignore[attr-defined]

    sys.modules["polars_mcp_runtime"] = module