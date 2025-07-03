import types, sys
import polars as pl

# pyright: reportMissingImports=false

# Provide a lightweight stub so the test suite can import polars_mcp_runtime
# even if the real package is not installed in CI.
module = types.ModuleType("polars_mcp_runtime")

def _submit_df(plan: pl.LazyFrame | pl.DataFrame):  # type: ignore[name-defined]
    """Very small stand-in for the real MCP submit helper.

    For unit-testing we just *collect* if the object is lazy, otherwise return
    the DataFrame unchanged.  No network calls are made.
    """
    if hasattr(plan, "collect"):
        return plan.collect()
    return plan

module.submit_df = _submit_df  # type: ignore[attr-defined]

sys.modules["polars_mcp_runtime"] = module