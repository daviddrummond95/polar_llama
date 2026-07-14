"""Price table + cost computation for `usage=True` accounting (issue #76).

The packaged default table (`polar_llama/data/prices.json`) is seeded from
the compile-time Rust table in `src/cost.rs` plus published cached-input
rates. It is a second source of truth vs. `src/cost.rs` -- tracked as
follow-up tech debt (unify via a `build.rs` generation step), flagged in the
issue #76 PR description; not addressed in this change.

Resolution order for a given (provider, model): per-call ``price_table=``
override > process-wide registry (:func:`set_price_table` /
:func:`register_model_price`) > packaged default. A ``price_table=`` value
may be a ``dict`` shaped like the packaged JSON, or a path to a JSON file in
the same shape.
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "Price",
    "load_default_price_table",
    "set_price_table",
    "get_price_table",
    "register_model_price",
    "resolve_price",
    "compute_cost",
    "effective_model",
]


#: Mirrors `get_default_model` in `src/expressions.rs` -- used ONLY so cost
#: resolution has a model name to look up when the caller didn't pass
#: `model=` explicitly (Rust picks its own default independently; this is a
#: best-effort mirror for pricing display, not load-bearing for the request
#: itself). If the two ever drift, the symptom is a wrong/null `cost_usd`
#: for the affected provider's default model, never a wrong request.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-8",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def effective_model(provider: Optional[str], model: Optional[str]) -> Optional[str]:
    """`model` if given, else the mirrored default for `provider` (see DEFAULT_MODELS)."""
    if model is not None:
        return model
    if provider is None:
        return DEFAULT_MODELS["openai"]
    return DEFAULT_MODELS.get(provider.lower())


@dataclass(frozen=True)
class Price:
    """Per-1M-token pricing for one (provider, model)."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: Optional[float] = None

    def effective_cached_input_per_1m(self) -> float:
        """`cached_input_per_1m` if set, else fall back to `input_per_1m`."""
        return (
            self.cached_input_per_1m
            if self.cached_input_per_1m is not None
            else self.input_per_1m
        )


def _price_from_entry(entry: Dict[str, Any]) -> Price:
    return Price(
        input_per_1m=float(entry["input_per_1m"]),
        output_per_1m=float(entry["output_per_1m"]),
        cached_input_per_1m=(
            float(entry["cached_input_per_1m"])
            if entry.get("cached_input_per_1m") is not None
            else None
        ),
    )


def _normalize_table(raw: Dict[str, Any]) -> Dict[str, Dict[str, Price]]:
    table: Dict[str, Dict[str, Price]] = {}
    for provider, models in raw.items():
        if provider.startswith("_"):
            continue
        if not isinstance(models, dict):
            continue
        table[provider.lower()] = {
            model: _price_from_entry(entry) for model, entry in models.items()
        }
    return table


@functools.lru_cache(maxsize=1)
def load_default_price_table() -> Dict[str, Dict[str, Price]]:
    """Load and cache the packaged default price table."""
    try:
        import importlib.resources as resources

        data_path = resources.files("polar_llama").joinpath("data/prices.json")
        raw = json.loads(data_path.read_text())
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        # Fallback for unusual install layouts (e.g. editable installs where
        # importlib.resources can't resolve the package data directory).
        fallback = Path(__file__).parent / "data" / "prices.json"
        raw = json.loads(fallback.read_text())
    return _normalize_table(raw)


# Process-wide registry override, merged over the packaged default.
_REGISTRY: Dict[str, Dict[str, Price]] = {}

# One warning per (provider, model) per process for unknown-model cost lookups.
_WARNED_UNKNOWN: set = set()


def set_price_table(table: Union[Dict[str, Any], str, Path]) -> None:
    """Replace the process-wide price registry (merged over the packaged default).

    Parameters
    ----------
    table : dict or str or Path
        Either a dict shaped like the packaged ``prices.json`` (
        ``{"<provider>": {"<model>": {"input_per_1m": ..., "output_per_1m":
        ..., "cached_input_per_1m": ...}}}``), or a path to a JSON file in
        that shape.
    """
    global _REGISTRY
    if isinstance(table, (str, Path)):
        raw = json.loads(Path(table).read_text())
    else:
        raw = table
    _REGISTRY = _normalize_table(raw)


def get_price_table() -> Dict[str, Dict[str, Price]]:
    """Return the effective table: packaged defaults merged with registry overrides."""
    merged: Dict[str, Dict[str, Price]] = {
        provider: dict(models)
        for provider, models in load_default_price_table().items()
    }
    for provider, models in _REGISTRY.items():
        merged.setdefault(provider, {}).update(models)
    return merged


def register_model_price(
    provider: str,
    model: str,
    *,
    input_per_1m: float,
    output_per_1m: float,
    cached_input_per_1m: Optional[float] = None,
) -> None:
    """Upsert a single (provider, model) price into the process-wide registry."""
    provider_key = provider.lower()
    _REGISTRY.setdefault(provider_key, {})[model] = Price(
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
        cached_input_per_1m=cached_input_per_1m,
    )


def _strip_bedrock_region_prefix(model: str) -> Optional[str]:
    """Strip a leading Bedrock cross-region inference-profile prefix.

    ``us.anthropic.claude-...`` -> ``anthropic.claude-...``. Returns ``None``
    if `model` doesn't look like it has one of the known prefixes.
    """
    for prefix in ("us.", "eu.", "apac."):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return None


def resolve_price(
    provider: Optional[str],
    model: Optional[str],
    override: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> Optional[Price]:
    """Resolve pricing for (provider, model).

    Resolution order: `override` (per-call dict/path) > process-wide
    registry > packaged default. For Bedrock, a miss retries once with the
    leading region prefix (``us.``/``eu.``/``apac.``) stripped, since
    cross-region inference-profile IDs commonly aren't listed verbatim.
    Returns ``None`` (never raises) when nothing matches.
    """
    if not provider or not model:
        return None
    provider_key = provider.lower()

    candidates: list = []
    if override is not None:
        if isinstance(override, (str, Path)):
            raw = json.loads(Path(override).read_text())
        else:
            raw = override
        candidates.append(_normalize_table(raw))
    candidates.append(get_price_table())

    for table in candidates:
        provider_table = table.get(provider_key)
        if not provider_table:
            continue
        if model in provider_table:
            return provider_table[model]
        if provider_key == "bedrock":
            stripped = _strip_bedrock_region_prefix(model)
            if stripped is not None and stripped in provider_table:
                return provider_table[stripped]
    return None


def compute_cost(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_tokens: Optional[int],
    price: Price,
) -> float:
    """``((input - cached) * in_rate + cached * cached_in_rate + output * out_rate) / 1e6``.

    Null ``cached_tokens`` is treated as 0 for the formula (but the
    caller-facing ``cached_tokens`` field itself stays null -- this function
    only affects the cost arithmetic).
    """
    input_v = input_tokens or 0
    output_v = output_tokens or 0
    cached_v = cached_tokens or 0
    non_cached = max(input_v - cached_v, 0)
    cost = (
        non_cached * price.input_per_1m
        + cached_v * price.effective_cached_input_per_1m()
        + output_v * price.output_per_1m
    )
    return cost / 1_000_000.0


def warn_unknown_model_once(provider: Optional[str], model: Optional[str]) -> None:
    """Emit one UserWarning per (provider, model) per process for cost-null lookups."""
    key = (provider or "", model or "")
    if key in _WARNED_UNKNOWN:
        return
    _WARNED_UNKNOWN.add(key)
    warnings.warn(
        f"No pricing for {provider}/{model}: cost_usd will be null. "
        "Register with polar_llama.register_model_price(...) or pass "
        "price_table=... to the inference call.",
        UserWarning,
        stacklevel=3,
    )
