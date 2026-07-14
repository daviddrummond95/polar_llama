# Deterministic Run Manifests

## Overview

A `RunManifest` (`polar_llama/manifest.py`, issue #85) is a small,
JSON-serializable audit record of *what a run asked for* -- provider, model,
prompt/schema content (as hashes, not text), sampling params, and the
fingerprint the runtime itself used for dedupe/checkpointing -- with a
`manifest_id` computed deterministically from those fields. Two runs with
identical configuration get the same `manifest_id` even if they ran hours
apart and produced different `created_at` timestamps or different token
counts.

Pure Python (dataclasses + stdlib `hashlib`/`json`), no Rust changes, mirroring
`polar_llama/hitl.py` / `polar_llama/codebook.py`: DataFrame-in/DataFrame-out
(or dataclass-out) orchestration, not a row-wise expression -- `build_manifest`
and friends are **not** added to the `.llama` expression namespace.

```
   inference_async(...) / inference_messages(...)
         |
         v
   build_manifest(out, symbol=..., provider=..., model=..., ...)
         |                -- config_fingerprint reuses the exact
         |                   `request_fingerprint` the #75 checkpoint /
         |                   #77 dedupe paths already compute
         v
   manifest.save(path)     -- atomic JSON sidecar, sorted keys
         |
         v
   with_manifest_id(out, manifest)  -- attach a manifest_id column to results
         |
         v
   ... later, in a different process/run ...
         v
   load_manifest(path)     -- raises ManifestIntegrityError if tampered
         |
         v
   replay(manifest, new_df, "prompt", system_prompt=..., ...)
         |                -- raises ManifestMismatchError if content diverges
         v
   re-run against fresh row data, with a `response` + `manifest_id` column
```

## Design principles

**Reuse, not reimplementation.** `config_fingerprint` is computed by calling
`polar_llama.keys.request_fingerprint` -- the exact function
`inference_async`/`inference_messages` call for the #75 checkpoint path and
the #77 dedupe/response-cache path (moved there verbatim from
`polar_llama/__init__.py`; the old `__init__.py`-level name,
`_request_fingerprint`, is now just an alias so those call sites are
untouched). A manifest's `config_fingerprint` is therefore *guaranteed* to
equal the runtime fingerprint for the same `(symbol, provider, model,
schema, system_prompt)` -- not a parallel hash that could drift out of sync.
Likewise `checkpoint_id`/`response_cache_id` are read straight out of a
`Checkpoint`/`ResponseCache` store's `_meta.json` (issues #75/#77) rather
than recomputed, `aggregate_usage` sums a `usage=True` output column's
`USAGE_DTYPE` struct (issue #76) unmodified, and `dedupe_stats` snapshots a
`polar_llama.DedupeStats` (issue #77) verbatim.

**The determinism hash excludes anything the runtime can't pin down.**
`manifest_id` is a sha256 over a canonical-JSON subset of fields
(`RunManifest.hashed_fields()`), using the exact canonicalization
`config_fingerprint` uses (`sort_keys=True, separators=(",", ":"),
ensure_ascii=False`). Explicitly **excluded**: `manifest_id` itself,
`created_at` (the one field two identical runs are *permitted* to differ
on), `aggregate_usage`/`dedupe_stats` (nondeterministic output, not
request-shaping), and `checkpoint_id`/`response_cache_id` (identify a
*resume mechanism*, not the request -- same philosophy as
`config_fingerprint` excluding `cache_*`). `polar_llama_version` and
`endpoint` (the provider base-URL override) **are** in the hash -- a
library upgrade or a different endpoint is a config change for audit
purposes.

**Prompt-hash vs. replay tension -- verify-then-replay.** The manifest
stores sha256 hashes of `system_prompt`/`prompt_template`/the response
schema, never their plaintext, so a manifest is safe to share as an audit
artifact without leaking prompt IP. This means `replay()` cannot
reconstruct a call from the manifest alone by design: the caller re-supplies
the prompt/schema/row data, and `replay()` verifies (by hash) that what was
supplied matches what the manifest recorded *before* re-running -- catching
silent drift instead of silently replaying against changed content. Callers
who want fully self-contained replay can opt in via
`build_manifest(..., store_texts=True)`, which embeds the plaintext in
`RunManifest.texts` (itself excluded from the determinism hash, since the
hashes already cover that content).

**Params / server-path caveat.** `inference_async`/`inference_messages` do
not forward `temperature`/`max_tokens`/`seed` to hosted providers today --
those only exist on the local-MLX path (`.llama.inference_local`). So
`params={}` on a hosted-path manifest means "provider server-side defaults
apply", which this library cannot pin or verify; the manifest guarantees
what the *client* asked for, not what the provider actually used. This
caveat applies to `replay()` too: it re-issues the client-visible request,
not a guarantee of identical provider-side sampling.

## Quick start

```python
import polars as pl
from polar_llama import (
    Provider, inference_async,
    build_manifest, with_manifest_id, load_manifest, replay,
)

SYS = "You are a terse assistant."
df = pl.DataFrame({"prompt": ["Summarize photosynthesis.", "Explain gravity."]})

out = df.with_columns(
    response=inference_async(
        pl.col("prompt"), provider=Provider.OPENAI, model="gpt-4o-mini",
        system_prompt=SYS,
    )
)

# Build + save a manifest describing this run's configuration.
manifest = build_manifest(
    out, symbol="inference_async", provider="openai", model="gpt-4o-mini",
    system_prompt=SYS,
)
manifest.save("runs/2026-07-14.manifest.json")

# Attach the manifest_id to the results DataFrame for downstream traceability.
out = with_manifest_id(out, manifest)

# ... later, in a different process ...
loaded = load_manifest("runs/2026-07-14.manifest.json")  # integrity-checked
new_df = pl.DataFrame({"prompt": ["Explain entropy."]})
replayed = replay(loaded, new_df, "prompt", system_prompt=SYS)
# replayed has `response` + `manifest_id` columns; raises ManifestMismatchError
# if SYS (or the response_model, or prompt_template) doesn't hash-match.
```

## Determinism

```python
m1 = build_manifest(provider="openai", model="gpt-4o-mini", system_prompt="hi")
m2 = build_manifest(provider="openai", model="gpt-4o-mini", system_prompt="hi")
assert m1.manifest_id == m2.manifest_id          # identical config -> identical id
assert m1.created_at != m2.created_at or True    # created_at may differ; excluded from the hash

m3 = build_manifest(provider="openai", model="gpt-4o", system_prompt="hi")
assert m3.manifest_id != m1.manifest_id          # any config change flips the id
```

Every field in `RunManifest.hashed_fields()` participates:
`format_version`, `polar_llama_version`, `symbol`, `provider`, `model`,
`config_fingerprint`, `system_prompt_hash`, `prompt_template_hash`,
`response_model_name`, `response_schema_hash`, `params`, `seed`,
`endpoint`, `row_count`.

## Usage & checkpoint/cache linkage

```python
from polar_llama import Checkpoint, DedupeStats, ResponseCache

out = df.with_columns(
    r=inference_async(pl.col("prompt"), model="gpt-4o-mini", usage=True)
)
manifest = build_manifest(out, usage_column="r", model="gpt-4o-mini")
manifest.aggregate_usage
# {"input_tokens": ..., "output_tokens": ..., "cached_tokens": ...,
#  "cost_usd": ..., "rows_with_usage": ...}
```

`checkpoint=`/`response_cache=` (a path, `Checkpoint`, or `ResponseCache`)
populate `checkpoint_id`/`response_cache_id` from that store's `_meta.json`
fingerprint; `dedupe_stats=` (a `polar_llama.DedupeStats`) snapshots
`rows_total`/`rows_null`/`cache_hits`/`rows_collapsed`/`calls_made`. All
three are recorded for audit but excluded from `manifest_id` -- see
"Design principles" above.

## Integrity checking

```python
from polar_llama import load_manifest, ManifestIntegrityError

try:
    m = load_manifest("runs/2026-07-14.manifest.json")
except ManifestIntegrityError:
    ...  # the file was hand-edited or corrupted after save_manifest wrote it
```

`load_manifest` recomputes `manifest_id` over the loaded determinism fields
and compares it to the stored value -- a tamper/corruption check, not just a
JSON-parse check.

## Replay

`replay(manifest, df, input_column, *, system_prompt=None, response_model=None,
prompt_template=None, verify=True)` re-issues the request a manifest
describes against re-supplied row data:

1. If `verify` (default `True`): hashes the supplied `system_prompt` /
   `prompt_template` / `response_model` schema and compares against the
   manifest's recorded hashes, and recomputes `config_fingerprint` (held
   against the manifest's *recorded* endpoint, so this stays a pure content
   check). Any mismatch raises `ManifestMismatchError` naming every
   diverging field. Separately, current environment drift -- the active
   endpoint override, or the installed `polar-llama` version -- differing
   from what the manifest recorded is a `UserWarning`, not an error (it
   doesn't necessarily mean the *request* changed).
2. If the manifest was built with `store_texts=True`, an omitted
   `system_prompt`/`prompt_template` is auto-filled from
   `manifest.texts` -- fully self-contained replay.
3. Reconstructs the call: `inference_async` for an
   `symbol="inference_async"` manifest, `inference_messages` for
   `symbol="inference_messages"` (note: `inference_messages` has no
   `system_prompt` parameter -- messages carry their own system turn, so
   `replay` never forwards one there). `usage=` is set to
   `manifest.aggregate_usage is not None`, matching whether the original
   run used `usage=True`.
4. Returns `df` with a `response` column and a `manifest_id` column
   (via `with_manifest_id`) appended. Row data is never reconstructed from
   the manifest -- it never stores rows.

`params`/`seed` are recorded on the manifest for audit but not forwarded by
`replay` -- see the params/server-path caveat above; on hosted paths there
is nothing to forward them *to*, so replay runs against current server
defaults, same as the original run did.

## API reference

| Name | Purpose |
|---|---|
| `RunManifest` | Frozen dataclass; see `polar_llama/manifest.py` docstring for every field. |
| `build_manifest(df=None, *, symbol=, provider=, model=, system_prompt=, response_model=, prompt_template=, params=, seed=, usage_column=, checkpoint=, response_cache=, dedupe_stats=, store_texts=)` | Build a `RunManifest`. |
| `with_manifest_id(df, manifest, *, column="manifest_id")` | Attach a constant `manifest_id` column to `df`. |
| `save_manifest(manifest, path)` / `manifest.save(path)` | Atomic JSON sidecar write. |
| `load_manifest(path)` / `RunManifest.load(path)` | Load + integrity-check. |
| `replay(manifest, df, input_column, *, system_prompt=, response_model=, prompt_template=, verify=True)` | Verify-then-replay. |
| `ManifestIntegrityError` | Raised by `load_manifest` on a tampered/corrupted file. |
| `ManifestMismatchError` | Raised by `replay(verify=True)` on a content mismatch. |
| `MANIFEST_FORMAT_VERSION` | Current on-disk format version (`1`). |

See also `tests/test_manifest.py` for the full determinism/round-trip/replay
test suite (CI-safe -- no network calls, no API keys, no mlx).
