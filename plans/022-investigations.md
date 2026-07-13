# Plan 022: Answer three time-boxed investigations (mlx KV-share indexing, instant-distance health, tool-validation default) and record findings in this file

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Do NOT create or edit `plans/README.md` — the
> advisor maintains the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- polar_llama/local/_mlx_patches.py polar_llama/tools.py src/ann.rs Cargo.toml Cargo.lock pyproject.toml`
> If any of these files changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `afc78da`, 2026-07-06

## What this plan is (read this first)

This is an **INVESTIGATE plan**. The deliverable is a **written report appended
to THIS file** under a `## Findings` heading (Step 5), plus *at most* new
characterization tests marked `xfail`/`skip`. You must make **NO production
code changes** — no edits to `polar_llama/`, `src/`, `Cargo.toml`,
`pyproject.toml`, or anything else outside this plan file and (optionally) one
new test file.

Three independent questions, each time-boxed to ~1–2 hours. If one is blocked
(missing `[local]` extra, no network), write what WAS checkable in its Findings
section and move on to the next — a partially-answered investigation with the
blocker documented still counts as done for that section.

Each investigation's Findings section must contain these six items:
1. **Question** (restated in one sentence)
2. **Method** (what you ran/read)
3. **Evidence** (file:line citations, URLs, versions — every claim cited)
4. **Answer**
5. **Recommendation** — exactly one of `keep` / `change` / `investigate-more`, with justification
6. **Effort estimate** for any follow-up plan (S/M/L, or "none needed")

## Why this matters

Three open questions carry latent risk that nobody has written down: (1) a
monkeypatch of mlx-lm's Gemma-3n model has a read/write index asymmetry that
either is provably safe (then document why) or silently corrupts batched
generation (then flag it); (2) the sole ANN engine behind the public
`knn_hnsw` expression is `instant-distance 0.6`, whose maintenance status is
unknown; (3) tool-argument validation in `execute_tool_calls` is silently OFF
unless the caller remembers `tools=`, even though the emission side already
carries the schemas. Answering these turns "unknown risk" into either "documented
non-issue" or a scoped follow-up plan.

## Current state

All excerpts below were read from the working tree at commit `afc78da` and
line numbers verified.

### Investigation 1 files — mlx KV-share indexing

- `polar_llama/local/_mlx_patches.py` — the monkeypatch. Module docstring
  (lines 1–44) explains it works around
  https://github.com/ml-explore/mlx-lm/issues/1384 (present in mlx-lm <= 0.31.3).
- `docs/mlx_lm_1384_fix.md` — the project's own full analysis of the upstream
  bug and parity validation. Read it before starting.
- Installed upstream model file (verified present, mlx-lm **0.31.3**, venv is
  Python 3.14):
  `.venv/lib/python3.14/site-packages/mlx_lm/models/gemma3n.py`

The questioned loop, `polar_llama/local/_mlx_patches.py:222-248` (inside the
patched `language_model_call`):

```python
        intermediates = [(None, None)] * len(self.layers)
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = self.config.layer_types[i] == "full_attention"

            if is_global:
                mask = global_mask
            else:
                mask = sliding_window_mask

            cache_idx = self.layer_idx_to_cache_idx[i]
            c = cache[cache_idx]
            if i >= self.first_kv_shared_layer_idx and c is not None:
                shared_kv, offset = intermediates[cache_idx]
            else:
                shared_kv, offset = None, None

            h, kv, offset = layer(
                h,
                mask,
                c,
                per_layer_input,
                shared_kv=shared_kv,
                offset=offset,
            )
            intermediates[i] = (kv, offset)
```

The asymmetry: **read** at `intermediates[cache_idx]` (line 236), **store** at
`intermediates[i]` (line 248). This is safe if and only if, for every layer
that a shared layer's `cache_idx` points at, the pointed-at (concrete) layer
stored its `(kv, offset)` under that same index — i.e. iff
`layer_idx_to_cache_idx[j] == j` for every concrete layer `j`.

Preliminary evidence gathered during planning (you must re-verify and cite it
yourself): the installed
`.venv/lib/python3.14/site-packages/mlx_lm/models/gemma3n.py:446-454` builds
the mapping as:

```python
        self.layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(self.config.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
            else:
                if layer_type == "full_attention":
                    self.layer_idx_to_cache_idx.append(shared_full_idx)
                elif layer_type == "sliding_attention":
                    self.layer_idx_to_cache_idx.append(shared_sliding_idx)
                else:
                    raise NotImplementedError(f"Unknown layer type: {layer_type}")
```

i.e. identity for concrete layers (`i < first_kv_shared_layer_idx`), and
`shared_full_idx`/`shared_sliding_idx` (both computed at gemma3n.py:438-444
as indices **within the concrete prefix**, so always `< first_kv_shared_layer_idx`)
for shared layers. If your read confirms this, the asymmetry is safe: a shared
layer's `cache_idx` always names a concrete layer, and that concrete layer
stored under its own index because identity holds on the concrete prefix.

Also relevant: the patch's upstream-fixed guard at
`polar_llama/local/_mlx_patches.py:73-78` — the patch no-ops when
`"cache.state"` is absent from `Gemma3nAttention.__call__`'s source. Verified
during planning: installed gemma3n.py **line 133** still contains
`keys, values = cache.state`, so the patch is still live on 0.31.3.

Note a naming discrepancy you should resolve as part of this investigation:
the finding brief says "PR #1384" but the module docstring
(`_mlx_patches.py:6-7`) cites **issue** #1384. Check upstream mlx-lm (GitHub
issue tracker / release notes / newer PyPI versions) for whether a fix has
merged, and in which release, so the report can say when the monkeypatch
becomes retirable.

### Investigation 2 files — instant-distance health

- `Cargo.toml:37` — `instant-distance = { version = "0.6", features = ["with-serde"] }`
- `Cargo.lock:1503-1504` — resolved version is `instant-distance 0.6.1`
- `src/ann.rs` — sole consumer. Line 1:
  `use instant_distance::{Builder, HnswMap, Point, Search};`
  `build_hnsw_index` (lines 30–39) builds `HnswMap<EmbeddingPoint, usize>`
  with `Builder::default()`; `EmbeddingPoint::distance` (lines 8–27) is
  cosine distance.
- `polar_llama/__init__.py:922` — public `knn_hnsw(...)` expression;
  namespace method at line 1412.
- `docs/VECTOR_SIMILARITY_AND_ANN.md` — user-facing docs for the feature.

### Investigation 3 files — tool-validation default

- `polar_llama/tools.py` — the whole tool-use pipeline.
  - Emission side: `tools_to_response_model` (lines 114–204). Line 203
    attaches the normalized specs to the generated Pydantic model:
    ```python
    response_model.__polar_llama_tools__ = specs
    ```
  - Execution side: `execute_tool_calls` (lines 405–498). Validation is
    opt-in — schemas are built ONLY when the caller passes `tools=`
    (lines 466–470):
    ```python
    tool_schemas: Optional[Dict[str, Any]] = None
    if tools is not None:
        tool_schemas = {
            spec["name"]: spec["input_schema"] for spec in _normalize_tools(tools)
        }
    ```
    The Python executor path validates per call at lines 586–591
    (`unknown tool` fail-fast, then `_validate_arguments`); the Rust/MCP path
    gets `kwargs["tool_schemas"] = json.dumps(tool_schemas)` at lines 485–486.
  - `_normalize_tool` (lines 50–95): a Pydantic `BaseModel` subclass is
    normalized as ONE tool (`name=class name`, `input_schema=model_json_schema()`)
    — so passing the emission response model as `tools=[model]` today would
    WRONGLY register a single tool named e.g. `"ToolCalls"`, not unpack
    `__polar_llama_tools__`. This matters for the prototype design.
  - `_validate_arguments` (lines 621–639): full validation only when the
    `jsonschema` package is importable; otherwise required-keys-only.
    **Verified: `jsonschema` is NOT installed in this repo's `.venv`**, so the
    Python fallback path is the one that runs locally (the docstring at
    lines 622–623 says the Rust path always validates fully).
- Key structural fact for the "auto-discover" question: the emission output is
  a Polars **column** (struct with a `calls` field); the response-model object
  — the thing carrying `__polar_llama_tools__` — is NOT reachable from the
  column. Discovery therefore cannot come "from the column" without new
  plumbing; the realistic designs are (a) accept the response model itself as
  `tools=` (unpack `__polar_llama_tools__` in `_normalize_tools`), or (b) have
  emission embed schemas into the column/metadata. Evaluate both in the report.
- Exemplar tests: `tests/test_tool_use.py` (14 tests, all pass today;
  `model.__polar_llama_tools__` is exercised at lines 193 and 215).

### Conventions

- Pytest markers (pyproject.toml:65-68): `local_gpu` = requires Apple GPU +
  mlx (skipped in CI); `local` = local backend logic tests (CI-safe). Any new
  characterization test that needs mlx/Apple GPU must be marked
  `@pytest.mark.local_gpu`; a pure-logic test (e.g. asserting the
  `layer_idx_to_cache_idx` identity property from config, no GPU) can be
  `@pytest.mark.local`.
- Design constraints you must NOT contradict in recommendations:
  `pyproject.toml:51-54` documents that mlx-lm 0.31.3 requires
  transformers>=5 and must not be capped; `.cargo/audit.toml` documents pyo3
  pinned <0.29 by pyo3-polars. Do not recommend dependency bumps that violate
  these.

## Commands you will need

All run from `/Users/daviddrummond/SideProjects/polar-llama` (use absolute
paths if your cwd resets).

| Purpose | Command | Expected on success (verified 2026-07-06) |
|---------|---------|-------------------------------------------|
| Sanity: venv works | `.venv/bin/python -c "import polar_llama, pytest; print('ok')"` | prints `ok` |
| mlx-lm presence + version + path | `.venv/bin/python -c "import mlx_lm; print(mlx_lm.__version__); import mlx_lm.models.gemma3n as m; print(m.__file__)"` | `0.31.3` and `.../site-packages/mlx_lm/models/gemma3n.py` (ImportError ⇒ STOP condition A for investigation 1) |
| Patch trigger still live | `grep -n "cache.state" .venv/lib/python3.14/site-packages/mlx_lm/models/gemma3n.py` | one hit at line 133 |
| Resolved ANN version | `grep -n -A2 '^name = "instant-distance"' Cargo.lock` | `version = "0.6.1"` |
| Tool-use tests baseline | `.venv/bin/python -m pytest tests/test_tool_use.py -q` | `14 passed` |
| New characterization tests (if you write any) | `.venv/bin/python -m pytest tests/<your new file> -q -m "not local_gpu"` | all pass or xfail/skip; zero failures |
| Nothing else touched | `git status --porcelain` | only `plans/022-investigations.md` modified plus optional new `tests/test_*.py` file(s); untracked files that predate this plan (`.DS_Store`, `test_cache_*.py`, `test_optional_llm.py` at repo root) are not yours — leave them alone |

Network commands (best-effort; failure is NOT an error, it triggers the
per-investigation "report what was checkable" path):

| Purpose | Command |
|---------|---------|
| Latest mlx-lm on PyPI | `.venv/bin/pip index versions mlx-lm` (or `curl -s https://pypi.org/pypi/mlx-lm/json \| python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`) |
| instant-distance on crates.io | `cargo search instant-distance --limit 3` (or `curl -s https://crates.io/api/v1/crates/instant-distance` and read `newest_version`, `updated_at`) |
| GitHub signals | WebSearch/WebFetch of `https://github.com/instant-labs/instant-distance` (releases, open issues) and `https://github.com/ml-explore/mlx-lm/issues/1384` — only if web tooling is available in your environment |

## Suggested executor toolkit

- If WebSearch/WebFetch tools are available, use them for the upstream
  mlx-lm issue #1384 status and instant-distance GitHub maintenance signals.
  If not available, fall back to the `pip index` / `curl` commands above; if
  those also fail (offline), record "network unavailable — checked
  Cargo.lock/installed versions only" in the relevant Findings section.
- Read `docs/mlx_lm_1384_fix.md` and `docs/VECTOR_SIMILARITY_AND_ANN.md`
  before writing Findings sections 1 and 2 respectively.

## Scope

**In scope** (the only files you may modify/create):
- `plans/022-investigations.md` — append the `## Findings` section (this file).
- Optionally: ONE new test file, `tests/test_022_characterization.py`
  (characterization tests only, marked `xfail`/`skip`/`local`/`local_gpu` as
  appropriate — tests that document current behavior, not tests that demand
  new behavior).

**Out of scope** (do NOT touch, even though they look related):
- ANY file under `polar_llama/`, `src/`, `docs/` — no production code changes,
  no doc edits, even one-line "obvious fixes". If investigation 1 finds real
  corruption, you WRITE IT UP; you do not fix it.
- `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `requirements*.txt` — no
  dependency changes, no version bumps, no adding `jsonschema`.
- `.venv/` — read-only; never edit installed packages.
- `plans/README.md` and every other file in `plans/` — advisor-maintained.
- Existing test files — do not modify `tests/test_tool_use.py` etc.

## Git workflow

- Branch: `advisor/022-investigations` (create from `main` at or after `afc78da`).
- Commit style: sentence-case imperative summary, matching the repo's log
  (e.g. `Refresh cargo-audit ignore rationale for the pyo3 CVEs`). Suggested:
  one commit, `Record findings for plan 022 investigations` (plus the optional
  test file in the same commit).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Baseline and environment probe

Run the drift check from the header, then the first five commands from the
table above. Record which of the following are true (these gate the
investigations): mlx-lm importable (yes/no + version), network reachable for
PyPI/crates.io (yes/no).

**Verify**: `.venv/bin/python -m pytest tests/test_tool_use.py -q` → `14 passed`.

### Step 2: Investigation 1 — Gemma-3n KV-share indexing in the monkeypatch

Time box: ~2 hours.

1. Read `polar_llama/local/_mlx_patches.py` in full (269 lines) and
   `docs/mlx_lm_1384_fix.md`.
2. Read the installed
   `.venv/lib/python3.14/site-packages/mlx_lm/models/gemma3n.py`, specifically:
   the `layer_idx_to_cache_idx` construction (lines 438–456 in 0.31.3), the
   `first_kv_shared_layer_idx` definition (~line 389), and `make_cache`
   (~line 554: caches are created only for the concrete-layer prefix).
3. Answer: is `layer_idx_to_cache_idx[j] == j` for every concrete layer `j`
   (i.e. every `j < first_kv_shared_layer_idx`), and is every shared layer's
   `cache_idx` strictly less than `first_kv_shared_layer_idx`? Cite exact
   installed-file lines. If BOTH hold, the store-at-`i` / read-at-`cache_idx`
   asymmetry in `_mlx_patches.py:236/248` is safe — explain the invariant in
   the report in 3–5 sentences. If either fails, describe the concrete
   corruption scenario (which layer reads whose KV) as a characterization
   note — do NOT write a fix.
4. Upstream status: determine whether mlx-lm has fixed issue #1384
   (check the GitHub issue if web is available; regardless, run the PyPI
   latest-version command and, if a newer version's source is fetchable,
   check whether `Gemma3nAttention.__call__` still reads `cache.state`).
   Report: "fixed in version X ⇒ monkeypatch retirable once pin moves past X"
   or "unfixed as of version X" or "network unavailable; installed 0.31.3
   still affected (gemma3n.py:133)". Note that the patch already self-disables
   on fixed versions via the guard at `_mlx_patches.py:73-78`, so
   "retirable" means "deletable dead code", not "urgent".
5. Optional characterization test (no GPU needed): a `@pytest.mark.local`
   test in `tests/test_022_characterization.py` that imports
   `mlx_lm.models.gemma3n` (guarded with
   `pytest.importorskip("mlx_lm.models.gemma3n")`), constructs the model
   config or replicates the mapping construction, and asserts the identity
   property on the concrete prefix. Only write it if it can pass/skip without
   downloading model weights.

**Verify**: Findings section 1 exists in this file with all six required
items, and every code claim carries a `file:line` citation.

### Step 3: Investigation 2 — instant-distance maintenance health

Time box: ~1 hour. NO migration work — assessment only.

1. Confirm the dependency surface: `Cargo.toml:37` (pin `0.6`,
   `with-serde`), `Cargo.lock` resolved `0.6.1`, and that `src/ann.rs` is the
   only consumer (`grep -rn "instant_distance" src/` should hit only
   `src/ann.rs`).
2. Gather maintenance signals (best-effort, network permitting): latest
   release + date on crates.io; last commit and open-issue count on
   `github.com/instant-labs/instant-distance`; any open issues about
   correctness or performance that would affect this repo's usage
   (`Builder::default()`, cosine distance, `HnswMap` with `with-serde`).
3. Assess and recommend exactly one of:
   - **keep** — actively maintained or stable-and-done with no issues
     affecting our usage;
   - **watch** (report as `keep` with a watch note) — stale but no known
     defects; state what event should trigger re-evaluation;
   - **replace** (report as `investigate-more`) — known defects or
     abandonment that matters; in that case give a one-paragraph rationale
     each for the candidates `hnsw_rs` and `usearch` (license, maintenance,
     serde support, API fit with `src/ann.rs`'s `HnswMap<EmbeddingPoint, usize>`
     usage) and an effort estimate for a migration plan. Do NOT migrate.
4. If the network is unreachable, report exactly what was checkable offline
   (Cargo.lock version, code surface, `cargo tree -i instant-distance` if it
   works offline) and mark the recommendation `investigate-more` with the
   blocker named.

**Verify**: Findings section 2 exists with all six items; recommendation is
one of keep/change/investigate-more; every external claim has a URL or a
"network unavailable" note.

### Step 4: Investigation 3 — can tool-call validation be on by default?

Time box: ~2 hours. Prototype ON PAPER (pseudo-code in the report) — no
production edits.

1. Trace the emission→execution handoff end to end and cite it:
   `tools_to_response_model` attaches specs (`tools.py:203`) → user passes the
   model as `response_model=` to inference → the resulting COLUMN is fed to
   `execute_tool_calls`, which never sees the model object → validation only
   happens when `tools=` is re-supplied (`tools.py:466-470`, enforced at
   `tools.py:586-591` for the Python path, `tools.py:485-486` for the Rust
   path). Confirm by grep: `grep -rn "__polar_llama_tools__" polar_llama/ tests/`
   → attachment at `polar_llama/tools.py:203`, docs mention at `tools.py:151`,
   reads only in `tests/test_tool_use.py:193,215` — i.e. NOTHING in the
   execution path reads it today.
2. Design sketch (pseudo-code in the report) for each option, with tradeoffs:
   - **Option A**: `execute_tool_calls(tools=response_model)` — teach
     `_normalize_tools`/`execute_tool_calls` that an object bearing
     `__polar_llama_tools__` means "unpack these specs". Note the trap
     documented in "Current state": today a BaseModel subclass is normalized
     as ONE tool via `model_json_schema()` (`tools.py:62-67`), so option A
     must check `__polar_llama_tools__` BEFORE the BaseModel branch.
   - **Option B**: emission embeds the schemas in the output (e.g. a
     `_tools` metadata field in the struct column or a serialized sidecar),
     execution auto-discovers from the column. Assess cost: schema bytes
     repeated per row vs once, dtype changes, backward compat of the column
     shape (`{"calls": [...]}` is documented at `tools.py:127-130` and
     consumed by `_serialize_calls_batch` at `tools.py:501-528`).
   - **Option C**: status quo + louder docs (opt-in stays).
3. Enumerate backward-compat risks of default-on validation — calls that pass
   today but would be rejected: (a) LLM-emitted arguments with extra/missing
   fields currently execute when `tools=` is omitted; (b) local behavior
   differs by environment because `_validate_arguments` (`tools.py:621-639`)
   does full validation only when `jsonschema` is installed (it is NOT in
   this repo's venv) while the Rust path always validates fully — default-on
   makes this inconsistency user-visible; (c) columns built by hand or by a
   different tool list than the one executed against would newly fail with
   `unknown tool`.
4. Recommend default-on vs opt-in (map to keep/change/investigate-more) with
   rationale and an effort estimate for the follow-up implementation plan.
5. Optional characterization test: a `@pytest.mark.local`-style plain pytest
   in `tests/test_022_characterization.py` pinning CURRENT behavior, e.g.
   "execute_tool_calls with an executor and NO tools= executes a call whose
   arguments violate the schema" (model it on the executor-path tests in
   `tests/test_tool_use.py`). This documents the status quo the follow-up
   plan would change.

**Verify**: Findings section 3 exists with all six items, includes pseudo-code
for at least options A and B, and lists at least the three backward-compat
risks above (plus any you find).

### Step 5: Assemble the Findings section and commit

Append to THIS file (`plans/022-investigations.md`), after the
"Maintenance notes" section, exactly this structure:

```markdown
## Findings

_Investigated at commit `<sha you ran against>`, <date>. Environment:
mlx-lm <version or "absent">, network <available|unavailable>._

### 1. Gemma-3n KV-share indexing (`_mlx_patches.py`)
- **Question**: ...
- **Method**: ...
- **Evidence**: ...
- **Answer**: ...
- **Recommendation**: keep | change | investigate-more — ...
- **Follow-up effort**: ...

### 2. instant-distance 0.6 health (`src/ann.rs`)
(same six items)

### 3. Tool-validation default (`polar_llama/tools.py`)
(same six items, plus the pseudo-code sketches and backward-compat list)
```

Then commit on `advisor/022-investigations`.

**Verify**:
- `grep -c "Recommendation" plans/022-investigations.md` → at least 3 in the
  Findings section (one per investigation).
- `git status --porcelain` → only this plan file (M) and, if written,
  `tests/test_022_characterization.py` (A/??); pre-existing untracked root
  files unchanged.
- `.venv/bin/python -m pytest tests/test_tool_use.py -q` → still `14 passed`
  (proves you changed no production behavior).

## Test plan

Characterization tests are OPTIONAL and capped at one new file,
`tests/test_022_characterization.py`:

- Investigation 1 (optional): assert the `layer_idx_to_cache_idx` identity
  property on the concrete prefix using the installed mlx-lm module; guard
  with `pytest.importorskip("mlx_lm.models.gemma3n")` and mark
  `@pytest.mark.local`. Must not download weights or require a GPU; if it
  would, mark `@pytest.mark.local_gpu` or skip it entirely.
- Investigation 3 (optional): pin current opt-out behavior of
  `execute_tool_calls` with a Python `executor=` and no `tools=`; model the
  structure on `tests/test_tool_use.py` (fixtures and assertion style).
- If a test documents a BUG found in investigation 1, mark it
  `@pytest.mark.xfail(reason="mlx_lm #1384 patch indexing — see plans/022-investigations.md Findings §1", strict=False)`.
- Verification: `.venv/bin/python -m pytest tests/test_022_characterization.py tests/test_tool_use.py -q -m "not local_gpu"`
  → zero failures (passes, skips, and xfails are all acceptable).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `plans/022-investigations.md` contains a `## Findings` heading with
      three subsections (`### 1.`, `### 2.`, `### 3.`), each containing all
      six items (Question/Method/Evidence/Answer/Recommendation/Follow-up effort).
- [ ] Every Recommendation line contains exactly one of `keep`, `change`,
      `investigate-more`.
- [ ] Findings §1 cites at least one `file:line` into the installed
      mlx-lm `gemma3n.py` (or documents that mlx-lm was not importable).
- [ ] Findings §2 names the resolved version `0.6.1` (from `Cargo.lock`) and
      either external evidence (URL/date) or an explicit "network unavailable" note.
- [ ] Findings §3 contains pseudo-code for the auto-discovery wiring and a
      backward-compat risk list with ≥3 entries.
- [ ] `git status --porcelain` shows only `plans/022-investigations.md`
      modified and optionally `tests/test_022_characterization.py` added —
      nothing under `polar_llama/`, `src/`, or any dependency manifest.
- [ ] `.venv/bin/python -m pytest tests/test_tool_use.py -q` → `14 passed`.
- [ ] If a new test file exists:
      `.venv/bin/python -m pytest tests/test_022_characterization.py -q -m "not local_gpu"`
      exits 0 (skips/xfails allowed, failures not).

## STOP conditions

Stop and report back (do not improvise) if:

- **(A) per-investigation, does not stop the whole plan**: the tooling for one
  investigation is unavailable — mlx-lm not importable for §1, no network for
  the external half of §2/§1.4. Write "what was checkable" in that Findings
  section, mark its recommendation `investigate-more`, and continue with the
  other investigations. Only this plan's OTHER stop conditions halt everything.
- The drift check shows `polar_llama/local/_mlx_patches.py` or
  `polar_llama/tools.py` changed since `afc78da` and the "Current state"
  excerpts no longer match — the questions may already be answered or moot.
- The installed mlx-lm version is NOT 0.31.3 (e.g. the venv was rebuilt):
  re-derive the gemma3n.py line numbers before citing, and if
  `Gemma3nAttention.__call__` no longer contains `cache.state`, report that
  the patch is already dormant and pivot §1 to "confirm retirability".
- Answering any question seems to require editing production code (e.g. adding
  instrumentation to `_mlx_patches.py`) — it never does; use reads, the
  Python REPL against installed modules, and standalone test files only.
- You find what looks like real, currently-reachable data corruption in §1:
  finish the write-up, mark the recommendation `change`, flag it prominently
  at the top of the Findings section as **MAINTAINER ATTENTION**, and stop
  before writing any fix.

## Maintenance notes

For the human/agent who owns this after the findings land:

- Findings §1's safety argument is version-scoped: it holds for the
  `layer_idx_to_cache_idx` construction in mlx-lm 0.31.3. If the `[local]`
  extra's mlx-lm floor moves (`pyproject.toml:54`, currently `>=0.31`), the
  invariant must be re-checked against the new `gemma3n.py` — or better, the
  monkeypatch retired if upstream fixed #1384 (the guard at
  `_mlx_patches.py:73-78` self-disables, but dead code should be deleted via
  a follow-up plan).
- Findings §2's verdict should be revisited whenever `src/ann.rs` grows
  features (filtering, larger indexes, f32 embeddings) — a "keep" for the
  current tiny surface is not a "keep" forever.
- Findings §3 is the input to a possible follow-up implementation plan
  (validation default-on / `tools=response_model` support). That plan — not
  this one — owns the `_normalize_tool` ordering trap (`tools.py:62-67`) and
  the jsonschema-optional inconsistency (`tools.py:621-639`).
- Reviewer: check that the Findings cite the INSTALLED gemma3n.py (inside
  `.venv/.../mlx_lm/models/`), not the patch file, for the mapping claim; and
  that no production file was touched (`git diff --stat main` should show
  only this plan file and at most one new test file).
