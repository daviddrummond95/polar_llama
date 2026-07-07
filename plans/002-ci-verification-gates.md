# Plan 002: Make CI verification real — gate on cargo test, un-swallow pytest/safety, fix fake-pass skips

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- .github/workflows/CI.yml tests/test_structured_outputs.py tests/test_taxonomy_tagging.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/001-fix-cargo-audit-ci-break.md
- **Category**: tests
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

CI currently gives green checkmarks without verifying much: no workflow ever
runs `cargo test`, so all 15 Rust unit tests (in `src/cache.rs`, `src/cost.rs`,
`src/ann.rs`, `src/mcp.rs`) never execute in CI; the coverage job's pytest and
tarpaulin runs end in `|| true` so the whole job is non-gating; the Python
dependency scan (`safety check || true`) can never fail; and two test files
"skip" by printing a message and doing a bare `return`, so keyless CI reports
them as PASSED while executing nothing. After this plan lands, a red CI run
means something actually failed, a green run means the Rust unit tests, the
Python test suite, and the dependency scan actually passed, and skipped tests
are reported as SKIPPED.

## Current state

All excerpts verified against the working tree at commit `afc78da`.

Relevant files:

- `.github/workflows/CI.yml` — the main CI workflow. Jobs: `security-audit`,
  `coverage`, `linux_tests` (matrix py3.9–3.12), wheel builds
  (`linux`/`windows`/`macos`), `sdist`, `release`. `RUSTFLAGS: "-Dwarnings"`
  is set globally at line 23.
- `tests/test_structured_outputs.py` — live-API Groq structured-output tests
  with bare-return key guards (lines 24–26 and 84–86).
- `tests/test_taxonomy_tagging.py` — live-API Anthropic taxonomy tests with
  bare-return key guards (lines 13–15, 100–102, 177–179).
- `tests/test_embeddings.py` — exemplar for the repo's correct skip pattern
  (`pytest.mark.skipif`, lines 13–16).
- `.github/workflows/test-llm-apis.yml` — separate manual live-API workflow.
  NOT in scope; it is where the live-API versions of these tests are meant to
  run.

### Fact 1: no workflow runs `cargo test`

`grep -rn "cargo test" .github/workflows/` returns nothing today. The Rust
unit tests exist and pass keyless — verified locally at `afc78da`:

```
$ cargo test --lib
running 15 tests
test ann::tests::test_embedding_point_distance ... ok
test cache::tests::test_analyze_batch_no_caching ... ok
... (15 total across ann, cache, cost, mcp modules)
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Also verified with CI's global env applied — `RUSTFLAGS="-Dwarnings"
cargo test --lib` → 15 passed, exit 0. This matters because CI.yml sets
`RUSTFLAGS: "-Dwarnings"` globally (line 23) and `cargo clippy
--all-features` (without `--all-targets`) does not lint the `cfg(test)`
build, so test-only warnings could not have been caught before.

`tests/model_client_tests.rs` (a Rust *integration* test target) needs live
API keys. `cargo test --lib` runs only the unit tests inside `src/` and does
NOT compile or run `tests/model_client_tests.rs` — that is exactly why
`--lib` is required and a bare `cargo test` must not be used.

Linux linking note (pre-verified — do not re-litigate): pyo3 is built with
the `extension-module` feature (`Cargo.toml:14`), which historically could
break `cargo test` linking on Linux (the pyo3 FAQ issue). This was verified
NOT to be a problem here: a minimal repro crate with the exact same shape
(pyo3 0.27, `features = ["extension-module", "abi3-py39"]`,
`crate-type = ["cdylib", "rlib"]`, the same `build.rs` calling
`pyo3_build_config::add_extension_module_link_args()`, a `#[pyclass]` +
`#[pymodule]` in lib.rs, and an inline `#[test]`) compiles, links, and passes
`cargo test --lib` on Linux (`rust:1-bookworm`, python3 + python3-dev
installed). The `linux_tests` job installs Python via `actions/setup-python`
before any Rust step, so pyo3-build-config finds an interpreter. No
`Cargo.toml` changes are needed — and none are in scope.

### Fact 2: coverage job is entirely non-gating

`.github/workflows/CI.yml:85-94`:

```yaml
      - name: Run Rust coverage
        run: |
          # Run cargo-tarpaulin for Rust code coverage
          cargo tarpaulin --out Xml --output-dir ./coverage --skip-clean --exclude-files 'tests/*' || true

      - name: Run Python coverage
        run: |
          source .venv/bin/activate
          # Run pytest with coverage (exclude Apple-GPU-only tests; local logic tests still run)
          pytest tests --cov=polar_llama --cov-report=xml:coverage/python-coverage.xml --cov-report=term -m "not local_gpu" --ignore=tests/test_parallel_inference.py || true
```

### Fact 3: safety scan can never fail

`.github/workflows/CI.yml:49-55`:

```yaml
      - name: Check for Python dependency vulnerabilities
        run: |
          python -m pip install --upgrade pip
          pip install safety
          pip install -r requirements.txt
          # Run safety check (allow to fail for now, but show results)
          safety check || true
```

`requirements.txt` is 8 unpinned packages: polars, maturin, ruff, pytest,
mypy, pandas, pyarrow, matplotlib. Verified locally at `afc78da`
(2026-07-07): with a Python 3.12 venv, `pip-audit -r requirements.txt`
resolves the latest versions and exits 0 with "No known vulnerabilities
found". (Beware: with an old Python 3.9 interpreter the resolver caps
versions — pytest 8.4.x, pyarrow 21, pillow 11 — and pip-audit then reports
8 known vulnerabilities. The CI job installs Python 3.10 via
`actions/setup-python`; see Step 4.)

### Fact 4: fake-pass key guards

`tests/test_structured_outputs.py:21-26` (same pattern again at lines 84–86):

```python
def test_structured_output_basic():
    """Test basic structured output with Groq API."""
    # Check if API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping test: GROQ_API_KEY not set")
        return
```

`tests/test_taxonomy_tagging.py:10-15` (same pattern again at lines 100–102
and 177–179):

```python
def test_taxonomy_tagging_basic():
    """Test basic taxonomy tagging with a simple two-field taxonomy."""
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping test: ANTHROPIC_API_KEY not set")
        return
```

Complete sweep of the pattern (verified — these 5 are ALL occurrences):

```
$ grep -rn 'print("Skipping' tests/
tests/test_structured_outputs.py:25
tests/test_structured_outputs.py:85
tests/test_taxonomy_tagging.py:14
tests/test_taxonomy_tagging.py:101
tests/test_taxonomy_tagging.py:178
```

Verified keyless behavior today (fake passes):

```
$ GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests/test_structured_outputs.py tests/test_taxonomy_tagging.py -v
...
5 passed in 0.08s
```

(Empty-string env vars work as "keyless" because both files call
`load_dotenv()`, and python-dotenv does NOT override variables already
present in the environment — so the developer's local `.env` keys are
masked. `os.getenv("...")` returns `""`, which is falsy. Never delete or
edit the repo's `.env`.)

The repo's correct skip pattern — exemplar `tests/test_embeddings.py:13-16`:

```python
skip_if_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
```

Neither `test_structured_outputs.py` nor `test_taxonomy_tagging.py` currently
imports `pytest` — the import must be added.

### Constraint: documented tradeoffs to honor

- The cargo-audit `--ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177`
  flags at `CI.yml:47` and their comment block (lines 41–46) are a documented
  decision (pyo3 pinned < 0.29 by pyo3-polars; rationale in
  `.cargo/audit.toml`). Do not touch them. Plan 001 governs the audit
  configuration (its own in-scope file is `.cargo/audit.toml`).
- The ruff/mypy `|| true` lines at `CI.yml:205-217` are a separate plan's
  scope (tool configs must exist first). Do not touch them.

## Commands you will need

Run from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.
A built `.venv` already exists (Python 3.14, has pytest, polar_llama, pyyaml).

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Rust unit tests (keyless) | `cargo test --lib` | `test result: ok. 15 passed; 0 failed` |
| Keyless pytest on changed files | `GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests/test_structured_outputs.py tests/test_taxonomy_tagging.py -v` | `5 skipped` (after Step 1; today: `5 passed`) |
| Broader Python suite (unchanged behavior) | `GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py -q` | exit 0, no failures |
| YAML syntax check | `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('yaml ok')"` | `yaml ok` |
| pip-audit dry run (needs python3.12 or 3.11 on PATH) | see Step 4 | `No known vulnerabilities found`, exit 0 |
| Sweep for fake-pass guards | `grep -rn 'print("Skipping' tests/` | no output, exit 1 |

Sweep for swallowed pytest/safety lines (kept out of the table because the
command contains `|` characters) — expected: no output, exit code 1:

```
grep -nE '(pytest|safety).*\|\| true' .github/workflows/CI.yml
```

Note: if `cargo test --lib` recompiles from scratch it can take several
minutes; that is normal. Beware a side effect: `cargo test` may re-pin a
dependency in `Cargo.lock` (a routine 1-line update from a newer registry
index). `Cargo.lock` is out of scope — if `git status` shows it modified
after running cargo, revert it with `git checkout -- Cargo.lock`.

## Scope

**In scope** (the only files you may modify):

- `.github/workflows/CI.yml`
- `tests/test_structured_outputs.py`
- `tests/test_taxonomy_tagging.py`

**Out of scope** (do NOT touch, even though they look related):

- `CI.yml` lines 205–217 (ruff/mypy `|| true`) — owned by a later plan that
  first introduces tool configs; making them gating now would break CI.
- `CI.yml` lines 40–47 (cargo-audit step and its ignore flags) — owned by
  plans/001-fix-cargo-audit-ci-break.md and documents a pinned-pyo3 decision.
- `CI.yml:88` tarpaulin `|| true` — coverage *instrumentation* may stay
  non-gating (see Step 3 justification); only pytest/safety gating is in scope.
- `.github/workflows/test-llm-apis.yml` — live-API workflow, by design.
- `tests/test_parallel_inference.py` and its `--ignore` exclusion — live-API
  by design; its own key guards already use `pytest.skip` correctly.
- `Makefile`, any Rust source in `src/`, `tests/model_client_tests.rs`,
  `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `Cargo.lock` (revert
  incidental cargo re-pins — see Commands note), `.env` (never read, edit,
  or delete the `.env` file — it contains credentials).

## Git workflow

- Branch: `advisor/002-ci-verification-gates` (create from `main`).
- Commit per step or per logical unit. Message style: sentence-case
  imperative summary, matching `git log` (e.g. "Refresh cargo-audit ignore
  rationale for the pyo3 CVEs").
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Convert bare-return key guards to real pytest skips

In `tests/test_structured_outputs.py`:

1. Add `import pytest` to the imports at the top of the file (after
   `import polars as pl` is fine).
2. Replace BOTH occurrences of

   ```python
       if not os.getenv("GROQ_API_KEY"):
           print("Skipping test: GROQ_API_KEY not set")
           return
   ```

   (lines 24–26 and 84–86) with:

   ```python
       if not os.getenv("GROQ_API_KEY"):
           pytest.skip("GROQ_API_KEY not set")
   ```

In `tests/test_taxonomy_tagging.py`:

1. Add `import pytest` to the imports at the top of the file.
2. Replace ALL THREE occurrences of

   ```python
       if not os.getenv("ANTHROPIC_API_KEY"):
           print("Skipping test: ANTHROPIC_API_KEY not set")
           return
   ```

   (lines 13–15, 100–102, 177–179) with:

   ```python
       if not os.getenv("ANTHROPIC_API_KEY"):
           pytest.skip("ANTHROPIC_API_KEY not set")
   ```

Keep the `# Check if API key is available` comments if present. Do not
change anything else in these files (both have `if __name__ == "__main__":`
blocks that call the test functions directly; leave them as-is — under a
keyless direct `python tests/test_....py` run they will now raise
`Skipped`, which is acceptable; pytest is the supported runner).

**Verify**:
`GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests/test_structured_outputs.py tests/test_taxonomy_tagging.py -v`
→ output ends with `5 skipped` (and 0 passed, 0 failed); each test line shows
`SKIPPED`.

**Verify**: `grep -rn 'print("Skipping' tests/` → no output.

### Step 2: Add a gating `cargo test --lib` step to the `linux_tests` job

In `.github/workflows/CI.yml`, in the `linux_tests` job, insert a new step
between the `- name: Build and install package` step (currently ends at
line 143) and the `- name: Debug Package Installation` step (currently
starts at line 145):

```yaml
      # Run Rust unit tests (src/ inline #[test] modules). --lib deliberately
      # excludes tests/model_client_tests.rs, which needs live API keys and
      # runs in test-llm-apis.yml instead.
      - name: Run Rust unit tests
        run: cargo test --lib
```

Notes:
- Match the existing 6-space step indentation exactly.
- No `|| true`, no `continue-on-error` — this step must gate.
- The global `RUSTFLAGS: "-Dwarnings"` (line 23) applies; that is intended
  and pre-verified: `RUSTFLAGS="-Dwarnings" cargo test --lib` passes locally
  at `afc78da` (15 passed, exit 0). Do not add an `env:` override.
- Do NOT use bare `cargo test` (it would compile `tests/model_client_tests.rs`,
  whose tests require live keys).

**Verify**: `cargo test --lib` locally → `test result: ok. 15 passed; 0 failed`.

**Verify**: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('yaml ok')"` → `yaml ok`.

**Verify**: `grep -n "cargo test --lib" .github/workflows/CI.yml` → exactly
one match, inside the `linux_tests` job.

### Step 3: Make the coverage job's pytest gating

Decision (pre-made — do not re-litigate): keep the coverage job's pytest run
and remove its `|| true`, rather than deleting the "duplicate" run.
Justification: this run is not a pure duplicate of `linux_tests` — it is the
only thing that produces `coverage/python-coverage.xml` (via the `--cov`
flags), so deleting it would silently remove Python coverage reporting.
Making it gating also stops a failing suite from uploading misleading
coverage numbers to Codecov. The tarpaulin line keeps its `|| true`:
tarpaulin is coverage instrumentation (historically flaky under
`-Dwarnings`/nightly changes), and real Rust test gating is now provided by
Step 2's `cargo test --lib`.

In `.github/workflows/CI.yml`, change line 94 from:

```yaml
          pytest tests --cov=polar_llama --cov-report=xml:coverage/python-coverage.xml --cov-report=term -m "not local_gpu" --ignore=tests/test_parallel_inference.py || true
```

to (only the trailing `|| true` removed):

```yaml
          pytest tests --cov=polar_llama --cov-report=xml:coverage/python-coverage.xml --cov-report=term -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

Do NOT touch line 88 (tarpaulin) or lines 205–217 (ruff/mypy).

**Verify**: `grep -nE 'pytest.*\|\| true' .github/workflows/CI.yml` → no output.

**Verify**: the equivalent local run passes keyless:
`GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py -q`
→ exit 0, `0 failed` (skips are fine and expected).

### Step 4: Replace `safety check || true` with a gating pip-audit scan

In `.github/workflows/CI.yml`, replace the whole step at lines 49–55:

```yaml
      - name: Check for Python dependency vulnerabilities
        run: |
          python -m pip install --upgrade pip
          pip install safety
          pip install -r requirements.txt
          # Run safety check (allow to fail for now, but show results)
          safety check || true
```

with:

```yaml
      - name: Check for Python dependency vulnerabilities
        # Gating scan. Accepted advisories, if any ever become necessary, are
        # ignored explicitly with `--ignore-vuln <ID>` plus a dated comment
        # explaining why — never with `|| true`.
        run: |
          python -m pip install --upgrade pip
          pip install pip-audit
          pip-audit -r requirements.txt
```

Notes:
- `pip install -r requirements.txt` is intentionally dropped from this step:
  `pip-audit -r` resolves and audits the requirements in isolation; the
  project deps are not needed in the job environment (nothing else in the
  `security-audit` job uses them).
- The ignore mechanism is `--ignore-vuln <ID>` appended to the `pip-audit`
  line — mirror the documented style of the cargo-audit step above it
  (comment with rationale + revisit condition). Do not add any ignores now;
  none are needed (verified 2026-07-07: clean on a Python 3.12 resolution).
- The `security-audit` job uses `python-version: "3.10"` (CI.yml:32). If CI
  later fails here because the 3.10 resolver picks an older, vulnerable
  version of a dependency (available fix requires a newer Python), the
  correct fix is to bump this job's `python-version` to `"3.12"` — that is
  allowed within this plan's scope. Do NOT pin or edit `requirements.txt`.

**Verify** (local dry run; requires `python3.12` on PATH — it exists on this
machine at `~/.local/bin/python3.12`; `python3.11` also works):

```
S=$(mktemp -d)
python3.12 -m venv "$S/venv"
"$S/venv/bin/pip" install -q --upgrade pip pip-audit
"$S/venv/bin/pip-audit" -r requirements.txt; echo "exit=$?"
```

→ last lines: `No known vulnerabilities found` and `exit=0`.

**Verify**: `grep -nE 'safety' .github/workflows/CI.yml` → no output.

**Verify**: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('yaml ok')"` → `yaml ok`.

### Step 5: Confirm no collateral edits

**Verify**: `git status --porcelain` → only the three in-scope files modified
(plus `plans/README.md` if you update the index). If `Cargo.lock` shows as
modified (side effect of running `cargo test`), revert it:
`git checkout -- Cargo.lock`.

**Verify**: `git diff .github/workflows/CI.yml | grep -E '^[-+].*(ruff|mypy|cargo audit|RUSTSEC|tarpaulin)'`
→ no output (the ruff/mypy block, the cargo-audit step, and the tarpaulin
line are untouched).

## Test plan

No new test files. The behavioral changes are verified by:

- Keyless skip behavior (the regression this plan fixes):
  `GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests/test_structured_outputs.py tests/test_taxonomy_tagging.py -v`
  → `5 skipped`, 0 passed, 0 failed.
- Keyed behavior unchanged: not verified locally (would hit live APIs and
  spend money); the tests' bodies are untouched, only the guard mechanism
  changed. Structural pattern reference: `tests/test_parallel_inference.py:95`
  uses `pytest.skip(...)` the same way.
- Full CI-safe suite still green:
  `GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py -q`
  → exit 0.
- Rust unit tests: `cargo test --lib` → 15 passed (count may grow if other
  plans land first; 0 failed is the requirement).

## Done criteria

Machine-checkable. ALL must hold (run from repo root):

- [ ] `grep -c "cargo test --lib" .github/workflows/CI.yml` → `1`
- [ ] `grep -nE '(pytest|safety).*\|\| true' .github/workflows/CI.yml` → no output (exit 1)
- [ ] `grep -n "safety" .github/workflows/CI.yml` → no output (exit 1)
- [ ] `grep -rn 'print("Skipping' tests/` → no output (exit 1)
- [ ] `GROQ_API_KEY="" ANTHROPIC_API_KEY="" .venv/bin/python -m pytest tests/test_structured_outputs.py tests/test_taxonomy_tagging.py -v` → `5 skipped`, 0 passed, 0 failed
- [ ] `cargo test --lib` → exit 0, `0 failed`
- [ ] `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('yaml ok')"` → `yaml ok`
- [ ] `git diff afc78da..HEAD -- .github/workflows/CI.yml | grep -cE '^[-+].*(ruff|mypy)'` → `0`
- [ ] `git status --porcelain` shows no modified files outside the in-scope list (plans/README.md excepted)
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains the index)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows any in-scope file changed since `afc78da` and the
  "Current state" excerpts no longer match the live code.
- `cargo test --lib` fails locally for reasons unrelated to API keys. The
  failures are exactly what this plan exists to surface — report them; do
  NOT "fix" Rust source (all of `src/` is out of scope).
- The keyless pytest run after Step 1 shows anything other than exactly
  5 skipped / 0 passed / 0 failed for the two changed files (e.g. an
  ImportError on `pytest`, or a test that starts executing an API call —
  that would mean the guard replacement went wrong or your environment has
  keys exported in the shell; check `env | grep -E 'GROQ|ANTHROPIC'`).
- The local pip-audit dry run in Step 4 reports any vulnerability on a
  Python 3.12 resolution — the dependency landscape has drifted since
  2026-07-07; report the advisory IDs rather than silently adding
  `--ignore-vuln` flags.
- Plan 001 (plans/001-fix-cargo-audit-ci-break.md) has not landed — this
  plan depends on it: the pip-audit change lives in the same `security-audit`
  job as the cargo-audit step 001 repairs, and a broken preceding step would
  mask this plan's result. (Locally you can still complete every step; the
  dependency only matters for interpreting CI results.)
- Any fix appears to require touching `requirements.txt`, `pyproject.toml`,
  `src/`, or `tests/test_parallel_inference.py`.
- You find yourself needing a live API key to satisfy any verification —
  none of this plan's done criteria may require one.

## Maintenance notes

- **Redundant compile in the matrix**: `cargo test --lib` now runs in all 4
  `linux_tests` matrix legs (py3.9–3.12). The Rust lib is Python-version-
  independent, so 3 runs are redundant. Acceptable cost for now; if CI time
  becomes a concern, move the step to a dedicated single-leg job (or into
  the `coverage` job before tarpaulin).
- **tarpaulin still non-gating** (`CI.yml:88 || true`): deliberate — it is
  instrumentation, not the test gate. If tarpaulin stabilizes, consider
  removing that `|| true` too and adding `fail_ci_if_error: true` to the
  Codecov upload.
- **pip-audit resolution is Python-version-sensitive**: with unpinned
  requirements, the audited versions are whatever the job's Python resolves.
  If the `security-audit` job's Python (3.10 today) ages out of a
  dependency's support window, the scan may flag vulnerabilities whose only
  fix needs a newer interpreter — bump the job's `python-version`, don't
  ignore the advisory.
- **pyo3 `extension-module` vs `cargo test`**: today `cargo test --lib`
  links fine on Linux with pyo3 0.27 + `extension-module` + `abi3-py39`
  (empirically verified via a minimal repro on `rust:1-bookworm`). If a
  future pyo3 upgrade reintroduces the classic link failure
  (`undefined reference to PyExc_*` in the test binary), the standard fix is
  the pyo3 FAQ feature-gate: make `pyo3/extension-module` an optional crate
  feature enabled by default and run tests with `--no-default-features` —
  not removing the CI step.
- **Reviewer focus**: (1) confirm the new CI step says `cargo test --lib`,
  not bare `cargo test` — the bare form compiles `tests/model_client_tests.rs`
  and will fail keyless CI; (2) confirm the ruff/mypy `|| true` block
  (CI.yml:205–217) and the cargo-audit ignore flags are untouched;
  (3) confirm both test files gained `import pytest`.
- **Deferred follow-ups**: making ruff/mypy gating (needs tool configs
  first — separate plan); the direct-run `if __name__ == "__main__":` blocks
  in the two test files now raise `Skipped` when keyless instead of printing
  — harmless, cleanup deferred; the two changed test files contain no
  assertions (bodies only print results and swallow exceptions in
  `try/except`), so even WITH keys they can only fail on an exception outside
  the `try` blocks — adding real assertions is live-API-test work that
  belongs with the test-llm-apis.yml scope, not this plan; the five untracked
  `test_*.py` files at the repo root (e.g. `test_cache_messages.py`) are
  outside `tests/` and outside CI — not addressed here.
