# Plan 013: Add CLAUDE.md agent/contributor contract and fix the Makefile dev loop

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- Makefile CLAUDE.md CONTRIBUTING.md README.md pyproject.toml requirements.txt run.py .cargo/audit.toml Cargo.toml`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. (Changes to `pyproject.toml`,
> `requirements.txt`, or `run.py` are NOT automatically a STOP — they likely
> mean plans/011 or plans/020 landed; see Step 1 for how to adapt.)

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/011-consolidate-python-tooling.md (soft — only the install/lint command wording; this plan is executable before 011 lands)
- **Category**: dx
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

The repo has no CLAUDE.md, AGENTS.md, or CONTRIBUTING.md, so every AI agent or
new contributor rediscovers the build (maturin), the test tiers (CI-safe vs
Apple-GPU vs live-API), and — worse — periodically tries to "fix" deliberate,
documented tradeoffs like the pyo3 < 0.29 pin. The Makefile also has a broken
dev loop: on a fresh clone `make test` runs pytest before `maturin develop`
ever built the Rust extension (ImportError), and it runs the Apple-GPU-only
tests that CI deliberately excludes. This plan writes the contract down once
and makes the Makefile targets match how the project is actually built and
tested.

## Current state

Verified against the working tree at `afc78da` (2026-07-06):

- `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md` — do not exist
  (`ls CLAUDE.md AGENTS.md CONTRIBUTING.md` → "No such file or directory" for all three).
- `Makefile` — 40 lines, full current content:

```make
SHELL=/bin/bash

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv
	cargo fmt --all && cargo clippy --all-features
	.venv/bin/python -m ruff check . --fix || true
	if [ -d "polar_llama" ]; then \
		.venv/bin/python -m ruff format polar_llama || true; \
	fi
	if [ -d "tests" ]; then \
		.venv/bin/python -m ruff format tests || true; \
	fi
	if [ -d "polar_llama" ] && [ -d "tests" ]; then \
		.venv/bin/mypy --ignore-missing-imports polar_llama tests || true; \
	elif [ -d "polar_llama" ]; then \
		.venv/bin/mypy --ignore-missing-imports polar_llama || true; \
	elif [ -d "tests" ]; then \
		.venv/bin/mypy --ignore-missing-imports tests || true; \
	fi

test: .venv
	.venv/bin/python -m pytest tests

run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py
```

  Defects: `test:` depends only on `.venv`, not `install`, so the compiled
  extension may not exist; `test` runs ALL of `tests/` including `local_gpu`
  (Apple-GPU) tests; there are no `test-ci`, `build`, `lint`, `typecheck`, or
  `clean` targets; no `.PHONY` declarations at all.
- `requirements.txt` — 8 lines: `polars`, `maturin`, `ruff`, `pytest`, `mypy`,
  `pandas`, `pyarrow`, `matplotlib`. (So `ruff` and `mypy` are installed into
  `.venv` by the `.venv` target — the new `lint`/`typecheck` targets can rely
  on them.)
- `run.py` — exists at repo root (a live-API demo script; plans/020 proposes
  deleting it, not landed as of `afc78da`).
- `pyproject.toml:64-68` — pytest markers:

```toml
[tool.pytest.ini_options]
markers = [
    "local_gpu: requires Apple GPU + mlx (skipped in CI)",
    "local: local backend logic tests (CI-safe)",
]
```

- `pyproject.toml:49-55` — the mlx-lm/transformers tradeoff (tradeoff #3 below):

```toml
local = [
    "mlx>=0.31; python_version >= '3.10'",
    # mlx-lm 0.31.3 pulls its own required transformers (>=5); do NOT cap it
    # (a <5 pin makes this extra unsatisfiable). A smoke install of 0.31.3 ran
    # batch_generate successfully against transformers>=5.
    "mlx-lm>=0.31; python_version >= '3.10'",
]
```

- `pyproject.toml` has NO `[tool.ruff]` or `[tool.mypy]` sections today
  (plans/011 adds Python tooling config).
- `.cargo/audit.toml:11-30` — the pyo3 pin tradeoff (tradeoff #1). Key lines:

```
    # ---- pyo3 < 0.29 advisories: blocked by the pyo3-polars pin ----
    ...
    # pyo3-polars provides the `#[polars_expr]` plugin machinery this crate is
    # built on. As of 2026-07-04 its latest release (0.27.0) still depends on
    # pyo3 0.28 ... NO published pyo3-polars release supports pyo3 0.29 ...
    "RUSTSEC-2026-0176",  # OOB read in nth/nth_back for PyList/PyTuple iterators (fixed in pyo3 >= 0.29)
    "RUSTSEC-2026-0177",  # Missing Sync bound on PyCFunction::new_closure closures (fixed in pyo3 >= 0.29)
```

- `Cargo.toml:18-20` and `Cargo.toml:27-34` — the TLS/crypto tradeoff
  (tradeoff #2):

```toml
# rustls with the OS certificate store, so TLS verification stays on even
# behind corporate proxies with custom CAs
reqwest = { version = "0.12", features = ["json", "rustls-tls-native-roots"], default-features = false }
...
# Use the ring crypto provider (not aws-lc-rs) so the Windows wheel builds:
# aws-lc-sys fails to compile under MSVC ("C atomics require C11 or later").
```

- `.github/workflows/CI.yml` — `RUSTFLAGS: "-Dwarnings"` is set globally
  (line 23); the CI-safe pytest invocation (line 197) is
  `python -m pytest tests -v --tb=short -m "not local_gpu" --ignore=tests/test_parallel_inference.py`;
  Rust lint (line 204) is `cargo fmt --all && cargo clippy --all-features`;
  the `release` job runs `if: "startsWith(github.ref, 'refs/tags/')"` and
  publishes to PyPI via trusted publishing (`id-token: write`,
  `maturin-action` `command: upload`).
- `.github/workflows/test-llm-apis.yml` — manually triggered live-API test
  workflow (needs real provider keys).
- `tests/model_client_tests.rs` — Rust integration tests that hit live
  provider APIs (need keys); `cargo test --lib` runs only the keyless unit
  tests inside `src/`.
- `.gitignore:4` — `.env` is gitignored; `.env.example` documents which key
  names exist. A real `.env` exists locally — never read or copy its values.
- `README.md:569-571`:

```
#### Contributing

We welcome contributions to Polar Llama! If you're interested in improving the library or adding new features, please feel free to fork the repository and submit a pull request.
```

- Commit message convention (from `git log`): sentence-case imperative
  summaries, e.g. `Add collapsed-prefill and batched quantized-KV to the local backend`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Dry-run a make target | `make -n <target>` | prints recipe, exit 0 |
| Venv + deps + extension build | `make install` | exit 0 (first run takes minutes: Rust compile) |
| CI-safe Python tests | `.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0, no failures (skips are OK) |
| Rust unit tests (keyless) | `cargo test --lib` | exit 0 |
| Rust format check | `cargo fmt --all -- --check` | exit 0, no output |
| Line count | `grep -c "" CLAUDE.md` | a number < 150 |

None of the done criteria require API keys.

## Scope

**In scope** (the only files you may modify/create):
- `CLAUDE.md` (create)
- `CONTRIBUTING.md` (create)
- `Makefile` (edit)
- `README.md` (ONE line added in the Contributing section — Step 5 only; skip
  entirely if the section moved/changed)

**Out of scope** (do NOT touch, even though they look related):
- `.github/workflows/*` — CI is owned by other plans; do not "align" it.
- `pyproject.toml`, `requirements.txt` — Python tooling consolidation is
  plans/011-consolidate-python-tooling.md.
- `run.py` — deletion is plans/020's job; this plan only conditions the
  Makefile `run`/`run-release` targets on its existence (Step 3).
- The `pre-commit:` Makefile target's `|| true` behavior — plans/011 fixes it;
  leave the `pre-commit` recipe byte-for-byte unchanged.
- Any Rust or Python source file.
- `plans/README.md` index content beyond your own status row.

## Git workflow

- Branch: `advisor/013-claude-md-and-makefile` (branched from `main`)
- Commit style: sentence-case imperative summary, e.g.
  `Add CLAUDE.md contributor contract and Makefile dev-loop targets`
- One commit for the whole plan is fine (it is small), or one per step.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Baseline check

From the repo root, confirm the CI-safe suite passes BEFORE any change, so a
later failure is attributable:

```
make install
.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

**Verify**: both exit 0 (pytest may report skipped tests — that is expected).
If pytest fails here, the failure is pre-existing: record the failing test
names, do NOT try to fix them, and treat "make test-ci exits 0" in the Done
criteria as "fails only with the same pre-existing failures".

### Step 1: Determine which dependent plans have landed

Two soft dependencies change wording only:

1. **plans/011-consolidate-python-tooling.md** (Python tooling): check
   `git log --oneline -20` and `plans/README.md` for it being DONE, and check
   whether `pyproject.toml` now contains a `[tool.ruff]` section. If landed,
   open plans/011 and use ITS canonical install/lint commands in `CLAUDE.md`
   (Step 2) and keep the Makefile lines it rewrote. If NOT landed (the state
   at planning time), use the current `requirements.txt`-based commands
   exactly as written in Step 2 below.
2. **plans/020** (deletes `run.py`): check whether `run.py` exists. If it
   still exists (state at planning time), leave the `run`/`run-release`
   Makefile targets unchanged. If it was deleted, remove both targets in
   Step 3 and remove them from the `.PHONY` line.

**Verify**: `ls run.py; grep -c "tool.ruff" pyproject.toml || true` → record
both results; they select the branches in Steps 2–3.

### Step 2: Create `CLAUDE.md`

Create `CLAUDE.md` at the repo root with exactly the content below (adapt only
the Setup/Lint commands per Step 1 if plans/011 landed; keep it under 120
lines, imperative, no marketing):

````markdown
# CLAUDE.md

Operating guide for AI agents and human contributors. Follow it exactly.

## What this is

polar-llama: a Polars plugin for parallel LLM inference. Hybrid Rust+Python:

- `src/` — Rust core (pyo3 0.27 + pyo3-polars 0.26, tokio, reqwest).
  Provider clients in `src/model_client/` (openai, anthropic, gemini, groq,
  bedrock). `src/expressions.rs` registers the Polars expressions;
  `src/cache.rs` prompt caching; `src/cost.rs` token costs; `src/ann.rs`
  HNSW similarity; `src/mcp.rs` MCP client.
- `polar_llama/` — Python API layer (`__init__.py` is the main API;
  `tools.py`, `optimize.py`, `local/` = on-device MLX backend).
- Built with maturin; published to PyPI as `polar-llama`.

## Setup

```
make install          # venv + deps + maturin develop (debug build)
make install-release  # release build (slower compile, faster runtime)
```

Manual equivalent: `python3 -m venv .venv`,
`.venv/bin/pip install -r requirements.txt`, then
`source .venv/bin/activate && maturin develop`.
Re-run `maturin develop` (or `make install`) after ANY Rust change — the
Python tests import the compiled extension.

## Tests

- CI-safe Python suite (no API keys, no GPU) — this is the gate CI runs:
  `pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
  (or `make test-ci`).
- Markers (defined in `pyproject.toml`): `local_gpu` = requires Apple GPU +
  mlx, skipped in CI; `local` = local-backend logic tests, CI-safe.
- Rust unit tests (keyless): `cargo test --lib`.
- Live-API tests: `tests/model_client_tests.rs` (Rust) and
  `tests/test_parallel_inference.py` (Python) need real provider keys. Never
  make them a merge gate; they run via the manual workflow
  `.github/workflows/test-llm-apis.yml`.

## Lint / format

- Rust: format with `cargo fmt --all`; CI enforces
  `cargo fmt --all -- --check` and `cargo clippy --all-features` with
  `RUSTFLAGS=-Dwarnings` (all warnings are errors). Run `make lint` locally.
- Python: `ruff` (`make lint`) and `mypy` (`make typecheck`).
- Errors in Rust flow through `ModelClientError` and `PolarsResult` — see
  `src/model_client/mod.rs`. Match that pattern; do not `unwrap()` in
  request paths.

## Environment and secrets

- Provider API keys live in `.env` (gitignored). Key names are documented in
  `.env.example`. Never commit keys; never print `.env` contents.
- Tests requiring keys skip when the keys are absent.

## Documented tradeoffs — do NOT "fix" these

1. **pyo3 is pinned < 0.29** and two RUSTSEC advisories are accepted.
   Blocked upstream: no released pyo3-polars supports pyo3 0.29. Full
   rationale in `.cargo/audit.toml`. Do not bump pyo3 or drop the ignores.
2. **TLS/crypto choices in `Cargo.toml`**: reqwest uses
   `rustls-tls-native-roots` (OS cert store, keeps TLS verification on
   behind corporate proxies); the AWS SDK is forced onto the `ring` crypto
   provider because aws-lc-sys does not build under MSVC (Windows wheels).
   See the comments in `Cargo.toml`. Do not switch these features.
3. **`[local]` extra transformers floor**: mlx-lm requires transformers >= 5;
   do NOT add a `<5` cap to the extra (it becomes unsatisfiable). See the
   comment in `pyproject.toml` `[project.optional-dependencies]`.

## Release

Releases are tag-driven: push a version tag and `.github/workflows/CI.yml`
builds linux/windows/macos wheels + sdist and publishes to PyPI via trusted
publishing (no tokens in the repo). Before tagging: bump `version` in
`Cargo.toml` and update `CHANGELOG.md`.
````

**Verify**:
- `test -f CLAUDE.md && grep -c "" CLAUDE.md` → a number under 120
- `grep -q "audit.toml" CLAUDE.md && grep -q "ring" CLAUDE.md && grep -qi "transformers" CLAUDE.md && echo OK` → `OK`

### Step 3: Fix the Makefile

Edit `Makefile`. Keep `SHELL=/bin/bash`, the `.venv`, `install`,
`install-release`, and `pre-commit` recipes byte-for-byte unchanged. Make
these changes:

1. Add a `.PHONY` line after `SHELL=/bin/bash`:
   `.PHONY: install install-release build pre-commit test test-ci lint typecheck clean run run-release`
   (drop `run run-release` from it only if Step 1 found `run.py` deleted).
2. Change `test:` to depend on `install` (recipe unchanged):

```make
test: install
	.venv/bin/python -m pytest tests
```

3. Add these targets (tab-indented recipes — Make requires tabs):

```make
test-ci: install
	.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py

build: install

lint: .venv
	cargo fmt --all -- --check
	RUSTFLAGS="-Dwarnings" cargo clippy --all-features
	.venv/bin/python -m ruff check .

typecheck: .venv
	.venv/bin/mypy --ignore-missing-imports polar_llama tests

clean:
	cargo clean
	rm -rf .venv target/wheels
```

4. `run`/`run-release`: per Step 1 — if `run.py` still exists, leave both
   targets exactly as they are; if plans/020 deleted `run.py`, delete both
   targets. Note which branch you took in your final report.

Do NOT change `test` to the CI-safe filter — `make test` intentionally runs
the full local suite (including `local_gpu` on Apple hardware); `make test-ci`
is the CI-equivalent gate.

**Verify**:
- `make -n test` → prints the `install` recipe (containing `maturin develop`)
  BEFORE the `pytest tests` line; exit 0
- `make -n test-ci` → output contains
  `pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
- `make -n build && make -n lint && make -n typecheck && make -n clean` → all exit 0

### Step 4: Create `CONTRIBUTING.md`

Create `CONTRIBUTING.md` at the repo root with exactly:

```markdown
# Contributing to polar-llama

Thanks for contributing. The full developer contract — project layout, setup,
build, test tiers, lint, and the documented tradeoffs you must not change —
lives in [CLAUDE.md](CLAUDE.md). It applies to human contributors and AI
agents alike; read it first.

Quick start: `make install`, then `make test-ci`.

## Pull requests

- Branch from `main`.
- Run `make lint`, `make typecheck`, and `make test-ci` before opening a PR.
  CI enforces `cargo fmt` / `cargo clippy` with warnings-as-errors.
- Write sentence-case imperative commit summaries (e.g. "Add collapsed-prefill
  and batched quantized-KV to the local backend").
- Update `CHANGELOG.md` for user-visible changes.
- Never commit API keys; `.env` is gitignored — keep it that way.
```

**Verify**: `grep -q "CLAUDE.md" CONTRIBUTING.md && echo OK` → `OK`

### Step 5: Link CONTRIBUTING.md from the README (one line, optional)

Open `README.md` and find the `#### Contributing` section (around line 569).
If the section still reads as quoted in "Current state", append one sentence
to its paragraph:
`See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the PR checklist.`

If the section has been rewritten or moved (plans/012 owns the README), SKIP
this step entirely and note the skip in your report — do not restructure the
README.

**Verify**: `grep -c "CONTRIBUTING.md" README.md` → `1` (or step skipped)

## Test plan

This plan adds no runtime code, so no new automated tests. Regression proof:

- `make test-ci` → exit 0 (same result as the Step 0 baseline; skipped tests
  OK; any failure must already have been present in Step 0).
- `cargo test --lib` → exit 0 (no Rust sources touched; this proves the
  Makefile edits did not break the cargo workspace, e.g. via a stray file).
- The `make -n` dry-run checks in Steps 3 are the primary verification.

## Done criteria

Machine-checkable. ALL must hold (none require API keys):

- [ ] `test -f CLAUDE.md && test -f CONTRIBUTING.md` → exit 0
- [ ] `grep -c "" CLAUDE.md` → less than 150 (target: under 120)
- [ ] CLAUDE.md names all three documented tradeoffs:
      `grep -q "audit.toml" CLAUDE.md && grep -q "ring" CLAUDE.md && grep -qi "transformers" CLAUDE.md` → exit 0
- [ ] `make -n test` output contains `maturin develop` on an earlier line than `pytest tests`
- [ ] `make -n test-ci` output contains `pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
- [ ] `make -n build && make -n lint && make -n typecheck && make -n clean` → exit 0
- [ ] `make test-ci` → exit 0 (or fails only with failures recorded as pre-existing in Step 0)
- [ ] `git status --porcelain` shows changes only to: `CLAUDE.md`, `CONTRIBUTING.md`, `Makefile`, optionally `README.md` (Step 5), and your `plans/README.md` status row
- [ ] The `pre-commit`, `.venv`, `install`, `install-release` recipes are byte-identical to "Current state" (`git diff Makefile` touches only `.PHONY`, `test`, and the new/removed targets)

## STOP conditions

Stop and report back (do not improvise) if:

- The current `Makefile` does not match the 40-line excerpt in "Current
  state" in the `.venv`/`install`/`test`/`pre-commit` recipes (another plan
  rewrote it — reconcile with the advisor instead of merging by hand).
- plans/011 has landed but its plan file's canonical install commands are
  ambiguous or contradict what is actually in `pyproject.toml` — do not guess
  which install path to document in CLAUDE.md.
- `make install` fails in Step 0 (the dev loop is broken at baseline; fixing
  the build is out of scope).
- `.cargo/audit.toml`, the `Cargo.toml` TLS comments, or the `pyproject.toml`
  transformers comment no longer exist where cited — the tradeoff list in
  CLAUDE.md would be wrong; report instead of paraphrasing from memory.
- Any step's verification fails twice after a reasonable fix attempt.
- You find yourself needing to edit `.github/workflows/*`, `pyproject.toml`,
  `requirements.txt`, or any file under `src/` or `polar_llama/`.

## Maintenance notes

- CLAUDE.md duplicates facts that live elsewhere (CI commands, pytest
  markers, release flow). Whoever changes `.github/workflows/CI.yml`, the
  pytest markers in `pyproject.toml`, or the release process must update
  CLAUDE.md in the same PR — reviewers should ask for this.
- When plans/011 lands (if after this plan), it must update the Setup and
  Lint sections of CLAUDE.md and the `.venv`/`lint`/`typecheck` Makefile
  targets to the consolidated tooling; when plans/020 lands, it must remove
  the `run`/`run-release` targets and their `.PHONY` entries.
- When pola-rs ships a pyo3-0.29-compatible pyo3-polars, tradeoff #1 in
  CLAUDE.md and the ignores in `.cargo/audit.toml` should be removed together.
- Reviewer focus: (a) the tradeoffs section must match the in-repo comments,
  not paraphrase them loosely; (b) `make test` must still run the FULL suite
  (that is intentional), with `make test-ci` as the CI-equivalent; (c) no
  drive-by edits to the `pre-commit` target.
- Deferred: making `make lint` pass clean (ruff has no config yet and may
  flag existing code — plans/011 owns lint cleanliness); README overhaul
  (plans/012).
