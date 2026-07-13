# Plan 015: Cache Rust builds in CI and install cargo tools as prebuilt binaries

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- .github/workflows/CI.yml`
> If the file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. (Note: plan 001, a dependency of
> this plan, does NOT touch `.github/workflows/CI.yml` — its scope is
> `Cargo.lock` and `.cargo/audit.toml` — so a clean drift check is expected
> even after 001 lands.)

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-fix-cargo-audit-ci-break.md
- **Category**: dx
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

Two CI jobs compile their tooling from source on every single run: the
`security-audit` job runs `cargo install cargo-audit` and the `coverage` job
runs `cargo install cargo-tarpaulin` — each is several minutes of pure
compile time before the job does any real work. Neither job caches Rust
build artifacts at all, so `coverage` additionally does a cold
`maturin develop --release` (this crate pulls in polars — a very large
dependency tree) plus a cold tarpaulin instrumented build. Switching the
tool installs to prebuilt binaries (`taiki-e/install-action`) and adding
`Swatinem/rust-cache` to the three Rust-building jobs cuts many minutes from
every push/PR cycle with no behavior change to what CI checks.

Why this depends on plan 001: at the planning commit, CI is red because
`cargo audit` fails on RUSTSEC-2026-0204 (crossbeam-epoch). This plan's edits
are safe to make regardless, but the post-merge CI run that confirms the
speedup can only be green after 001's lockfile fix lands. Execute 001 first.

## Current state

Relevant file (the ONLY file this plan modifies):

- `.github/workflows/CI.yml` — the main CI workflow. Jobs: `security-audit`,
  `coverage`, `linux_tests` (matrix py3.9–3.12), `linux`/`windows`/`macos`
  (wheel builds via `PyO3/maturin-action` with `sccache: "true"`), `sdist`,
  `release`.

Verified facts as of `afc78da` (2026-07-06):

1. `grep -c "cargo install" .github/workflows/CI.yml` → `2`
2. `grep -c "RUSTC_WRAPPER" .github/workflows/CI.yml` → `0` — meaning the
   existing `mozilla-actions/sccache-action@v0.0.3` step in `linux_tests`
   installs sccache but **nothing ever invokes it** (sccache only engages
   when `RUSTC_WRAPPER=sccache` is set). It is dead weight today.
3. No job has `Swatinem/rust-cache`.

`.github/workflows/CI.yml:26-38` — the `security-audit` job's opening steps:

```yaml
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.10"

      - name: Set up Rust
        run: rustup show

      - name: Install cargo-audit
        run: cargo install cargo-audit
```

`.github/workflows/CI.yml:57-83` — the `coverage` job's opening steps
(abridged in the middle; the pip-install block at lines 71–78 is unchanged
by this plan):

```yaml
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.10"

      - name: Set up Rust
        run: rustup show

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
```
```yaml
      - name: Build package
        run: |
          source .venv/bin/activate
          maturin develop --release
```

`.github/workflows/CI.yml:111-125` — the `linux_tests` job's opening steps
(note the existing sccache-action line at 125):

```yaml
  linux_tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [x86_64]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}

      - name: Set up Rust
        run: rustup show
      - uses: mozilla-actions/sccache-action@v0.0.3
```

Global workflow env (`.github/workflows/CI.yml:22-23`) — do not change:

```yaml
env:
  RUSTFLAGS: "-Dwarnings"
```

### Pinned versions to use (resolved and verified at planning time, 2026-07-07)

Plan 016 (`plans/016-sha-pin-actions.md`) will SHA-pin every EXISTING action
reference. To keep 016's work from being redone, every action reference this
plan **adds or replaces** must already use the full 40-hex commit SHA with a
trailing version comment — 016's exact convention is
`owner/repo@<40-hex-sha> # <version-tag>` (the trailing comment is
load-bearing: Dependabot parses and updates it alongside the SHA; the repo's
`dependabot.yml` already has a `github-actions` entry).

These SHAs were resolved at planning time via
`gh api repos/{owner}/{repo}/git/ref/tags/{tag}` (annotated tags dereferenced
to their commit with `gh api repos/{owner}/{repo}/git/tags/{tag-sha}`):

| Action | Version tag | Commit SHA |
|--------|-------------|------------|
| `taiki-e/install-action` | `v2.82.10` | `50414676f9f5d50a65992c6dd2ed02641263226c` |
| `Swatinem/rust-cache` | `v2.9.1` | `c19371144df3bb44fab255c43d04cbc2ab54d1c4` |
| `mozilla-actions/sccache-action` | `v0.0.10` | `9e7fa8a12102821edf02ca5dbea1acd0f89a2696` |

Tool versions to pin in `taiki-e/install-action` (both have manifests in
install-action's supported-tools list, verified at planning time, so they
install as prebuilt binaries in seconds):

- `cargo-audit@0.22.2` (matches the version plan 001 verified locally)
- `cargo-tarpaulin@0.37.0` (latest release at planning time)

You MAY re-verify the SHAs at execution time (commands in "Commands you will
need"); if a re-resolved SHA differs from the table, that is a STOP condition
(a tag was moved — a supply-chain red flag).

### Design decisions this plan implements (do not re-litigate)

1. **rust-cache goes into all 3 Rust-building jobs** (`security-audit`,
   `coverage`, `linux_tests`), placed after the `Set up Rust` step and before
   any cargo/maturin invocation. In `security-audit` its benefit is small
   (registry index only — cargo-audit no longer compiles anything after this
   plan) but it is harmless and keeps the jobs uniform.
2. **rust-cache in `linux_tests` deliberately gets NO `key:` input** — the
   crate builds with `abi3-py39` (`Cargo.toml:14`:
   `pyo3 = { version = "0.27", features = ["extension-module", "abi3-py39"] }`),
   so the Rust build is Python-version-independent and all 4 matrix legs can
   share one cache. The first leg to finish saves it; the others log a
   "cache already exists" warning, which is normal.
3. **sccache is activated, not just installed.** The finding's option was
   "keep sccache and add rust-cache — or justify choosing one". Justification
   for what this plan does: sccache in `linux_tests` is currently inert
   (verified fact 2 above), and the old `sccache-action@v0.0.3` installs an
   sccache release that predates GitHub's cache-service v2 (the legacy cache
   service was shut down in 2025; sccache >= 0.8.2 / action >= v0.0.6 is
   required). Activating the old binary would break builds. So this plan
   bumps that ONE existing reference to the SHA-pinned v0.0.10 (016 will then
   skip it as already pinned) and sets `SCCACHE_GHA_ENABLED` +
   `RUSTC_WRAPPER` at job level in `coverage` and `linux_tests`. rust-cache
   and sccache compose: rust-cache restores whole compiled dependency trees
   keyed on `Cargo.lock`; sccache adds per-compilation-unit caching that
   still hits when `Cargo.lock` changes. rust-cache also sets
   `CARGO_INCREMENTAL=0`, which sccache requires.
4. The wheel-build jobs (`linux`, `windows`, `macos`, `sdist`) are NOT
   touched: `maturin-action` already runs its own sccache via
   `sccache: "true"`, and rust-cache's paths don't apply inside
   maturin-action's manylinux Docker container.

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Drift check | `git diff --stat afc78da..HEAD -- .github/workflows/CI.yml` | empty output (no drift) |
| YAML parses | `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"` | prints `ok`, exit 0 |
| No source-compiled tool installs | `grep -c "cargo install" .github/workflows/CI.yml` | prints `0` (grep exits 1 when the count is 0 — that is the success case here) |
| rust-cache in 3 jobs | `grep -c "Swatinem/rust-cache" .github/workflows/CI.yml` | prints `3` |
| Prebuilt installer used twice | `grep -c "taiki-e/install-action" .github/workflows/CI.yml` | prints `2` |
| sccache installed in 2 jobs | `grep -c "mozilla-actions/sccache-action" .github/workflows/CI.yml` | prints `2` |
| sccache actually wired up | `grep -c "RUSTC_WRAPPER" .github/workflows/CI.yml` | prints `2` |
| New refs are SHA-pinned | copy the exact `grep -cE` command from Step 5 (it contains pipe characters that cannot be rendered inside this table) | prints `7` |
| (Optional) re-resolve a tag SHA | `gh api repos/taiki-e/install-action/git/ref/tags/v2.82.10 --jq '.object.sha'` | `50414676f9f5d50a65992c6dd2ed02641263226c` |
| (Optional) deref annotated tag | `gh api repos/Swatinem/rust-cache/git/ref/tags/v2.9.1 --jq '.object.sha'` then `gh api repos/Swatinem/rust-cache/git/tags/<that-sha> --jq '.object.sha'` | first returns a tag-object SHA; second returns `c19371144df3bb44fab255c43d04cbc2ab54d1c4` (same two-step for `mozilla-actions/sccache-action` v0.0.10 → `9e7fa8a12102821edf02ca5dbea1acd0f89a2696`) |

Notes:
- Use `.venv/bin/python` for the YAML check — the repo's existing virtualenv
  has PyYAML installed; the system `python3` on the planning machine did not.
  If `.venv` is missing, create it per the repo README or
  `pip install pyyaml` into any Python you use for the check.
- No Rust build, maturin build, or API keys are needed for any verification
  in this plan. All real caching behavior is verified CI-side after merge
  (see Maintenance notes).

## Scope

**In scope** (the only file you may modify):
- `.github/workflows/CI.yml`

**Out of scope** (do NOT touch, even though they look related):
- `.github/workflows/test-llm-apis.yml` — live-API workflow; separate concern.
- `.github/dependabot.yml` — already has a `github-actions` entry; nothing to add.
- SHA-pinning of the OTHER existing action references (`actions/checkout@v7`,
  `actions/setup-python@v6`, `codecov/codecov-action@v5`,
  `actions/upload-artifact@v5`, `actions/download-artifact@v6`,
  `PyO3/maturin-action@v1`) — that is plan 016's job. The ONLY pre-existing
  reference this plan changes is `mozilla-actions/sccache-action@v0.0.3` in
  `linux_tests` (justified in "Design decisions", item 3).
- Gating semantics — the `|| true` on tarpaulin/pytest/safety steps and any
  `continue-on-error` policy belong to plans/002-ci-verification-gates.md.
- The wheel-build jobs (`linux`, `windows`, `macos`), `sdist`, `release`.
- The global `RUSTFLAGS: "-Dwarnings"` env, the `concurrency` block, and all
  test/pre-commit step contents.

## Git workflow

- Branch: `advisor/015-ci-caching` (create from `main`).
- One commit. Message style is sentence-case imperative, matching `git log`
  (e.g. "Refresh cargo-audit ignore rationale for the pyo3 CVEs"). Suggested:
  `Cache Rust builds in CI and install cargo tools as prebuilt binaries`
- Do NOT push or open a PR unless the operator instructed it.
- The working tree may contain pre-existing untracked files (`.DS_Store`,
  `test_cache_*.py`, `test_optional_llm.py`, `plans/`). Leave them alone.

## Steps

### Step 1: Create the branch

```
git checkout -b advisor/015-ci-caching main
```

**Verify**: `git branch --show-current` → `advisor/015-ci-caching`

### Step 2: `security-audit` — prebuilt cargo-audit + rust-cache

In `.github/workflows/CI.yml`, in the `security-audit` job, replace this
block (currently lines 34–38):

```yaml
      - name: Set up Rust
        run: rustup show

      - name: Install cargo-audit
        run: cargo install cargo-audit
```

with:

```yaml
      - name: Set up Rust
        run: rustup show

      - name: Rust cache
        uses: Swatinem/rust-cache@c19371144df3bb44fab255c43d04cbc2ab54d1c4 # v2.9.1

      - name: Install cargo-audit
        uses: taiki-e/install-action@50414676f9f5d50a65992c6dd2ed02641263226c # v2.82.10
        with:
          tool: cargo-audit@0.22.2
```

Leave the `Run cargo audit` step (and its pyo3 rationale comment block) and
the `Check for Python dependency vulnerabilities` step byte-for-byte
unchanged.

**Verify**:
`.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"` → `ok`,
and `grep -c "cargo install" .github/workflows/CI.yml` → `1` (only the
tarpaulin one remains).

### Step 3: `coverage` — prebuilt tarpaulin, rust-cache, functional sccache

Two edits in the `coverage` job.

**3a.** Add a job-level `env:` block. Replace (currently lines 57–58):

```yaml
  coverage:
    runs-on: ubuntu-latest
```

with:

```yaml
  coverage:
    runs-on: ubuntu-latest
    env:
      SCCACHE_GHA_ENABLED: "true"
      RUSTC_WRAPPER: "sccache"
```

(These job-level vars merge with the workflow-level `RUSTFLAGS`; they do not
replace it. The first cargo invocation in this job happens after the
sccache-action step below, so the wrapper binary exists by the time it is
needed.)

**3b.** Replace the tool-install block (originally lines 65–69):

```yaml
      - name: Set up Rust
        run: rustup show

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
```

with:

```yaml
      - name: Set up Rust
        run: rustup show

      - name: Rust cache
        uses: Swatinem/rust-cache@c19371144df3bb44fab255c43d04cbc2ab54d1c4 # v2.9.1

      - name: Set up sccache
        uses: mozilla-actions/sccache-action@9e7fa8a12102821edf02ca5dbea1acd0f89a2696 # v0.0.10

      - name: Install cargo-tarpaulin
        uses: taiki-e/install-action@50414676f9f5d50a65992c6dd2ed02641263226c # v2.82.10
        with:
          tool: cargo-tarpaulin@0.37.0
```

Leave the `Install Python dependencies`, `Build package`, `Run Rust
coverage`, `Run Python coverage`, and both upload steps unchanged (their
`|| true` gating is plan 002's business).

**Verify**:
`.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"` → `ok`,
and `grep -c "cargo install" .github/workflows/CI.yml` → `0` (grep exits 1;
that is expected).

### Step 4: `linux_tests` — rust-cache + activate the existing sccache

Three edits in the `linux_tests` job.

**4a.** Add the same job-level `env:` block. Replace (currently lines 111–112):

```yaml
  linux_tests:
    runs-on: ubuntu-latest
```

with:

```yaml
  linux_tests:
    runs-on: ubuntu-latest
    env:
      SCCACHE_GHA_ENABLED: "true"
      RUSTC_WRAPPER: "sccache"
```

**4b + 4c.** Replace (currently lines 123–125; note there is trailing
whitespace on the line after the sccache line in the original file — it is
part of the following blank/comment region, leave the rest of the job as-is):

```yaml
      - name: Set up Rust
        run: rustup show
      - uses: mozilla-actions/sccache-action@v0.0.3
```

with:

```yaml
      - name: Set up Rust
        run: rustup show

      - name: Rust cache
        uses: Swatinem/rust-cache@c19371144df3bb44fab255c43d04cbc2ab54d1c4 # v2.9.1

      - name: Set up sccache
        uses: mozilla-actions/sccache-action@9e7fa8a12102821edf02ca5dbea1acd0f89a2696 # v0.0.10
```

Do NOT add a `key:` input to rust-cache here — the abi3-py39 build is
Python-version-independent and the matrix legs intentionally share one cache
(see "Design decisions", item 2).

**Verify**:
`.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"` → `ok`,
and `grep -c "sccache-action@v0.0.3" .github/workflows/CI.yml` → `0`
(grep exits 1; expected).

### Step 5: Full verification battery

Run every row of the "Commands you will need" table (the non-optional ones)
and confirm each expected value:

```
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"
grep -c "cargo install" .github/workflows/CI.yml
grep -c "Swatinem/rust-cache" .github/workflows/CI.yml
grep -c "taiki-e/install-action" .github/workflows/CI.yml
grep -c "mozilla-actions/sccache-action" .github/workflows/CI.yml
grep -c "RUSTC_WRAPPER" .github/workflows/CI.yml
grep -cE 'uses: (Swatinem/rust-cache|taiki-e/install-action|mozilla-actions/sccache-action)@[0-9a-f]{40} # v' .github/workflows/CI.yml
```

**Verify**: outputs, in order: `ok`, `0`, `3`, `2`, `2`, `2`, `7`.

Also confirm nothing else changed:

```
git diff --stat
```

**Verify**: exactly one file changed, `.github/workflows/CI.yml`.

### Step 6: Commit

```
git add .github/workflows/CI.yml
git commit -m "Cache Rust builds in CI and install cargo tools as prebuilt binaries"
```

**Verify**: `git show --stat HEAD` lists exactly 1 file changed:
`.github/workflows/CI.yml`.

## Test plan

No new tests: this plan changes only CI infrastructure — which tools get
installed and what gets cached — not what CI checks or any polar-llama code
path. The verification gates are the lint-level checks in Step 5 (YAML
parses; the exact grep counts). The behavioral proof is CI-side and
post-merge: the first CI run after this lands must be green (requires plan
001 to have landed) and the second run (warm caches) should show
`security-audit` and `coverage` dropping by several minutes each — record
the before/after wall-clock in the PR or plan-index notes.

Do NOT attempt to run the workflow locally (no `act`, no live API keys —
`tests/model_client_tests.rs` integration tests are exercised only by the
manual `test-llm-apis.yml` workflow, which this plan does not touch).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml')); print('ok')"` prints `ok`
- [ ] `grep -c "cargo install" .github/workflows/CI.yml` prints `0`
- [ ] `grep -c "Swatinem/rust-cache" .github/workflows/CI.yml` prints `3` (one per job: `security-audit`, `coverage`, `linux_tests`)
- [ ] `grep -c "taiki-e/install-action" .github/workflows/CI.yml` prints `2`
- [ ] `grep -c "mozilla-actions/sccache-action" .github/workflows/CI.yml` prints `2` and `grep -c "sccache-action@v0.0.3" .github/workflows/CI.yml` prints `0`
- [ ] `grep -c "RUSTC_WRAPPER" .github/workflows/CI.yml` prints `2`
- [ ] `grep -cE 'uses: (Swatinem/rust-cache|taiki-e/install-action|mozilla-actions/sccache-action)@[0-9a-f]{40} # v' .github/workflows/CI.yml` prints `7`
- [ ] `git show --stat HEAD` lists only `.github/workflows/CI.yml` as changed
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains the index)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `.github/workflows/CI.yml` changed since `afc78da`,
  or any "Current state" excerpt (job structure, step names, line content)
  does not match the live file — in particular if the `security-audit`,
  `coverage`, or `linux_tests` job structure differs from the excerpts.
- You re-resolve any of the three action tag SHAs and get a value different
  from the table in "Current state" — a moved tag is a supply-chain red
  flag, not something to silently accept.
- The YAML parse check fails twice after a reasonable fix attempt.
- Any grep count in Step 5 is off by more than your own just-made edit can
  explain (e.g. `Swatinem/rust-cache` count is 4 — someone else added one).
- You find yourself needing to edit any file other than
  `.github/workflows/CI.yml` (including `dependabot.yml` or
  `test-llm-apis.yml`).
- Plan 001 has NOT landed and you were asked to also validate the post-merge
  CI run — the run will be red for reasons unrelated to this plan.

## Maintenance notes

- **First CI run after merge is the real verification.** Watch for:
  (a) `security-audit` and `coverage` no longer spending minutes on
  "Compiling ..." during tool install; (b) the second run showing
  `Rust cache` restore hits; (c) sccache stats — the sccache-action prints a
  cache-hit summary in its post step. Compare wall-clock against a pre-merge
  run and record the numbers.
- **Rollback lever for sccache**: if a job fails with `sccache: error` (e.g.
  GHA cache service quota/outage), delete the two job-level env lines
  (`SCCACHE_GHA_ENABLED`, `RUSTC_WRAPPER`) from the failing job — the
  sccache-action step itself is harmless when those are unset. rust-cache is
  independent and stays.
- **Cache storage**: GitHub caps a repo's Actions cache at 10 GB with LRU
  eviction. This plan adds roughly three rust-cache entries (`security-audit`
  is tiny; `coverage` and `linux_tests` each cache a polars-sized `target/`,
  which rust-cache prunes before saving) plus sccache's GHA-cache objects.
  If eviction thrash appears (every run is a cache miss), consider a
  `shared-key` across jobs or dropping rust-cache from `coverage`.
- **Version bumps are automated**: `dependabot.yml` already covers
  `github-actions`, and the `@<sha> # <tag>` format is what Dependabot
  parses — the three new pins will get bump PRs automatically.
- **Interaction with plan 016**: 016 SHA-pins all remaining action
  references in both workflow files. The 7 references this plan
  adds/replaces are already in 016's target format and must be skipped by
  016 (016's instructions already say to skip already-pinned lines). 016's
  inventory count of `uses:` lines will differ from its planning-time count
  of 30 because of this plan — that is expected and noted in 016.
- **Interaction with plan 002** (CI gating): 002 may change which steps gate
  (`|| true` removal etc.). No conflict — this plan did not touch any `run:`
  step contents in the test/coverage/audit steps.
- **Deferred**: pinning the tarpaulin/audit tool versions means new tool
  releases arrive only when someone bumps `tool:` — that is intentional
  (reproducible CI). Dependabot does not bump `taiki-e/install-action`
  `tool:` inputs; revisit the pins opportunistically.
- Reviewer should scrutinize: the three 40-hex SHAs against the upstream
  tags (commands are in "Commands you will need"), and that no `run:` step
  content changed anywhere in the diff.
