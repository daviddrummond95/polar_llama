# Plan 001: Unbreak CI — resolve cargo-audit advisory RUSTSEC-2026-0204 (crossbeam-epoch) and refresh the audit.toml review notes

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- Cargo.lock .cargo/audit.toml`
> If either in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: security
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

A new RustSec advisory published 2026-07-06 — RUSTSEC-2026-0204, "Invalid
pointer dereference in `fmt::Pointer` impl for `Atomic` and `Shared`" in
crossbeam-epoch 0.9.18 — now makes `cargo audit` exit non-zero for this repo.
The CI `security-audit` job (`.github/workflows/CI.yml:26-47`) runs exactly
`cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177` and gates,
so **every push and PR will be red** until the lockfile resolves
crossbeam-epoch >= 0.9.20. crossbeam-epoch is a transitive dependency (pulled
in via polars → rayon → crossbeam-deque), so the fix is a targeted
`cargo update -p crossbeam-epoch` — verified at planning time to resolve
0.9.18 → 0.9.20 with zero other package changes. While in `.cargo/audit.toml`,
refresh the "Last reviewed" date and record one-line dispositions for the two
non-gating audit WARNINGS so the next reviewer knows they were seen.

## Current state

Relevant files:

- `Cargo.lock` — the resolved dependency graph; pins crossbeam-epoch 0.9.18
  (lines 771–778). Also note: it still records `polar-llama` at version
  `0.5.0` (around line 1996–1998) even though `Cargo.toml:3` says `0.5.1` —
  any cargo command that touches the workspace auto-syncs this one line. That
  sync is an **expected side effect** of Step 2, not drift.
- `.cargo/audit.toml` — cargo-audit config with the two documented pyo3
  ignores and review notes (35 lines total).
- `.github/workflows/CI.yml` — the `security-audit` job that gates CI
  (read-only context; do NOT modify).

`Cargo.lock:771-778` as of afc78da:

```toml
[[package]]
name = "crossbeam-epoch"
version = "0.9.18"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "5b82ac4a3c2ca9c3460964f020e1402edd5753411d7737aa39c3714ad1b5420e"
dependencies = [
 "crossbeam-utils",
]
```

`.github/workflows/CI.yml:40-47` (the gating command — context only):

```yaml
      - name: Run cargo audit
        # RUSTSEC-2026-0176 / -0177 are pyo3 < 0.29 advisories (fixed in pyo3 >= 0.29).
        # pyo3 is pinned to 0.27 by pyo3-polars 0.26 (the PyO3<->Polars bridge);
        # even the latest pyo3-polars (0.27) only reaches pyo3 0.28, which is still
        # affected, and no pyo3-polars release supports pyo3 0.29 yet. Full
        # rationale + tracking in .cargo/audit.toml. Revisit when pyo3-polars
        # supports pyo3 >= 0.29.
        run: cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
```

`.cargo/audit.toml:29-35` as of afc78da (end of file; the ignore entries and
review-note lines you will touch in Step 4):

```toml
    "RUSTSEC-2026-0176",  # OOB read in nth/nth_back for PyList/PyTuple iterators (fixed in pyo3 >= 0.29)
    "RUSTSEC-2026-0177",  # Missing Sync bound on PyCFunction::new_closure closures (fixed in pyo3 >= 0.29)
]

# Last reviewed: 2026-07-04 (pyo3 0.27.2, pyo3-polars 0.26, polars 0.53)
# Review again: when pola-rs releases a pyo3-polars supporting pyo3 >= 0.29
```

Verified behavior at planning time (2026-07-06, on afc78da):

- `cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177` FAILS
  with `error: 1 vulnerability found!` — the one vulnerability is
  crossbeam-epoch 0.9.18 / RUSTSEC-2026-0204, "Solution: Upgrade to >=0.9.20".
- The same run prints `warning: 3 allowed warnings found` covering:
  - bincode 2.0.1, RUSTSEC-2025-0141 "Bincode is unmaintained" — transitive
    via polars-utils 0.53 (i.e. via polars; confirmed with `cargo tree -i bincode`).
  - rand 0.8.5, RUSTSEC-2026-0097 "Rand is unsound with a custom logger using
    `rand::rng()`" — transitive via instant-distance 0.6.1 (the HNSW crate
    used by `src/ann.rs`).
  - rand 0.9.2, same RUSTSEC-2026-0097 — transitive via polars-compute /
    polars-core 0.53.
  Warnings do NOT fail the job. They stay warnings; do NOT add them to the
  ignore list.
- `cargo update --dry-run -p crossbeam-epoch` resolves
  `crossbeam-epoch v0.9.18 -> v0.9.20`, locking exactly 1 package. So the
  targeted update works under the current tree — no polars/rayon bump needed.

Documented constraints this plan must honor:

- The pyo3 ignores (RUSTSEC-2026-0176 / RUSTSEC-2026-0177) are a documented,
  accepted tradeoff — pyo3 is pinned < 0.29 by pyo3-polars (full rationale in
  `.cargo/audit.toml:11-30`). Leave both entries and their comment block
  **byte-for-byte unchanged**. Do not propose or perform any pyo3/pyo3-polars
  version change.

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Targeted dep update | `cargo update -p crossbeam-epoch` | prints `Updating crossbeam-epoch v0.9.18 -> v0.9.20` (or a later 0.9.x), exit 0 |
| Confirm lockfile version | `grep -A1 'name = "crossbeam-epoch"' Cargo.lock` | shows `version = "0.9.20"` or later |
| Security audit | `cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177` | exit 0; output contains no `error:` line; still prints 3 allowed warnings (bincode + rand x2) |
| Compile check | `cargo check` | exit 0 (a warm `target/` exists locally, so this is minutes, not a cold build) |
| TOML sanity | `cargo audit --version` | exits 0 (also proves cargo-audit is installed; it is at `~/.cargo/bin/cargo-audit`, v0.22.2) |

Notes:
- `cargo audit` fetches the advisory DB from the network on each run; a
  `Fetching advisory database ...` preamble is normal.
- No Python environment, maturin build, or API keys are needed for this plan.
- Do NOT run a bare `cargo update` (no `-p`) — that is a blanket upgrade of
  the whole tree and is out of scope.

## Scope

**In scope** (the only files you may modify):
- `Cargo.lock` — the crossbeam-epoch bump, plus the automatic
  `polar-llama 0.5.0 -> 0.5.1` version-sync hunk cargo adds on its own.
- `.cargo/audit.toml` — review-date refresh + warning dispositions only.

**Out of scope** (do NOT touch, even though they look related):
- `Cargo.toml` — no version bumps of any direct dependency; crossbeam-epoch
  is transitive and the fix lives entirely in the lockfile.
- Anything pyo3 / pyo3-polars — documented pin, see constraints above.
- `.github/workflows/CI.yml` — the audit command and its `--ignore` flags are
  correct as-is.
- The `[advisories] ignore` list contents in `.cargo/audit.toml` — do not add
  or remove entries; bincode/rand are warnings, not errors, and must NOT be
  ignored.
- `src/`, `polar_llama/`, `tests/`, `pyproject.toml`, `requirements.txt`.

## Git workflow

- Branch: `advisor/001-fix-cargo-audit-ci-break` (create from `main`).
- One commit covering both files. Message style is sentence-case imperative,
  matching `git log` (e.g. "Refresh cargo-audit ignore rationale for the pyo3
  CVEs"). Suggested message:
  `Update crossbeam-epoch to 0.9.20 for RUSTSEC-2026-0204; refresh audit.toml review notes`
- Do NOT push or open a PR unless the operator instructed it.
- The repo working tree may contain pre-existing untracked files
  (`.DS_Store`, `test_cache_*.py`, `test_optional_llm.py`, `plans/`). Leave
  them alone; do not stage or delete them.

## Steps

### Step 1: Create the branch

```
git checkout -b advisor/001-fix-cargo-audit-ci-break main
```

**Verify**: `git branch --show-current` → `advisor/001-fix-cargo-audit-ci-break`

### Step 2: Targeted lockfile update of crossbeam-epoch

```
cargo update -p crossbeam-epoch
```

Expected output includes a line like:

```
    Updating crossbeam-epoch v0.9.18 -> v0.9.20
```

(0.9.20 is the minimum acceptable version; a later 0.9.x is also fine.)

Then inspect the diff:

```
git diff Cargo.lock
```

The diff must contain ONLY these two logical changes:
1. The `crossbeam-epoch` package block: `version` bumped from `0.9.18` to
   `>= 0.9.20` with a new `checksum`.
2. The `polar-llama` package block: `version = "0.5.0"` → `version = "0.5.1"`
   (cargo auto-syncing the lockfile with `Cargo.toml:3`; expected, keep it).

If the diff bumps ANY other package, that is a STOP condition.

**Verify**: `grep -A1 'name = "crossbeam-epoch"' Cargo.lock` → shows
`version = "0.9.20"` (or later 0.9.x); `git diff --stat` → only `Cargo.lock` changed.

### Step 3: Confirm the crate still compiles

```
cargo check
```

**Verify**: exit code 0. (Warnings are acceptable for this local check; the
patch-level bump of a transitive crate should produce no new ones.)

### Step 4: Refresh `.cargo/audit.toml` review notes

Edit `.cargo/audit.toml` only in the region AFTER the closing `]` of the
ignore list (line 31). Do not change lines 1–31. Replace the current trailing
two lines (lines 33–34):

```toml
# Last reviewed: 2026-07-04 (pyo3 0.27.2, pyo3-polars 0.26, polars 0.53)
# Review again: when pola-rs releases a pyo3-polars supporting pyo3 >= 0.29
```

with (use the actual date you execute this plan in place of `<YYYY-MM-DD>`):

```toml
# ---- Audit WARNINGS (informational only — warnings do not fail the CI job, ----
# ---- and must NOT be added to the ignore list above)                       ----
#
# RUSTSEC-2025-0141 (bincode 2.0.1, unmaintained): transitive via
#   polars-utils 0.53; nothing actionable until polars migrates off bincode.
#   Reviewed <YYYY-MM-DD>, accepted as warning.
# RUSTSEC-2026-0097 (rand 0.8.5 via instant-distance, rand 0.9.2 via
#   polars-compute; unsound with a custom logger inside rand::rng()): we do
#   not install a custom logger, and both copies are transitive. Reviewed
#   <YYYY-MM-DD>, accepted as warning.

# Last reviewed: <YYYY-MM-DD> (pyo3 0.27.2, pyo3-polars 0.26, polars 0.53,
# crossbeam-epoch >= 0.9.20 per RUSTSEC-2026-0204)
# Review again: when pola-rs releases a pyo3-polars supporting pyo3 >= 0.29
```

**Verify**:
`git diff .cargo/audit.toml | grep '^-' | grep -v '^---'` → the ONLY removed
lines are the two old trailing comment lines (`# Last reviewed: 2026-07-04 ...`
and `# Review again: ...`). No `-` line may contain `RUSTSEC-2026-0176`,
`RUSTSEC-2026-0177`, `ignore = [`, or `]` — that proves lines 1–31 (the
ignore list and its pyo3 rationale block) are untouched.

### Step 5: Run the exact CI audit command

```
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
echo "exit=$?"
```

**Verify**: prints `exit=0`; the output contains NO line starting with
`error:`; the output still lists the 3 allowed warnings (bincode 2.0.1
unmaintained, rand 0.8.5 unsound, rand 0.9.2 unsound). If a NEW vulnerability
(any RUSTSEC id other than 2026-0204/0176/0177) appears as an `error`, that
is a STOP condition — the advisory DB is live and may have grown since this
plan was written.

### Step 6: Commit

```
git add Cargo.lock .cargo/audit.toml
git commit -m "Update crossbeam-epoch to 0.9.20 for RUSTSEC-2026-0204; refresh audit.toml review notes"
```

**Verify**: `git status --porcelain` shows no staged/modified tracked files
(pre-existing untracked files may remain); `git show --stat HEAD` lists
exactly 2 files changed: `Cargo.lock` and `.cargo/audit.toml`.

## Test plan

No new tests: this plan changes only the dependency lockfile (patch-level
bump of a transitive crate) and a config comment file — there is no
polar-llama code path to test. The verification gates are:

- `cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177` → exit 0 (Step 5).
- `cargo check` → exit 0 (Step 3).

Do NOT run `cargo test` against `tests/model_client_tests.rs` — those
integration tests require live API keys and are exercised only by the manual
`test-llm-apis.yml` workflow. If you want an extra (optional) safety check,
`cargo test --lib` runs the key-free unit tests.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177` exits 0 and its output contains no `error:` line
- [ ] `grep -A1 'name = "crossbeam-epoch"' Cargo.lock` shows version `0.9.20` or later
- [ ] `cargo check` exits 0
- [ ] `.cargo/audit.toml` still contains exactly the entries `"RUSTSEC-2026-0176"` and `"RUSTSEC-2026-0177"` in its `ignore` list, and no other advisory is ignored: `grep -c '^    "RUSTSEC-2026-01' .cargo/audit.toml` → `2` AND `grep -c '^[[:space:]]*"RUSTSEC' .cargo/audit.toml` → `2` (the two counts match, so nothing beyond the two pyo3 advisories was added to the ignore list)
- [ ] `.cargo/audit.toml` mentions `RUSTSEC-2025-0141` and `RUSTSEC-2026-0097` in comments: `grep -c 'RUSTSEC-2025-0141\|RUSTSEC-2026-0097' .cargo/audit.toml` → `2` or more
- [ ] `git show --stat HEAD` lists only `Cargo.lock` and `.cargo/audit.toml` as changed
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains the index)

## STOP conditions

Stop and report back (do not improvise) if:

- `cargo update -p crossbeam-epoch` does not reach 0.9.20 — e.g. it reports
  "was already the latest version" while the lockfile still says 0.9.18, or
  it errors with a resolution conflict. Report the resolver output; do NOT
  force with `--precise` pins in Cargo.toml or a blanket `cargo update`.
- The `git diff Cargo.lock` after Step 2 touches any package other than
  `crossbeam-epoch` and the `polar-llama` 0.5.0→0.5.1 version sync.
- Step 5's `cargo audit` reports a vulnerability with any RUSTSEC id other
  than the three named in this plan (0204 fixed, 0176/0177 ignored) — a new
  advisory has landed since planning; it needs its own review, not an ignore.
- `cargo check` fails after the update (would mean crossbeam-epoch 0.9.20
  broke API compatibility — extremely unlikely for a patch release, but do
  not paper over it).
- `.cargo/audit.toml` at HEAD no longer matches the "Current state" excerpt
  (lines 29–35) — someone else already touched the review notes.

## Maintenance notes

- The next `polars`/`rayon` upgrade will re-resolve crossbeam-epoch anyway;
  this lockfile bump carries no forward obligation.
- Reviewer should scrutinize: (a) the `Cargo.lock` diff is minimal — exactly
  one crate bumped plus the 0.5.0→0.5.1 version sync; (b) the audit.toml
  ignore list is byte-identical to before (the pyo3 tradeoff block, lines
  1–31, must be untouched).
- The bincode (RUSTSEC-2025-0141) and rand (RUSTSEC-2026-0097) WARNINGS were
  deliberately left as warnings. If cargo-audit or RustSec later escalates
  either to a hard vulnerability, it will need a real disposition (upstream
  polars bump or a reviewed ignore with rationale) — that decision is
  explicitly deferred out of this plan.
- Deferred: the committed `Cargo.lock` was stale on the `polar-llama` version
  field (0.5.0 vs Cargo.toml's 0.5.1) since the 0.5.1 release commit; this
  plan's commit fixes it as a side effect. If release tooling is ever added,
  it should regenerate the lockfile as part of version bumps.
