# Plan 016: Pin all GitHub Actions to full commit SHAs (release path first), auto-bumped by Dependabot

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- .github/workflows/CI.yml .github/workflows/test-llm-apis.yml .github/dependabot.yml`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding. NOTE:
> plans/015-ci-caching.md (a soft dependency) also edits `CI.yml` — if it has
> landed, line numbers below will have shifted and the
> `mozilla-actions/sccache-action` pin may have changed or moved. That alone
> is NOT a STOP condition: re-derive the action inventory with the Step 2
> command and proceed, matching `uses:` lines by content, not by line number.
> Any OTHER kind of mismatch (an action in this plan's inventory missing
> entirely, a new workflow file, a `uses:` line already SHA-pinned
> differently than expected) IS a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/015-ci-caching.md (soft — same file, `.github/workflows/CI.yml`; this plan can be executed before or after it, see drift check)
- **Category**: security
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

Every action in this repo's workflows is referenced by a **mutable tag**
(`@v1`, `@v7`, ...). A tag can be repointed at any time by whoever controls
the action's repo — including an attacker who compromises it. The release job
(`.github/workflows/CI.yml:311-340`) runs with `permissions: id-token: write`
for PyPI trusted publishing and invokes the third-party
`PyO3/maturin-action@v1` to upload; a repointed/compromised tag there can
exfiltrate the OIDC token or publish a tampered `polar-llama` wheel to PyPI.
The coverage job passes `CODECOV_TOKEN` to `codecov/codecov-action@v5`
(a vendor that has had a real supply-chain incident: its Bash Uploader was
tampered with in 2021). Pinning every `uses:` to a full 40-character commit
SHA makes the reference immutable; the existing Dependabot `github-actions`
entry keeps the pins fresh via PRs, so this costs nothing ongoing.

## Current state

Relevant files:

- `.github/workflows/CI.yml` — main CI: security-audit, coverage, linux_tests
  matrix, wheel builds (linux/windows/macos), sdist, and the tag-triggered
  `release` job that publishes to PyPI via trusted publishing (OIDC).
  27 `uses:` lines, all tag-pinned.
- `.github/workflows/test-llm-apis.yml` — manual (workflow_dispatch) live-API
  test workflow; receives provider API keys from repo secrets. 3 `uses:`
  lines, all tag-pinned.
- `.github/dependabot.yml` — already contains a `github-actions` ecosystem
  entry (lines 39-55), so SHA pins will receive automated bump PRs. This file
  needs **verification only**, no edit (Step 5).

### Complete inventory of tag-pinned actions (verified at afc78da)

7 unique `action@tag` pairs across 30 `uses:` lines:

| Action | Tag | Occurrences (file:line) |
|--------|-----|-------------------------|
| `actions/checkout` | `v7` | CI.yml:29,60,118,226,256,280,299; test-llm-apis.yml:21 |
| `actions/setup-python` | `v6` | CI.yml:30,61,119,227,257,281; test-llm-apis.yml:26 |
| `codecov/codecov-action` | `v5` | CI.yml:97 |
| `actions/upload-artifact` | `v5` | CI.yml:106,245,269,291,306; test-llm-apis.yml:66 |
| `mozilla-actions/sccache-action` | `v0.0.3` | CI.yml:125 |
| `PyO3/maturin-action` | `v1` | CI.yml:238,263,285,301,337 |
| `actions/download-artifact` | `v6` | CI.yml:321,328 |

Highest-value pins, in priority order (finish these even if something else
resists): **release job** (CI.yml:321, 328, 337 — OIDC publish path) >
**coverage job** (CI.yml:97 — receives `CODECOV_TOKEN`) > everything else.
Pin ALL of them, including GitHub first-party `actions/*`, for consistency.

### Excerpts as of afc78da

`.github/workflows/CI.yml:311-340` (the release job — the critical path):

```yaml
  release:
    name: Release
    if: "startsWith(github.ref, 'refs/tags/')"
    needs: [linux, windows, macos, sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write # IMPORTANT: mandatory for trusted publishing
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v6
        with:
          path: dist
          pattern: wheels-*
          merge-multiple: true

      - name: Download sdist
        uses: actions/download-artifact@v6
        with:
          name: sdist
          path: dist

      - name: List distribution files
        run: ls -la dist/

      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --non-interactive --skip-existing dist/*
```

`.github/workflows/CI.yml:96-103` (coverage token consumer — reference the
secret's location only; never echo or copy its value anywhere):

```yaml
      - name: Upload coverage reports to Codecov
        uses: codecov/codecov-action@v5
        with:
          files: ./coverage/cobertura.xml,./coverage/python-coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false
          token: ${{ secrets.CODECOV_TOKEN }}
```

`.github/dependabot.yml:39-45` (already present — verify, do not edit):

```yaml
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
```

### Target shape

Every `uses:` line becomes `owner/repo@<40-hex-sha> # <most-specific-version-tag>`.
Example (SHA below is REAL and was verified at planning time, but you MUST
re-resolve at execution time — tags move; do not copy SHAs from this plan):

```yaml
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.1.x
```

The trailing `# vX.Y.Z` comment is load-bearing: humans read the version from
it, and Dependabot parses and updates it alongside the SHA.

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| GitHub API access | `gh auth status` | prints "Logged in to github.com" (read access to public repos is all that's needed) |
| List all `uses:` lines | `grep -En 'uses:' .github/workflows/*.yml` | 30 lines at afc78da (may differ if plan 015 landed — re-inventory) |
| Tag-pin detector | `grep -En 'uses:.*@v[0-9]' .github/workflows/*.yml` | 30 lines before the change; **0 lines after** |
| SHA-pin counter | `grep -Eh 'uses:' .github/workflows/*.yml \| grep -cE '@[0-9a-f]{40} # v'` | equals the total number of `uses:` lines after the change |
| YAML parse | `.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')+['.github/dependabot.yml']]; print('yaml ok')"` | prints `yaml ok` |
| Dependabot entry | `grep -n 'package-ecosystem: "github-actions"' .github/dependabot.yml` | one match (line 40 at afc78da) |

Notes:
- `gh` (v2.94.0) is installed and authenticated on this machine. No live LLM
  API keys are needed anywhere in this plan. If `gh` is somehow unavailable,
  the fallback is anonymous curl (rate-limited to 60 req/hr — enough for the
  ~15 calls needed): `curl -s https://api.github.com/repos/OWNER/REPO/git/ref/tags/TAG`.
- No Rust/Python build is needed — this plan touches only YAML under `.github/`.

## Scope

**In scope** (the only files you may modify):
- `.github/workflows/CI.yml` — replace tags with SHAs on `uses:` lines only.
- `.github/workflows/test-llm-apis.yml` — same.
- `.github/dependabot.yml` — verify the `github-actions` entry exists (it
  does at afc78da); edit ONLY if it has been removed since, adding it back
  exactly as excerpted above.

**Out of scope** (do NOT touch, even though it looks related):
- Any workflow logic: job structure, steps, `run:` blocks, `with:` inputs,
  `env:`, matrices, triggers. Only the ref part of `uses:` lines changes.
- `permissions:` blocks — the release job's `id-token: write` is required
  for PyPI trusted publishing; leave every permissions block byte-identical.
- Adding, removing, or swapping actions (e.g. do not "upgrade" any action to
  a newer major while you're in there — pin the SHA of the tag currently
  referenced).
- The `cargo`/`pip` entries in `.github/dependabot.yml`.
- Everything outside `.github/`.

## Git workflow

- Branch: `advisor/016-sha-pin-actions` (create from `main`).
- One commit. Message style is sentence-case imperative, matching `git log`
  (e.g. "Refresh cargo-audit ignore rationale for the pyo3 CVEs"). Suggested:
  `Pin all GitHub Actions to full commit SHAs`
- Do NOT push or open a PR unless the operator instructed it.
- The working tree may contain pre-existing untracked files (`.DS_Store`,
  `test_cache_*.py`, `test_optional_llm.py`, `plans/`). Leave them alone.

## Steps

### Step 1: Create the branch

```
git checkout -b advisor/016-sha-pin-actions main
```

**Verify**: `git branch --show-current` → `advisor/016-sha-pin-actions`

### Step 2: Build the live inventory of tag-pinned actions

```
grep -hE 'uses:' .github/workflows/*.yml | sed 's/.*uses: //' | sort -u
```

**Verify**: at afc78da this prints exactly these 7 entries (if plan 015
landed first, the list may differ — e.g. `mozilla-actions/sccache-action`
gone or re-versioned, or new caching actions added; that is fine, pin
whatever the live list contains):

```
PyO3/maturin-action@v1
actions/checkout@v7
actions/download-artifact@v6
actions/setup-python@v6
actions/upload-artifact@v5
codecov/codecov-action@v5
mozilla-actions/sccache-action@v0.0.3
```

If any entry is already a 40-hex SHA, someone pinned it since planning —
treat as a STOP condition (partial prior work needs human review).

### Step 3: Resolve each tag to a commit SHA — at execution time, not from this plan

Tags move; SHAs written into this plan file would already be stale. Resolve
every tag NOW with this exact sequence. For each `OWNER/REPO` and `TAG` from
Step 2's list, run:

```bash
# 3a. Resolve the tag ref
gh api repos/OWNER/REPO/git/ref/tags/TAG --jq '.object.type + " " + .object.sha'
```

- If the output starts with `commit ` → the SHA printed is the commit SHA.
  Use it. (Lightweight tag — e.g. `actions/checkout` `v7` resolved this way
  at planning time: `commit 9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0`.)
- If the output starts with `tag ` → annotated tag; dereference the tag
  object to get the commit it points at:

```bash
# 3b. (annotated tags only) Dereference tag object -> commit SHA
gh api repos/OWNER/REPO/git/tags/<sha-from-3a> --jq '.object.sha'
```

Then find the most specific version tag pointing at that commit, for the
trailing comment (floating tags like `v1` or `v7` usually alias a concrete
`vX.Y.Z`):

```bash
# 3c. Find concrete tag name(s) for the comment
gh api "repos/OWNER/REPO/tags?per_page=100" --paginate \
  --jq '.[] | select(.commit.sha=="<COMMIT-SHA>") | .name'
```

Pick the most specific name printed (e.g. prefer `v1.9.0` over `v1`). If 3c
prints nothing (tag older than the first 100 + pagination miss, or release
tags pruned), fall back to using the original tag from Step 2 as the comment
(e.g. `# v0.0.3`) — the SHA is still correct and Dependabot still works.

Convenience: this loop does 3a-3c for the whole afc78da inventory in one go
(adjust the list to match your Step 2 output):

```bash
for at in PyO3/maturin-action@v1 actions/checkout@v7 \
          actions/download-artifact@v6 actions/setup-python@v6 \
          actions/upload-artifact@v5 codecov/codecov-action@v5 \
          mozilla-actions/sccache-action@v0.0.3; do
  repo=${at%@*}; tag=${at##*@}
  read -r type sha <<<"$(gh api "repos/$repo/git/ref/tags/$tag" --jq '.object.type + " " + .object.sha')"
  [ "$type" = "tag" ] && sha=$(gh api "repos/$repo/git/tags/$sha" --jq '.object.sha')
  name=$(gh api "repos/$repo/tags?per_page=100" --paginate \
    --jq ".[] | select(.commit.sha==\"$sha\") | .name" | sort -V | tail -1)
  echo "$repo@$tag -> $sha # ${name:-$tag}"
done
```

**Verify**: every line of output ends with a 40-hex SHA followed by
`# v<something>`, and there is one line per Step 2 entry. Sanity-check each
SHA is exactly 40 hex chars. If ANY tag fails to resolve (404, empty SHA),
that is a STOP condition — record which `OWNER/REPO@TAG` failed.

### Step 4: Rewrite every `uses:` line in both workflows

For each `action@tag -> SHA # vX.Y.Z` resolution from Step 3, replace ALL
occurrences in BOTH files. The old `@tag` strings are unique enough for a
global textual replace, e.g.:

```bash
sed -i '' \
  -e 's|uses: actions/checkout@v7|uses: actions/checkout@<SHA> # <vX.Y.Z>|g' \
  .github/workflows/CI.yml .github/workflows/test-llm-apis.yml
```

(one `-e` per action, substituting the real SHA and version from Step 3; or
make the same edits with your file-editing tool). Nothing else on those lines
changes — indentation, `- ` prefixes, and everything before `uses:` stay
byte-identical.

**Verify** (all three):
1. `grep -En 'uses:.*@v[0-9]' .github/workflows/*.yml` → no output, exit 1
   (no tag pins remain).
2. `n=$(grep -hcE 'uses:' .github/workflows/CI.yml .github/workflows/test-llm-apis.yml | paste -sd+ - | bc); p=$(grep -hE 'uses:' .github/workflows/*.yml | grep -cE '@[0-9a-f]{40} # v'); echo "uses=$n pinned=$p"` →
   the two numbers are equal (30 and 30 at afc78da).
3. `git diff -U0 .github/workflows/ | grep -E '^[+-]' | grep -v '^[+-][+-]' | grep -vE 'uses:'` → no output
   (every changed line is a `uses:` line; nothing else was touched).

### Step 5: Verify (do not edit) the Dependabot github-actions entry

```
grep -n 'package-ecosystem: "github-actions"' .github/dependabot.yml
```

**Verify**: exactly one match (line 40 at afc78da). If present: make NO edit
to this file and move on. Only if the entry has been removed since planning,
add the block excerpted in "Current state" (lines 39-55 at afc78da,
including the `schedule`, `labels`, `commit-message`, `reviewers`,
`assignees` keys) after the `pip` entry, then re-run the grep → one match.

### Step 6: Parse check on all touched YAML

```
.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')+['.github/dependabot.yml']]; print('yaml ok')"
```

**Verify**: prints `yaml ok`, exit 0. (Use the repo `.venv` — it has PyYAML;
the system python3 on this machine does NOT. If `.venv` is missing, any
python3 with PyYAML installed works: `python3 -m pip install pyyaml`.)

### Step 7: Commit

```
git add .github/workflows/CI.yml .github/workflows/test-llm-apis.yml
git commit -m "Pin all GitHub Actions to full commit SHAs"
```

(Include `.github/dependabot.yml` in the `git add` only if Step 5 required
the re-add edit.)

**Verify**: `git show --stat HEAD` lists exactly the 2 workflow files (3
files only in the Step 5 edge case); `git status --porcelain` shows no
remaining staged/modified tracked files.

## Test plan

No unit tests apply — the change is workflow metadata only; the repo's Rust
and Python code is untouched. The verification gates are the machine checks
in Steps 4-6 (tag-pin grep empty, SHA-pin count equals `uses:` count, YAML
parses). Do NOT attempt to run the workflows locally and do NOT trigger
`test-llm-apis.yml` (it needs live API keys and spends money).

Real-world validation happens on the first push/PR of the branch (if the
operator authorizes a push): the `security-audit`, `coverage`, `linux_tests`
and wheel jobs on the PR exercise every pinned action except the two
`download-artifact` steps and the publish step in `release` (tag-gated). If
CI is run, "all jobs still pass" is the acceptance signal; a failure of the
form "unable to resolve action" means a wrong SHA — recheck Step 3 for that
action.

## Done criteria

Machine-checkable. ALL must hold (none require network or API keys):

- [ ] `grep -En 'uses:.*@v[0-9]' .github/workflows/*.yml` returns 0 rows (exit code 1)
- [ ] Every `uses:` line is SHA+comment pinned: `grep -hE 'uses:' .github/workflows/*.yml | grep -vcE '@[0-9a-f]{40} # v'` → `0`
- [ ] `grep -c 'package-ecosystem: "github-actions"' .github/dependabot.yml` → `1`
- [ ] YAML parse command from Step 6 prints `yaml ok`
- [ ] `git diff afc78da..HEAD -- .github/workflows/ | grep -E '^[+-]' | grep -v '^[+-][+-]' | grep -vE 'uses:'` → empty, IF plan 015 has not landed (skip this check if it has — 015 legitimately changes non-`uses:` lines)
- [ ] `git show --stat HEAD` lists only in-scope files
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains the index)

## STOP conditions

Stop and report back (do not improvise) if:

- A SHA cannot be resolved for any tag in the Step 2 inventory (`gh api`
  404s, returns an empty/short SHA, or the dereference in 3b fails). Report
  exactly which `OWNER/REPO@TAG` failed. Do not guess a SHA from GitHub's UI
  history, do not skip that action, and do not substitute a different
  version. (Priority note for the report: failures on the release path —
  `actions/download-artifact`, `PyO3/maturin-action` — or on
  `codecov/codecov-action` are the ones that matter most.)
- `gh` is unauthenticated AND the anonymous-curl fallback is rate-limited
  before all tags resolve.
- Step 2 reveals a `uses:` reference that is already a 40-hex SHA (partial
  prior pinning — needs human reconciliation) or an action not in this
  plan's inventory whose tag you cannot resolve.
- The Step 4 diff check shows any changed line that is not a `uses:` line.
- The YAML parse in Step 6 fails after one fix attempt.
- The drift check shows in-scope changes NOT attributable to
  plans/015-ci-caching.md.

## Maintenance notes

- **Ongoing bumps are Dependabot's job**: the existing `github-actions`
  entry in `.github/dependabot.yml` opens weekly PRs that update both the
  SHA and the trailing `# vX.Y.Z` comment. Reviewers of those PRs should
  confirm the comment version matches the SHA (Dependabot does this
  correctly; hand-edits can desync them).
- **Reviewer focus for THIS change's PR**: (a) diff touches only `uses:`
  lines; (b) each SHA actually corresponds to the tag named in its comment —
  spot-check the release-job pins (`actions/download-artifact`,
  `PyO3/maturin-action`) and `codecov/codecov-action` by opening
  `https://github.com/OWNER/REPO/commits/<sha>` and confirming the tag;
  (c) `permissions:` blocks are unchanged.
- **Adding a new action later**: pin it SHA+comment from day one; the Done
  criteria grep in this plan (`uses:.*@v[0-9]`) makes a good CI lint if the
  team ever adds a workflow-hygiene check (deferred — out of scope here).
- **Interaction with plans/015-ci-caching.md**: whichever plan lands second
  must keep the other's invariant — 015's executor should SHA-pin any
  caching action it introduces if this plan landed first; this plan's
  executor pins whatever caching actions 015 introduced if it landed first.
- Deferred (explicitly out of scope): upgrading any action to a newer major;
  adding an actionlint/zizmor workflow-lint job; pinning the `rustup`/`uv`
  curl-pipe installs in the workflows (a different supply-chain surface,
  worth its own finding).
