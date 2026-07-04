# Submission steps: mlx-lm #1384 (Gemma 3n batched generation fix)

Exact steps to turn `patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch` into
an upstream PR against `ml-explore/mlx-lm`. The title/body to paste into the
PR are in `patches/PR_1384.md` (first line is the title, the rest is the
body).

> **CAVEAT — read first:** `patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch`
> was authored and validated against **mlx-lm 0.31.3** (the latest published
> release as of this writing). `main` may have drifted since — re-verify the
> patch still applies to `main` (step 3) *before* opening the PR, and re-run
> the validator (step 4) against whatever commit you actually branch from.
> If `main` has moved on and the patch no longer applies cleanly, re-derive
> the same two changes (offset snapshot + threaded shared KV, see
> `docs/mlx_lm_1384_fix.md`) by hand against current `gemma3n.py` rather than
> forcing the old hunks in.

## 1. Fork

Fork <https://github.com/ml-explore/mlx-lm> to your own GitHub account (web
UI, or `gh repo fork ml-explore/mlx-lm --clone=false`).

## 2. Clone the fork and create a branch

```bash
git clone git@github.com:<your-username>/mlx-lm.git
cd mlx-lm
git checkout -b fix/gemma3n-batched-shared-kv
```

## 3. Apply the patch

The diff targets `mlx_lm/models/gemma3n.py` and is a standard `-p1` diff
(`a/mlx_lm/...`, `b/mlx_lm/...`), so apply it from the **repo root** of your
`mlx-lm` clone:

```bash
# from the root of your mlx-lm clone (the dir containing mlx_lm/, docs/, etc.)
patch -p1 < /path/to/polar-llama/patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch
```

Equivalent `git apply` form (also from the repo root):

```bash
git apply --check /path/to/polar-llama/patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch  # dry run first
git apply /path/to/polar-llama/patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch
```

If either command reports rejected hunks or fails the `--check`, `main` has
drifted from 0.31.3 — see the caveat above; diff `mlx_lm/models/gemma3n.py`
against the version this patch was built from
(`mlx-lm==0.31.3`, `mlx_lm/models/gemma3n.py`) to find where it moved, and
re-apply the two logical changes by hand:

1. snapshot `offset = mx.array(cache.offset) if cache is not None else 0` in
   the non-shared branch of `Gemma3nAttention.__call__`, and
2. thread each concrete layer's post-`update_and_fetch` `(keys, values)` +
   offset through `LanguageModel.__call__` to the KV-shared layers instead of
   those layers reading `cache.state`.

Confirm only `mlx_lm/models/gemma3n.py` changed:

```bash
git status
git diff --stat
```

## 4. Run tests

**a. mlx-lm's own test suite** (from the repo root):

```bash
pip install -e ".[test]"   # or however mlx-lm's dev/test extras are declared
python -m pytest tests/ -k gemma3n
python -m pytest tests/   # full suite, to catch any unrelated regression
```

**b. Our token-parity validator** (`benchmarks/validate_1384_fix.py` from
this `polar-llama` repo) — sequential vs batched (`BatchGenerator`)
comparison, PASS on token-identical output or a genuine near-tie by logprob
gap:

```bash
# with your patched mlx-lm installed in the active environment (e.g. `pip install -e .`
# from the mlx-lm clone), run from the polar-llama repo:
python benchmarks/validate_1384_fix.py                 # both models, default settings
python benchmarks/validate_1384_fix.py --strict         # require exact match, no near-tie allowance
python benchmarks/validate_1384_fix.py --models hybrid  # gemma-3n only
python benchmarks/validate_1384_fix.py --models control # Qwen control only, checks no regression
```

Expect: hybrid (Gemma 3n) batched B=4 token-identical on 3/4 prompts + one
exact-logprob-tie divergence on the 4th (see `patches/PR_1384.md` for the
exact transcript); Qwen control 4/4 token-identical. If your results differ
materially from `patches/PR_1384.md` (e.g. more than one non-tie
divergence), stop and re-check the patch applied cleanly against the `main`
you're on before proceeding.

Both test runs should pass before you commit.

## 5. Commit, push, open the PR

```bash
git add mlx_lm/models/gemma3n.py
git commit -m "Fix Gemma 3n batched generation (shared-KV layers + RoPE offset aliasing)

Fixes #1384"
git push -u origin fix/gemma3n-batched-shared-kv
```

Open the PR against `ml-explore/mlx-lm` `main`:

```bash
gh pr create \
  --repo ml-explore/mlx-lm \
  --base main \
  --head <your-username>:fix/gemma3n-batched-shared-kv \
  --title "Fix Gemma 3n batched generation (shared-KV layers + RoPE offset aliasing) — fixes #1384" \
  --body-file /path/to/polar-llama/patches/PR_1384.md
```

(Or via the web UI: open a PR from your fork's branch, paste the title —
first line of `PR_1384.md`, after `# Title: ` — into the PR title field, and
paste everything below that first line into the PR description.)

Make sure the PR references `#1384` (e.g. via "Fixes #1384" in the body,
already included in `PR_1384.md`) so GitHub links and auto-closes the issue
on merge.

## 6. After opening

- Watch CI on the PR for anything the local test run didn't catch (different
  MLX/hardware versions, lint/type checks, etc.) and fix forward on the same
  branch if needed.
- Be ready to respond to maintainer review — in particular, be prepared to
  discuss the two-optional-kwargs API addition to `Gemma3nAttention.__call__`
  (`shared_kv`, `offset`) if a maintainer prefers a different internal
  plumbing mechanism; the root-cause analysis and validation data in
  `PR_1384.md` should not need to change even if the implementation shape is
  bikeshedded.
