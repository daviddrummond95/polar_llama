"""CI-safe tests for the collapsed shared-prefix prefill planner.

No mlx, no GPU, no network: only the pure planning/offset/raggedness logic
(:mod:`polar_llama.local.collapsed_prefill`) plus a fake-cache check of the
replication semantics. The MLX-backed mechanism itself is validated on real
hardware by ``benchmarks/validate_collapsed_prefill.py``.
"""

import copy

import pytest

from polar_llama.local.collapsed_prefill import (
    DEFAULT_MIN_PREFIX_TOKENS,
    CollapsePlan,
    common_token_prefix_len,
    plan_collapsed_prefill,
    replicate_prefix_caches,
)

pytestmark = pytest.mark.local


# ---------------------------------------------------------------------------
# common_token_prefix_len
# ---------------------------------------------------------------------------


def test_lcp_basic():
    rows = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 9, 9, 9],
        [1, 2, 3, 4, 7],
    ]
    assert common_token_prefix_len(rows) == 3


def test_lcp_single_row_is_full_length():
    assert common_token_prefix_len([[5, 6, 7]]) == 3


def test_lcp_identical_rows():
    assert common_token_prefix_len([[1, 2], [1, 2]]) == 2


def test_lcp_no_common_prefix():
    assert common_token_prefix_len([[1, 2], [2, 1]]) == 0


def test_lcp_prefix_row_shorter_than_others():
    # One row is itself a prefix of the others.
    assert common_token_prefix_len([[1, 2, 3, 4], [1, 2]]) == 2


def test_lcp_empty_input_raises():
    with pytest.raises(ValueError):
        common_token_prefix_len([])


# ---------------------------------------------------------------------------
# plan_collapsed_prefill
# ---------------------------------------------------------------------------


def _rows(prefix_len, suffixes):
    prefix = list(range(1000, 1000 + prefix_len))
    return [prefix + list(s) for s in suffixes]


def test_plan_reconstructs_every_row_exactly():
    # The load-bearing invariant: prefix + suffix == full prompt, per row.
    rows = _rows(20, [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    plan = plan_collapsed_prefill(rows, min_prefix_tokens=4)
    assert plan is not None
    assert plan.prefix_len == 20
    for original, suffix in zip(rows, plan.suffixes):
        assert list(plan.prefix_tokens) + list(suffix) == original
        assert len(suffix) >= 1


def test_plan_preserves_ragged_suffix_lengths_and_order():
    rows = _rows(10, [[1], [2, 3, 4, 5], [6, 7]])
    plan = plan_collapsed_prefill(rows, min_prefix_tokens=2)
    assert [len(s) for s in plan.suffixes] == [1, 4, 2]
    assert plan.suffix_lists() == [[1], [2, 3, 4, 5], [6, 7]]
    # suffix_lists must be fresh mutable lists (batch machinery mutates them)
    lists = plan.suffix_lists()
    lists[0].append(99)
    assert plan.suffix_lists() == [[1], [2, 3, 4, 5], [6, 7]]


def test_plan_clamps_prefix_so_every_suffix_is_nonempty():
    # Identical rows: LCP == full length, but the batch machinery needs at
    # least one token per row, so the prefix is clamped to len-1.
    rows = [[1, 2, 3, 4, 5]] * 3
    plan = plan_collapsed_prefill(rows, min_prefix_tokens=2)
    assert plan.prefix_len == 4
    assert all(list(s) == [5] for s in plan.suffixes)


def test_plan_clamps_when_one_row_is_a_prefix_of_another():
    rows = [[1, 2, 3, 4, 5, 6], [1, 2, 3]]
    plan = plan_collapsed_prefill(rows, min_prefix_tokens=2)
    # LCP is 3 but row 1 would get an empty suffix; clamp to 2.
    assert plan.prefix_len == 2
    assert plan.suffix_lists() == [[3, 4, 5, 6], [3]]


def test_plan_returns_none_below_min_prefix():
    rows = _rows(4, [[1], [2]])
    assert plan_collapsed_prefill(rows, min_prefix_tokens=5) is None
    assert plan_collapsed_prefill(rows, min_prefix_tokens=4) is not None


def test_plan_default_min_prefix_gate():
    short = _rows(DEFAULT_MIN_PREFIX_TOKENS - 1, [[1], [2]])
    assert plan_collapsed_prefill(short) is None
    long_enough = _rows(DEFAULT_MIN_PREFIX_TOKENS, [[1], [2]])
    assert plan_collapsed_prefill(long_enough) is not None


def test_plan_rejects_empty_rows_and_bad_args():
    with pytest.raises(ValueError):
        plan_collapsed_prefill([])
    with pytest.raises(ValueError):
        plan_collapsed_prefill([[1, 2, 3], []])
    with pytest.raises(ValueError):
        plan_collapsed_prefill([[1, 2, 3]], min_prefix_tokens=0)


def test_plan_token_accounting():
    rows = _rows(100, [[1, 2], [3, 4, 5], [6]])
    plan = plan_collapsed_prefill(rows)
    assert plan.naive_prefill_tokens == 102 + 103 + 101
    assert plan.collapsed_prefill_tokens == 100 + 2 + 3 + 1
    assert plan.saved_prefill_tokens == 2 * 100
    # Collapse always strictly saves prefix_len * (rows - 1) tokens.
    assert plan.saved_prefill_tokens == plan.prefix_len * (plan.rows - 1)


def test_plan_accepts_tuples_and_is_immutable():
    rows = [(9, 9, 9, 9, 9, 9, 9, 9, 1), (9, 9, 9, 9, 9, 9, 9, 9, 2)]
    plan = plan_collapsed_prefill(rows)
    assert isinstance(plan, CollapsePlan)
    assert plan.prefix_len == 8
    with pytest.raises(AttributeError):
        plan.prefix_tokens = ()  # frozen dataclass


# ---------------------------------------------------------------------------
# replicate_prefix_caches (fake caches; semantics only, no mlx)
# ---------------------------------------------------------------------------


class FakeLayerCache:
    def __init__(self, payload):
        self.payload = list(payload)


def test_replicate_share_mode_attaches_same_object():
    pc = [FakeLayerCache([1]), FakeLayerCache([2])]
    out = replicate_prefix_caches(pc, 3)
    assert len(out) == 3
    assert all(row is pc for row in out)  # zero copies: merge() reads only


def test_replicate_clone_mode_deep_copies_per_row():
    pc = [FakeLayerCache([1])]
    out = replicate_prefix_caches(pc, 2, clone_per_row=True)
    assert len(out) == 2
    assert out[0] is not pc and out[1] is not pc
    assert out[0][0] is not pc[0] and out[1][0] is not pc[0]
    out[0][0].payload.append(99)
    assert pc[0].payload == [1] and out[1][0].payload == [1]


def test_replicate_zero_rows_and_negative():
    assert replicate_prefix_caches([FakeLayerCache([])], 0) == []
    with pytest.raises(ValueError):
        replicate_prefix_caches([FakeLayerCache([])], -1)


def test_replicate_clone_is_deepcopy_compatible():
    # The real caches are replicated with copy.deepcopy in clone mode; make
    # sure our fake mirrors that contract for the test above to be honest.
    pc = [FakeLayerCache([7])]
    assert copy.deepcopy(pc)[0].payload == [7]


# ---------------------------------------------------------------------------
# End-to-end planning against a fake tokenizer (gate-workload shape)
# ---------------------------------------------------------------------------


class FakeTemplateTokenizer:
    """Mimics a chat template that folds the system prompt into the first
    user turn (the Gemma behaviour that broke the text-level prefix warm)."""

    def render(self, system, user):
        # System-only rendering would be just [BOS] -- the trap this module
        # sidesteps by deriving the prefix from FULL tokenizations.
        text = f"<u>{system}\n\n{user}</u>"
        return [2] + [ord(c) for c in text]  # 2 = BOS


def test_gate_workload_shape_collapses_at_the_row_seam():
    tok = FakeTemplateTokenizer()
    system = "shared rules " * 50
    rows = [tok.render(system, f"Record {i}: classify it") for i in range(8)]
    plan = plan_collapsed_prefill(rows)
    assert plan is not None
    # The LCP must swallow BOS + the whole folded system prompt + the common
    # "Record " head of the user turn: strictly more than the system alone.
    sys_only_len = len(tok.render(system, "")) - len("</u>")
    assert plan.prefix_len > sys_only_len
    for original, suffix in zip(rows, plan.suffixes):
        assert list(plan.prefix_tokens) + list(suffix) == original
