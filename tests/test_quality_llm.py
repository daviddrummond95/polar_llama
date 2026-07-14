"""Real-API tests for the survey data-quality embedding/LLM tier (issue
#80): `near_duplicate` and `likely_ai` via `QualityConfig(llm_tier=True)`.

Gated on `OPENAI_API_KEY` -- never runs in CI without a key (convention
from `tests/test_codebook.py::test_induce_and_apply_codebook_real_api`).
The heuristic tier (zero API calls) is fully covered, unconditionally, by
`tests/test_quality_flags.py`.
"""

import os

import polars as pl
import pytest

from polar_llama import Provider, QualityConfig, quality_report

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


AI_STYLE_ANSWER = (
    "As a valued customer, I appreciate the opportunity to share my feedback "
    "regarding this product. On one hand, the build quality is commendable "
    "and the packaging was handled with care. On the other hand, the setup "
    "process could benefit from clearer documentation for new users. In "
    "conclusion, I believe this represents a solid offering with room for "
    "incremental improvement, and I would recommend it to others seeking a "
    "reliable solution within this category of products."
)

HUMAN_STYLE_ANSWERS = [
    "tbh works fine i guess, box was a lil banged up tho lol",
    "meh. instructions sucked but got it working eventually. wouldve been nice to have a video",
    "pretty good!! fast shipping, kid loves it, would prob buy again ngl",
]

NEAR_DUP_A = "The onboarding process was confusing and I had to contact support twice to get it working."
NEAR_DUP_B = "The onboarding process was confusing and I had to contact support twice to get things working."


@pytest.fixture(scope="module")
def llm_fixture_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "respondent_id": ["dup_a", "dup_b", "ai_style", "human_1", "human_2", "human_3"],
            "oe1": [NEAR_DUP_A, NEAR_DUP_B, AI_STYLE_ANSWER, *HUMAN_STYLE_ANSWERS],
        }
    )


@pytest.fixture(scope="module")
def llm_report(llm_fixture_df) -> "pl.DataFrame":
    config = QualityConfig(
        id_column="respondent_id",
        text_columns=["oe1"],
        llm_tier=True,
        embedding_provider=Provider.OPENAI,
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
    )
    return quality_report(llm_fixture_df, config)


def test_llm_tier_preserves_height(llm_fixture_df, llm_report):
    assert llm_report.df.height == llm_fixture_df.height


def test_near_duplicate_pair_flags_each_other(llm_report):
    q = llm_report.df.select("respondent_id", "quality").rows(named=True)
    by_id = {row["respondent_id"]: row["quality"] for row in q}

    assert by_id["dup_a"]["near_duplicate"]["flag"] is True
    assert by_id["dup_b"]["near_duplicate"]["flag"] is True
    assert by_id["dup_a"]["near_duplicate"]["neighbor_id"] == "dup_b"
    assert by_id["dup_b"]["near_duplicate"]["neighbor_id"] == "dup_a"

    others_scores = [
        by_id[rid]["near_duplicate"]["score"]
        for rid in ("ai_style", "human_1", "human_2", "human_3")
        if by_id[rid]["near_duplicate"]["score"] is not None
    ]
    for score in others_scores:
        assert score < 0.97


def test_likely_ai_scores_ai_style_higher_than_human_style(llm_report):
    q = llm_report.df.select("respondent_id", "quality").rows(named=True)
    by_id = {row["respondent_id"]: row["quality"] for row in q}

    ai_score = by_id["ai_style"]["likely_ai"]["score"]
    human_scores = [
        by_id[rid]["likely_ai"]["score"] for rid in ("human_1", "human_2", "human_3")
    ]
    assert ai_score is not None
    assert all(s is not None for s in human_scores)

    # Graded ordering only -- never an absolute-value or perfection claim
    # (see the mandatory AI-detection limitation in docs/QUALITY_FLAGS.md).
    assert ai_score > max(human_scores)

    for rid in by_id:
        rationale = by_id[rid]["likely_ai"]["rationale"]
        assert rationale is not None
        assert len(rationale) > 0
