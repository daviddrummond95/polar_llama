"""Tests for the DSPy-style prompt optimization engine.

These tests use an injected ``inference_fn`` so they run without API keys
or network access.
"""

import json

import polars as pl
import pytest

from polar_llama import (
    BootstrapFewShot,
    InstructionOptimizer,
    OutputField,
    Predict,
    Signature,
    evaluate,
)


def make_echo_backend(answers):
    """Inference backend that returns canned answers in order, cycling."""

    calls = []

    def backend(messages, output_model):
        calls.append(messages)
        return [json.dumps({"answer": answers[i % len(answers)]}) for i in range(len(messages))]

    backend.calls = calls
    return backend


def uppercase_backend(messages, output_model):
    """Deterministic backend: answers with the uppercased question."""
    out = []
    for raw in messages:
        convo = json.loads(raw)
        user = [m for m in convo if m["role"] == "user"][-1]
        question = user["content"].split("question: ", 1)[-1]
        out.append(json.dumps({"answer": question.upper()}))
    return out


def exact_match(example, prediction):
    return float(example["answer"] == prediction["answer"])


class TestSignature:
    def test_shorthand_parsing(self):
        sig = Signature("context, question -> answer")
        assert list(sig.inputs) == ["context", "question"]
        assert list(sig.outputs) == ["answer"]

    def test_invalid_shorthand(self):
        with pytest.raises(ValueError):
            Signature("no arrow here")

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            Signature(" -> answer")

    def test_system_prompt_contains_fields_and_instructions(self):
        sig = Signature(
            "question -> answer",
            instructions="Be terse.",
            outputs={"answer": OutputField(desc="the final answer")},
        )
        prompt = sig.system_prompt()
        assert "Be terse." in prompt
        assert "question" in prompt
        assert "the final answer" in prompt
        assert "JSON" in prompt

    def test_with_instructions_returns_new_signature(self):
        sig = Signature("question -> answer", instructions="old")
        new = sig.with_instructions("new")
        assert sig.instructions == "old"
        assert new.instructions == "new"
        assert new.inputs.keys() == sig.inputs.keys()

    def test_output_model_is_pydantic(self):
        sig = Signature("question -> answer, confidence",
                        outputs={"confidence": OutputField(dtype=float)})
        model = sig.output_model()
        instance = model(answer="x", confidence=0.5)
        assert instance.answer == "x"
        assert instance.confidence == 0.5


class TestPredict:
    def test_predict_adds_prediction_columns(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        df = pl.DataFrame({"question": ["abc", "def"]})
        result = module(df)
        assert result.get_column("pred_answer").to_list() == ["ABC", "DEF"]
        # Original columns preserved
        assert result.get_column("question").to_list() == ["abc", "def"]

    def test_demos_are_included_in_messages(self):
        backend = make_echo_backend(["x"])
        module = Predict("question -> answer", inference_fn=backend).with_demos(
            [{"question": "1+1?", "answer": "2"}]
        )
        module(pl.DataFrame({"question": ["2+2?"]}))

        convo = json.loads(backend.calls[0][0])
        roles = [m["role"] for m in convo]
        assert roles == ["system", "user", "assistant", "user"]
        assert json.loads(convo[2]["content"]) == {"answer": "2"}

    def test_accepts_signature_shorthand(self):
        module = Predict("question -> answer", inference_fn=make_echo_backend(["ok"]))
        assert isinstance(module.signature, Signature)

    def test_handles_invalid_json_responses(self):
        def bad_backend(messages, output_model):
            return ["not json"] * len(messages)

        module = Predict("question -> answer", inference_fn=bad_backend)
        result = module(pl.DataFrame({"question": ["q"]}))
        assert result.get_column("pred_answer").to_list() == [None]

    def test_handles_none_responses(self):
        def none_backend(messages, output_model):
            return [None] * len(messages)

        module = Predict("question -> answer", inference_fn=none_backend)
        result = module(pl.DataFrame({"question": ["q"]}))
        assert result.get_column("pred_answer").to_list() == [None]


class TestEvaluate:
    def test_perfect_score(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        dataset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})
        result = evaluate(module, dataset, exact_match)
        assert result.score == 1.0

    def test_partial_score(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        dataset = pl.DataFrame(
            {"question": ["ab", "cd"], "answer": ["AB", "wrong"]}
        )
        result = evaluate(module, dataset, exact_match)
        assert result.score == 0.5
        assert result.scores == (1.0, 0.0)

    def test_failed_predictions_score_zero(self):
        def none_backend(messages, output_model):
            return [None] * len(messages)

        module = Predict("question -> answer", inference_fn=none_backend)
        dataset = pl.DataFrame({"question": ["q"], "answer": ["a"]})
        result = evaluate(module, dataset, exact_match)
        assert result.score == 0.0

    def test_metric_exception_scores_zero(self):
        def broken_metric(example, prediction):
            raise RuntimeError("boom")

        module = Predict("question -> answer", inference_fn=uppercase_backend)
        dataset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})
        result = evaluate(module, dataset, broken_metric)
        assert result.score == 0.0


class TestBootstrapFewShot:
    def test_collects_passing_demos(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        trainset = pl.DataFrame(
            {"question": ["ab", "cd", "ef"], "answer": ["AB", "nope", "EF"]}
        )
        compiled = BootstrapFewShot(metric=exact_match, max_demos=4).compile(module, trainset)
        # Only the two rows the module got right become demos
        assert len(compiled.demos) == 2
        assert compiled.demos[0] == {"question": "ab", "answer": "AB"}
        assert compiled.demos[1] == {"question": "ef", "answer": "EF"}

    def test_max_demos_cap(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        trainset = pl.DataFrame(
            {"question": ["a", "b", "c"], "answer": ["A", "B", "C"]}
        )
        compiled = BootstrapFewShot(metric=exact_match, max_demos=1).compile(module, trainset)
        assert len(compiled.demos) == 1

    def test_gold_outputs_option(self):
        # Backend always answers "X"; with use_gold_outputs and threshold 0.0
        # the demos carry the gold labels rather than the predictions.
        module = Predict("question -> answer", inference_fn=make_echo_backend(["X"]))
        trainset = pl.DataFrame({"question": ["q1"], "answer": ["gold"]})
        compiled = BootstrapFewShot(
            metric=exact_match, max_demos=1, threshold=0.0, use_gold_outputs=True
        ).compile(module, trainset)
        assert compiled.demos[0]["answer"] == "gold"

    def test_original_module_unchanged(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        trainset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})
        BootstrapFewShot(metric=exact_match).compile(module, trainset)
        assert module.demos == ()


class TestInstructionOptimizer:
    def test_keeps_best_candidate(self):
        # Backend rewards the instruction "SHOUT": it only answers correctly
        # when the system prompt contains it.
        def instruction_sensitive_backend(messages, output_model):
            out = []
            for raw in messages:
                convo = json.loads(raw)
                system = convo[0]["content"]
                question = convo[-1]["content"].split("question: ", 1)[-1]
                if "SHOUT" in system:
                    out.append(json.dumps({"answer": question.upper()}))
                else:
                    out.append(json.dumps({"answer": question}))
            return out

        module = Predict(
            "question -> answer",
            inference_fn=instruction_sensitive_backend,
        ).with_instructions("answer politely")

        trainset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})

        optimizer = InstructionOptimizer(
            metric=exact_match,
            proposer_fn=lambda prompt: ["please SHOUT the answer", "whisper it"],
        )
        compiled = optimizer.compile(module, trainset)

        assert "SHOUT" in compiled.signature.instructions
        assert evaluate(compiled, trainset, exact_match).score == 1.0
        # The trace records every evaluated candidate
        assert len(optimizer.history) == 3

    def test_keeps_original_when_no_candidate_improves(self):
        module = Predict("question -> answer", inference_fn=uppercase_backend)
        trainset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})

        optimizer = InstructionOptimizer(
            metric=exact_match,
            proposer_fn=lambda prompt: ["candidate one", "candidate two"],
        )
        compiled = optimizer.compile(module, trainset)
        assert compiled.signature.instructions == module.signature.instructions

    def test_llm_proposer_returns_json_list(self):
        # Regression: when the proposer LLM returns the `instructions` output
        # field as a JSON *array* (common with small local models), the
        # prediction lands as a polars List cell. The default `_propose` path
        # must flatten it instead of crashing on `if not raw` (Series truth
        # value is ambiguous).
        def listy_backend(messages, output_model):
            fields = list(output_model.model_fields)
            if "instructions" in fields:
                # proposer call: return a JSON list, not a newline string
                return [
                    json.dumps({"instructions": ["please SHOUT the answer", "whisper it"]})
                    for _ in messages
                ]
            out = []
            for raw in messages:
                convo = json.loads(raw)
                system = convo[0]["content"]
                question = convo[-1]["content"].split("question: ", 1)[-1]
                out.append(
                    json.dumps({"answer": question.upper() if "SHOUT" in system else question})
                )
            return out

        module = Predict("question -> answer", inference_fn=listy_backend).with_instructions(
            "answer politely"
        )
        trainset = pl.DataFrame({"question": ["ab"], "answer": ["AB"]})

        # No proposer_fn -> exercises the default LLM `_propose` path.
        optimizer = InstructionOptimizer(metric=exact_match, n_candidates=2)
        compiled = optimizer.compile(module, trainset)  # must not raise

        assert "SHOUT" in compiled.signature.instructions
        assert evaluate(compiled, trainset, exact_match).score == 1.0
