"""
DSPy-style prompt optimization engine for polar-llama.

This module provides a small, declarative framework for building and
optimizing LLM programs over Polars DataFrames, inspired by DSPy:

- ``Signature`` declares a task: typed input fields -> typed output fields,
  plus natural-language instructions.
- ``Predict`` is a module that executes a signature against an LLM using
  polar-llama's parallel inference (one batched, concurrent call per
  DataFrame instead of per row).
- ``evaluate`` scores a module against a labeled DataFrame with a metric.
- ``BootstrapFewShot`` improves a module by mining few-shot demonstrations
  from training rows the module already gets right (akin to
  ``dspy.BootstrapFewShot``).
- ``InstructionOptimizer`` asks an LLM to propose improved instruction
  variants and keeps the best-scoring one (akin to ``dspy.COPRO``).

Example
-------
>>> import polars as pl
>>> from polar_llama import optimize as po
>>>
>>> sig = po.Signature(
...     "question -> answer",
...     instructions="Answer the question concisely.",
... )
>>> module = po.Predict(sig, provider="openai", model="gpt-4o-mini")
>>>
>>> trainset = pl.DataFrame({
...     "question": ["What is 2+2?", "Capital of France?"],
...     "answer": ["4", "Paris"],
... })
>>>
>>> def exact_match(example, prediction):
...     return float(example["answer"].strip().lower()
...                  == (prediction["answer"] or "").strip().lower())
>>>
>>> optimizer = po.BootstrapFewShot(metric=exact_match, max_demos=2)
>>> compiled = optimizer.compile(module, trainset)
>>> result = compiled(pl.DataFrame({"question": ["What is 3+3?"]}))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import polars as pl

__all__ = [
    "InputField",
    "OutputField",
    "Signature",
    "Predict",
    "Evaluation",
    "evaluate",
    "BootstrapFewShot",
    "InstructionOptimizer",
]

# Type of a metric: (gold example row, prediction row) -> score.
# Booleans are coerced to 0.0/1.0.
Metric = Callable[[Dict[str, Any], Dict[str, Any]], Union[bool, float]]

# Pluggable inference backend, mainly for testing: receives the list of
# JSON-encoded message arrays (one per row) and the Pydantic output model,
# and returns one JSON string (or None) per row.
InferenceFn = Callable[[List[str], Type], List[Optional[str]]]

_PYTHON_TYPES: Dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


# ============================================================================
# Fields and Signatures
# ============================================================================

@dataclass(frozen=True)
class _Field:
    desc: str = ""
    dtype: type = str


class InputField(_Field):
    """An input field of a Signature."""


class OutputField(_Field):
    """An output field of a Signature."""


@dataclass(frozen=True)
class Signature:
    """
    A declarative task specification: inputs -> outputs with instructions.

    Can be created from a DSPy-style shorthand string::

        Signature("question -> answer")
        Signature("context, question -> answer", instructions="...")

    or with explicit fields::

        Signature(
            instructions="Classify the sentiment.",
            inputs={"text": InputField(desc="The document to classify")},
            outputs={"sentiment": OutputField(desc="positive, negative or neutral")},
        )
    """

    shorthand: Optional[str] = None
    instructions: str = ""
    inputs: Dict[str, InputField] = field(default_factory=dict)
    outputs: Dict[str, OutputField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shorthand:
            if "->" not in self.shorthand:
                raise ValueError(
                    f"Signature shorthand must look like 'a, b -> c', got: {self.shorthand!r}"
                )
            lhs, rhs = self.shorthand.split("->", 1)
            inputs = {name.strip(): InputField() for name in lhs.split(",") if name.strip()}
            outputs = {name.strip(): OutputField() for name in rhs.split(",") if name.strip()}
            # Explicit fields take precedence over shorthand-derived ones
            inputs.update(self.inputs)
            outputs.update(self.outputs)
            object.__setattr__(self, "inputs", inputs)
            object.__setattr__(self, "outputs", outputs)

        if not self.inputs or not self.outputs:
            raise ValueError("A Signature requires at least one input and one output field")

    def with_instructions(self, instructions: str) -> "Signature":
        """Return a copy of this signature with different instructions."""
        return Signature(
            instructions=instructions,
            inputs=dict(self.inputs),
            outputs=dict(self.outputs),
        )

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    def output_model(self) -> Type:
        """Build a Pydantic model matching the output fields (for structured outputs)."""
        from pydantic import Field as PydField, create_model

        fields = {
            name: (spec.dtype, PydField(..., description=spec.desc or name))
            for name, spec in self.outputs.items()
        }
        return create_model("SignatureOutput", **fields)

    def system_prompt(self) -> str:
        parts: List[str] = []
        if self.instructions:
            parts.append(self.instructions.strip())
        parts.append("You will be given input fields and must produce the output fields.")

        parts.append("\nInput fields:")
        for name, spec in self.inputs.items():
            parts.append(f"- {name}: {spec.desc}" if spec.desc else f"- {name}")

        parts.append("\nOutput fields:")
        for name, spec in self.outputs.items():
            type_name = _PYTHON_TYPES.get(spec.dtype, "string")
            desc = f" — {spec.desc}" if spec.desc else ""
            parts.append(f"- {name} ({type_name}){desc}")

        keys = ", ".join(self.outputs)
        parts.append(f"\nRespond with a JSON object containing exactly these keys: {keys}.")
        return "\n".join(parts)

    def render_inputs(self, row: Dict[str, Any]) -> str:
        """Render one example's input fields as the user message."""
        return "\n\n".join(f"{name}: {row[name]}" for name in self.inputs)

    def render_outputs(self, row: Dict[str, Any]) -> str:
        """Render one example's output fields as a JSON assistant message."""
        return json.dumps({name: row.get(name) for name in self.outputs})


# ============================================================================
# Predict module
# ============================================================================

@dataclass(frozen=True)
class Predict:
    """
    An executable LLM module for a Signature.

    Calling the module on a DataFrame runs one parallel, batched inference
    over all rows and returns the DataFrame with one ``pred_<field>`` column
    per output field.

    Parameters
    ----------
    signature : Signature or str
        The task specification (a shorthand string like ``"question -> answer"``
        is accepted).
    provider : str, optional
        polar-llama provider name (openai, anthropic, gemini, groq, bedrock).
    model : str, optional
        Model name for the provider.
    demos : sequence of dict, optional
        Few-shot demonstrations; each dict must contain the signature's
        input and output fields. Usually produced by an optimizer.
    inference_fn : callable, optional
        Override the inference backend (used for testing). Receives the
        JSON message arrays and the Pydantic output model, returns one JSON
        string (or None) per row.
    """

    signature: Signature
    provider: Optional[str] = None
    model: Optional[str] = None
    demos: Tuple[Dict[str, Any], ...] = ()
    inference_fn: Optional[InferenceFn] = None

    def __post_init__(self) -> None:
        if isinstance(self.signature, str):
            object.__setattr__(self, "signature", Signature(self.signature))
        object.__setattr__(self, "demos", tuple(self.demos))

    # ------------------------------------------------------------------
    # Derived modules
    # ------------------------------------------------------------------

    def with_instructions(self, instructions: str) -> "Predict":
        return replace(self, signature=self.signature.with_instructions(instructions))

    def with_demos(self, demos: Sequence[Dict[str, Any]]) -> "Predict":
        return replace(self, demos=tuple(demos))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def prediction_columns(self) -> Dict[str, str]:
        """Mapping of output field name -> prediction column name."""
        return {name: f"pred_{name}" for name in self.signature.outputs}

    def _build_messages(self, df: pl.DataFrame) -> List[str]:
        """Build the JSON message array for every row."""
        system = self.signature.system_prompt()

        demo_turns: List[Tuple[str, str]] = []
        for demo in self.demos:
            demo_turns.append(("user", self.signature.render_inputs(demo)))
            demo_turns.append(("assistant", self.signature.render_outputs(demo)))

        messages: List[str] = []
        for row in df.iter_rows(named=True):
            convo = [{"role": "system", "content": system}]
            convo.extend({"role": role, "content": content} for role, content in demo_turns)
            convo.append({"role": "user", "content": self.signature.render_inputs(row)})
            messages.append(json.dumps(convo))
        return messages

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run the module over a DataFrame, adding ``pred_<field>`` columns."""
        output_model = self.signature.output_model()
        messages = self._build_messages(df)

        if self.inference_fn is not None:
            raw = self.inference_fn(messages, output_model)
        else:
            raw = self._run_inference(messages, output_model)

        # Parse JSON responses into per-field prediction columns
        columns: Dict[str, List[Any]] = {col: [] for col in self.prediction_columns().values()}
        for response in raw:
            parsed: Dict[str, Any] = {}
            if response:
                try:
                    value = json.loads(response)
                    if isinstance(value, dict):
                        parsed = value
                except (TypeError, ValueError):
                    parsed = {}
            for name, col in self.prediction_columns().items():
                columns[col].append(parsed.get(name))

        height = df.height
        prediction_df = pl.DataFrame(
            {col: values + [None] * (height - len(values)) for col, values in columns.items()}
        )
        return pl.concat([df, prediction_df], how="horizontal")

    def _run_inference(self, messages: List[str], output_model: Type) -> List[Optional[str]]:
        # Imported lazily to avoid a circular import with the package root.
        from polar_llama import inference_messages

        frame = pl.DataFrame({"_messages": messages})
        # Request the raw JSON string (no response_model) so parsing stays in
        # one place; the schema is still enforced server-side via the
        # response_schema kwarg by passing the model through.
        result = frame.with_columns(
            _response=inference_messages(
                pl.col("_messages"),
                provider=self.provider,
                model=self.model,
                response_model=output_model,
            )
        )
        response = result.get_column("_response")
        if isinstance(response.dtype, pl.Struct):
            # Structured output path: re-serialize the struct rows to JSON
            return [
                json.dumps(row) if row is not None else None
                for row in response.to_list()
            ]
        return response.to_list()


# ============================================================================
# Evaluation
# ============================================================================

@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating a module against a labeled DataFrame."""

    score: float
    scores: Tuple[float, ...]
    predictions: pl.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Evaluation(score={self.score:.4f}, n={len(self.scores)})"


def evaluate(module: Predict, dataset: pl.DataFrame, metric: Metric) -> Evaluation:
    """
    Run ``module`` over ``dataset`` and score each row with ``metric``.

    The metric receives the gold row (all original columns) and the
    prediction (plain output-field names) and returns a bool or float.
    Rows whose prediction failed entirely score 0.0.
    """
    predictions = module(dataset)
    columns = module.prediction_columns()

    scores: List[float] = []
    for row in predictions.iter_rows(named=True):
        prediction = {name: row.get(col) for name, col in columns.items()}
        if all(value is None for value in prediction.values()):
            scores.append(0.0)
            continue
        try:
            scores.append(float(metric(row, prediction)))
        except Exception:  # noqa: BLE001 - a failing metric scores zero
            scores.append(0.0)

    score = sum(scores) / len(scores) if scores else 0.0
    return Evaluation(score=score, scores=tuple(scores), predictions=predictions)


# ============================================================================
# Optimizers
# ============================================================================

@dataclass
class BootstrapFewShot:
    """
    Bootstrap few-shot demonstrations from a training DataFrame.

    Runs the module over the trainset and keeps up to ``max_demos``
    examples whose predictions pass the metric (score >= ``threshold``)
    as demonstrations, mirroring ``dspy.BootstrapFewShot``.

    With ``use_gold_outputs=True``, the gold labels are used as the
    demonstration outputs instead of the module's own (passing) predictions
    (similar to ``dspy.LabeledFewShot``, but still filtered by the metric).
    """

    metric: Metric
    max_demos: int = 4
    threshold: float = 1.0
    use_gold_outputs: bool = False

    def compile(self, module: Predict, trainset: pl.DataFrame) -> Predict:
        evaluation = evaluate(module, trainset, self.metric)
        columns = module.prediction_columns()

        demos: List[Dict[str, Any]] = []
        for row, score in zip(evaluation.predictions.iter_rows(named=True), evaluation.scores):
            if score < self.threshold:
                continue
            demo = {name: row[name] for name in module.signature.inputs}
            if self.use_gold_outputs:
                demo.update({name: row[name] for name in module.signature.outputs})
            else:
                demo.update({name: row[col] for name, col in columns.items()})
            demos.append(demo)
            if len(demos) >= self.max_demos:
                break

        return module.with_demos(demos)


_PROPOSER_TEMPLATE = """\
You are an expert prompt engineer. A language model performs the task below.

Input fields: {inputs}
Output fields: {outputs}

Current instruction:
\"\"\"{instructions}\"\"\"

Current score on a held-out training set: {score:.3f} (1.0 is perfect).

Propose {n} alternative instructions that would make the language model more
accurate at this task. Each proposal must be self-contained, specific, and
phrased as a direct instruction to the model. Vary tone, level of detail,
and strategy across proposals."""


@dataclass
class InstructionOptimizer:
    """
    COPRO-style instruction search.

    Asks an LLM to propose ``n_candidates`` rewritten instructions for the
    module's signature, evaluates every candidate (plus the original) on the
    trainset, and returns the module with the best-scoring instructions.

    Parameters
    ----------
    metric : callable
        Scoring function, as for :func:`evaluate`.
    n_candidates : int
        Number of instruction rewrites to request per round.
    rounds : int
        Optimization rounds; each round proposes candidates derived from the
        current best instructions.
    provider, model : str, optional
        LLM used for proposing instructions. Defaults to the module's own
        provider/model.
    proposer_fn : callable, optional
        Override candidate generation (used for testing): receives the
        meta-prompt string and returns a list of candidate instructions.
    """

    metric: Metric
    n_candidates: int = 4
    rounds: int = 1
    provider: Optional[str] = None
    model: Optional[str] = None
    proposer_fn: Optional[Callable[[str], List[str]]] = None

    def compile(self, module: Predict, trainset: pl.DataFrame) -> Predict:
        best_module = module
        best = evaluate(module, trainset, self.metric)
        history: List[Tuple[str, float]] = [(module.signature.instructions, best.score)]

        for _ in range(max(1, self.rounds)):
            prompt = _PROPOSER_TEMPLATE.format(
                inputs=", ".join(best_module.signature.inputs),
                outputs=", ".join(best_module.signature.outputs),
                instructions=best_module.signature.instructions or "(no instructions)",
                score=best.score,
                n=self.n_candidates,
            )

            for candidate in self._propose(module, prompt):
                candidate = candidate.strip()
                if not candidate or candidate == best_module.signature.instructions:
                    continue
                candidate_module = best_module.with_instructions(candidate)
                result = evaluate(candidate_module, trainset, self.metric)
                history.append((candidate, result.score))
                if result.score > best.score:
                    best = result
                    best_module = candidate_module

        # Expose the search trace for inspection
        self.history = history
        return best_module

    def _propose(self, module: Predict, prompt: str) -> List[str]:
        if self.proposer_fn is not None:
            return self.proposer_fn(prompt)

        proposer = Predict(
            Signature(
                "task_description -> instructions",
                instructions=(
                    "Rewrite prompts to maximize task accuracy. Return the "
                    "requested number of alternative instructions."
                ),
                outputs={
                    "instructions": OutputField(
                        desc="One proposed instruction per line, no numbering"
                    )
                },
            ),
            provider=self.provider or module.provider,
            model=self.model or module.model,
            inference_fn=module.inference_fn,
        )
        result = proposer(pl.DataFrame({"task_description": [prompt]}))
        raw = result.get_column("pred_instructions")[0]
        # A model may return the instructions field as a JSON array rather than
        # a newline-delimited string. That lands here as a polars Series (a
        # List-typed cell) or a Python list; flatten either to newline-separated
        # text before splitting. (Indexing a List column returns a Series, whose
        # truth value is ambiguous -- so this must run before any `if not raw`.)
        if isinstance(raw, pl.Series):
            raw = "\n".join(str(x) for x in raw.to_list())
        elif isinstance(raw, (list, tuple)):
            raw = "\n".join(str(x) for x in raw)
        if raw is None or not str(raw).strip():
            return []
        return [line.strip("-• \t") for line in str(raw).splitlines() if line.strip()]
