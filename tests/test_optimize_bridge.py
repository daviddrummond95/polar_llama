"""CI-safe tests for the local optimize.py bridge (no mlx / GPU required).

Only the pure-Python JSON-extraction logic and symbol wiring are exercised here;
the end-to-end path needs Apple GPU + mlx and is covered by benchmarks.
"""
import json

from polar_llama.local.optimize_bridge import _extract, _find_json_object


class TestFindJsonObject:
    def test_plain_object(self):
        assert _find_json_object('{"a": 1}') == {"a": 1}

    def test_object_amid_prose_and_fences(self):
        text = "Sure!\n```json\n{\"category\": \"Gemma\"}\n```\nHope that helps."
        assert _find_json_object(text) == {"category": "Gemma"}

    def test_no_object(self):
        assert _find_json_object("no json here") is None

    def test_non_object_json_ignored(self):
        # A bare array is not a dict -> not accepted as the object.
        assert _find_json_object("[1, 2, 3]") is None

    def test_brace_inside_string_value(self):
        # A '}' inside a string value must not terminate the object early.
        assert _find_json_object('{"answer": "a closing brace } here"}') == {
            "answer": "a closing brace } here"
        }

    def test_escaped_quote_inside_string(self):
        assert _find_json_object(r'{"answer": "she said \"hi} there\""}') == {
            "answer": 'she said "hi} there"'
        }


class TestExtract:
    def test_clean_json(self):
        assert json.loads(_extract('{"category": "MLX"}', ["category"])) == {"category": "MLX"}

    def test_fenced_and_prose(self):
        text = "Here you go:\n```json\n{\"category\": \"Gemma\"}\n```"
        assert json.loads(_extract(text, ["category"])) == {"category": "Gemma"}

    def test_list_valued_field_is_flattened(self):
        # optimize.py's proposer expects newline-delimited text, not a list.
        out = _extract('{"instructions": ["opt one", "opt two"]}', ["instructions"])
        assert json.loads(out) == {"instructions": "opt one\nopt two"}

    def test_single_field_fallback_wraps_first_line(self):
        out = _extract("MLX\n(the framework)", ["category"])
        assert json.loads(out) == {"category": "MLX"}

    def test_multi_field_without_json_returns_none(self):
        assert _extract("no json here", ["a", "b"]) is None


def test_symbol_is_lazily_importable_without_mlx():
    # Importing the symbol must not import mlx (guarded inside the factory).
    from polar_llama.local import make_local_inference_fn

    assert callable(make_local_inference_fn)
