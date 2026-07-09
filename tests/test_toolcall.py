# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.eval.toolcall (ToolCallBenchmark and its matchers)."""

import asyncio
import json

import pytest

from omlx.engine.base import GenerationOutput
from omlx.eval.toolcall import (
    DATA_DIR,
    ToolCallBenchmark,
    _call_matches,
    _canonicalize_calls,
    _decode_arguments,
    _match_calls,
    _values_equal,
)

# ---------------------------------------------------------------------------
# _values_equal
# ---------------------------------------------------------------------------


class TestValuesEqual:
    def test_string_stripped_case_kept(self):
        assert _values_equal("  Paris  ", "Paris")
        assert not _values_equal("paris", "Paris")

    def test_int_float_numeric(self):
        assert _values_equal(1, 1.0)
        assert _values_equal(1.0, 1)
        assert not _values_equal(2, 1)

    def test_bool_strictness(self):
        assert _values_equal(True, True)
        assert not _values_equal(True, 1)  # bool does not match int
        assert not _values_equal(1, True)

    def test_string_json_decodes_to_expected_type(self):
        assert _values_equal("2", 2)  # quoted number decodes
        assert _values_equal("1.0", 1)
        assert not _values_equal("Paris", 2)

    def test_list_order_sensitive(self):
        assert _values_equal(["a", "b"], ["a", "b"])
        assert not _values_equal(["b", "a"], ["a", "b"])
        assert not _values_equal(["a"], ["a", "b"])

    def test_nested_dict(self):
        assert _values_equal({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
        assert not _values_equal({"a": 1}, {"a": 1, "b": 2})

    def test_none_exact(self):
        assert _values_equal(None, None)
        assert not _values_equal(0, None)


# ---------------------------------------------------------------------------
# _call_matches (single call vs single expected)
# ---------------------------------------------------------------------------


class TestCallMatches:
    def test_simple_correct(self):
        parsed = {"name": "get_weather", "args": {"city": "Paris"}}
        expected = {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
        assert _call_matches(parsed, expected)

    def test_wrong_name(self):
        parsed = {"name": "get_time", "args": {"city": "Paris"}}
        expected = {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
        assert not _call_matches(parsed, expected)

    def test_arg_value_alternatives(self):
        expected = {
            "name": "get_weather",
            "args": {"city": ["Paris", "paris"]},
            "optional": [],
        }
        assert _call_matches(
            {"name": "get_weather", "args": {"city": "paris"}}, expected
        )
        assert _call_matches(
            {"name": "get_weather", "args": {"city": "Paris"}}, expected
        )
        assert not _call_matches(
            {"name": "get_weather", "args": {"city": "London"}}, expected
        )

    def test_optional_arg_any_value(self):
        expected = {
            "name": "get_weather",
            "args": {"city": ["Paris"]},
            "optional": ["unit"],
        }
        # optional present with an arbitrary value is fine
        assert _call_matches(
            {"name": "get_weather", "args": {"city": "Paris", "unit": "kelvin"}},
            expected,
        )
        # optional absent is also fine
        assert _call_matches(
            {"name": "get_weather", "args": {"city": "Paris"}}, expected
        )

    def test_unexpected_arg_rejected(self):
        expected = {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
        assert not _call_matches(
            {"name": "get_weather", "args": {"city": "Paris", "junk": 1}}, expected
        )

    def test_missing_required_arg_rejected(self):
        expected = {
            "name": "get_weather",
            "args": {"city": ["Paris"], "unit": ["celsius"]},
            "optional": [],
        }
        assert not _call_matches(
            {"name": "get_weather", "args": {"city": "Paris"}}, expected
        )

    def test_numeric_coercion_in_call(self):
        expected = {"name": "set", "args": {"n": [2]}, "optional": []}
        assert _call_matches({"name": "set", "args": {"n": "2"}}, expected)
        assert _call_matches({"name": "set", "args": {"n": 2.0}}, expected)


# ---------------------------------------------------------------------------
# _match_calls (multi-call set matching)
# ---------------------------------------------------------------------------


class TestMatchCalls:
    def test_simple_single(self):
        parsed = [{"name": "f", "args": {"x": 1}}]
        expected = [{"name": "f", "args": {"x": [1]}, "optional": []}]
        assert _match_calls(parsed, expected)

    def test_multiple_wrong_tool_name_rejected(self):
        # "multiple" category: right args but wrong tool selected
        parsed = [{"name": "get_time", "args": {"city": "Paris"}}]
        expected = [
            {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
        ]
        assert not _match_calls(parsed, expected)

    def test_parallel_any_order_accepted(self):
        parsed = [
            {"name": "weather", "args": {"city": "London"}},
            {"name": "weather", "args": {"city": "Paris"}},
        ]
        expected = [
            {"name": "weather", "args": {"city": ["Paris"]}, "optional": []},
            {"name": "weather", "args": {"city": ["London"]}, "optional": []},
        ]
        assert _match_calls(parsed, expected)

    def test_parallel_missing_one_rejected(self):
        parsed = [{"name": "weather", "args": {"city": "Paris"}}]
        expected = [
            {"name": "weather", "args": {"city": ["Paris"]}, "optional": []},
            {"name": "weather", "args": {"city": ["London"]}, "optional": []},
        ]
        assert not _match_calls(parsed, expected)

    def test_extra_parsed_call_rejected(self):
        parsed = [
            {"name": "f", "args": {"x": 1}},
            {"name": "g", "args": {}},
        ]
        expected = [{"name": "f", "args": {"x": [1]}, "optional": []}]
        assert not _match_calls(parsed, expected)

    def test_irrelevance_zero_calls_correct(self):
        assert _match_calls([], [])

    def test_irrelevance_with_a_call_incorrect(self):
        assert not _match_calls([{"name": "f", "args": {}}], [])


# ---------------------------------------------------------------------------
# _canonicalize_calls / _decode_arguments
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_canonicalize_empty(self):
        assert _canonicalize_calls([]) == "[]"

    def test_canonicalize_sorted_keys(self):
        out = _canonicalize_calls([{"name": "f", "args": {"b": 2, "a": 1}}])
        assert json.loads(out) == [{"name": "f", "args": {"a": 1, "b": 2}}]
        # deterministic: sorted keys
        assert _canonicalize_calls(
            [{"name": "f", "args": {"a": 1, "b": 2}}]
        ) == _canonicalize_calls([{"name": "f", "args": {"b": 2, "a": 1}}])

    def test_canonicalize_missing_fields(self):
        assert json.loads(_canonicalize_calls([{}])) == [{"name": "", "args": {}}]

    def test_decode_arguments_dict(self):
        assert _decode_arguments({"a": 1}) == {"a": 1}

    def test_decode_arguments_json_string(self):
        assert _decode_arguments('{"city": "Paris"}') == {"city": "Paris"}

    def test_decode_arguments_bad(self):
        assert _decode_arguments("not json") == {}
        assert _decode_arguments(None) == {}
        assert _decode_arguments("[1, 2]") == {}  # not an object


# ---------------------------------------------------------------------------
# check_answer / format_prompt / extract_answer
# ---------------------------------------------------------------------------


class TestCheckAnswer:
    def setup_method(self):
        self.bench = ToolCallBenchmark()

    def test_valid_canonical_roundtrip(self):
        item = {"expected": [{"name": "f", "args": {"x": [1]}, "optional": []}]}
        predicted = _canonicalize_calls([{"name": "f", "args": {"x": 1}}])
        assert self.bench.check_answer(predicted, item)

    def test_empty_string_false(self):
        item = {"expected": []}
        assert not self.bench.check_answer("", item)

    def test_non_list_json_false(self):
        item = {"expected": []}
        assert not self.bench.check_answer('{"name": "f"}', item)

    def test_irrelevance_empty_list_correct(self):
        item = {"expected": []}
        assert self.bench.check_answer("[]", item)

    def test_extract_answer_identity(self):
        assert self.bench.extract_answer("[]", {}) == "[]"


class TestFormatPrompt:
    def test_user_only_no_system(self):
        bench = ToolCallBenchmark()
        item = {"question": "What is the weather in Paris?"}
        assert bench.format_prompt(item) == [
            {"role": "user", "content": "What is the weather in Paris?"}
        ]


# ---------------------------------------------------------------------------
# load_dataset (stratification / determinism)
# ---------------------------------------------------------------------------


def _synthetic_dataset() -> list[dict]:
    """40 items, 10 per category, each self-consistent with a tool."""
    cats = ["simple", "multiple", "parallel", "irrelevance"]
    items = []
    for cat in cats:
        for i in range(10):
            tool = {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
            expected = (
                []
                if cat == "irrelevance"
                else [
                    {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
                ]
            )
            items.append(
                {
                    "id": f"{cat}_{i:03d}",
                    "category": cat,
                    "question": f"{cat} question {i}",
                    "tools": [tool],
                    "expected": expected,
                }
            )
    return items


class TestLoadDataset:
    def setup_method(self):
        self.bench = ToolCallBenchmark()

    def _patch(self, monkeypatch):
        data = _synthetic_dataset()
        monkeypatch.setattr(
            "omlx.eval.toolcall.load_jsonl", lambda path: [dict(d) for d in data]
        )
        return data

    async def test_full_dataset_when_size_zero(self, monkeypatch):
        self._patch(monkeypatch)
        items = await self.bench.load_dataset(sample_size=0)
        assert len(items) == 40
        assert all("answer" in it for it in items)

    async def test_answer_key_added(self, monkeypatch):
        self._patch(monkeypatch)
        items = await self.bench.load_dataset(sample_size=0)
        simple = next(it for it in items if it["category"] == "simple")
        assert json.loads(simple["answer"]) == simple["expected"]
        irrel = next(it for it in items if it["category"] == "irrelevance")
        assert irrel["answer"] == "[]"

    async def test_stratified_covers_every_category(self, monkeypatch):
        self._patch(monkeypatch)
        items = await self.bench.load_dataset(sample_size=12)
        assert len(items) <= 12
        cats = {it["category"] for it in items}
        assert cats == {"simple", "multiple", "parallel", "irrelevance"}

    async def test_deterministic_same_size_same_ids(self, monkeypatch):
        self._patch(monkeypatch)
        a = await self.bench.load_dataset(sample_size=12)
        b = await self.bench.load_dataset(sample_size=12)
        assert [it["id"] for it in a] == [it["id"] for it in b]


# ---------------------------------------------------------------------------
# _eval_single (stub engine)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer: no tool-parser attributes.

    parse_tool_calls checks getattr(tokenizer, "has_tool_calling", False),
    which is absent here, so the native parser is skipped and the generic
    XML fallback runs.
    """


class _StubEngine:
    """Records the last chat() kwargs and returns a fixed output."""

    def __init__(self, output=None, raise_exc=None, model_type="qwen"):
        self._output = output
        self._raise_exc = raise_exc
        self.model_type = model_type
        self.tokenizer = _StubTokenizer()
        self.last_kwargs = None

    async def chat(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._output


def _weather_item():
    return {
        "id": "simple_001",
        "category": "simple",
        "question": "Weather in Paris?",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "expected": [
            {"name": "get_weather", "args": {"city": ["Paris"]}, "optional": []}
        ],
    }


class TestEvalSingle:
    def setup_method(self):
        self.bench = ToolCallBenchmark()

    def test_engine_side_tool_calls(self):
        item = _weather_item()
        output = GenerationOutput(
            text="raw model text",
            tool_calls=[
                {
                    "id": "x",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )
        engine = _StubEngine(output=output)
        result = asyncio.run(self.bench._eval_single(engine, item, 3))
        index, ret_item, canonical, prompt_text, raw_text = result
        assert index == 3
        assert ret_item is item
        assert json.loads(canonical) == [
            {"name": "get_weather", "args": {"city": "Paris"}}
        ]
        assert raw_text == "raw model text"
        assert prompt_text == "Weather in Paris?"
        # the parsed canonical passes check_answer
        assert self.bench.check_answer(canonical, item)

    def test_xml_fallback_parse(self):
        item = _weather_item()
        output = GenerationOutput(
            text=(
                'Sure. <tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Paris"}}</tool_call>'
            ),
            tool_calls=None,
        )
        engine = _StubEngine(output=output)
        result = asyncio.run(self.bench._eval_single(engine, item, 0))
        _, _, canonical, _, raw_text = result
        assert json.loads(canonical) == [
            {"name": "get_weather", "args": {"city": "Paris"}}
        ]
        assert self.bench.check_answer(canonical, item)
        assert raw_text == output.text

    def test_engine_exception_returns_empty(self):
        item = _weather_item()
        engine = _StubEngine(raise_exc=RuntimeError("boom"))
        result = asyncio.run(self.bench._eval_single(engine, item, 7))
        index, ret_item, canonical, prompt_text, raw_text = result
        assert index == 7
        assert canonical == ""
        assert raw_text == ""
        assert prompt_text == "Weather in Paris?"

    def test_chat_kwargs_forced(self):
        item = _weather_item()
        output = GenerationOutput(text="", tool_calls=None)
        engine = _StubEngine(output=output)
        asyncio.run(self.bench._eval_single(engine, item, 0))
        kw = engine.last_kwargs
        assert kw["tools"] == item["tools"]
        assert kw["temperature"] == 0.0
        assert kw["presence_penalty"] == 0.0
        assert kw["repetition_penalty"] == 1.0
        assert kw["max_tokens"] == 512
        assert kw["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_benchmarks():
    from omlx.eval import BENCHMARKS

    assert BENCHMARKS["toolcall"] is ToolCallBenchmark


def test_registry_valid_benchmarks():
    from omlx.admin.accuracy_benchmark import VALID_BENCHMARKS

    assert "toolcall" in VALID_BENCHMARKS


def test_registry_category_axis():
    from omlx.routing.store import CATEGORY_AXES

    assert CATEGORY_AXES["toolcall"] == "agentic"


# ---------------------------------------------------------------------------
# Bundled dataset sanity (skipped until the dataset lands)
# ---------------------------------------------------------------------------

_DATASET_PATH = DATA_DIR / "toolcall.jsonl"


@pytest.mark.skipif(not _DATASET_PATH.exists(), reason="dataset not yet bundled")
def test_bundled_dataset_valid():
    from omlx.eval.datasets import load_jsonl

    items = load_jsonl(_DATASET_PATH)
    assert items, "dataset is empty"
    valid_cats = {"simple", "multiple", "parallel", "irrelevance"}

    seen_ids = set()
    for it in items:
        item_id = it["id"]
        assert item_id not in seen_ids, f"duplicate id {item_id}"
        seen_ids.add(item_id)

        assert it["category"] in valid_cats, f"{item_id}: bad category"

        tools = it["tools"]
        assert isinstance(tools, list) and tools, f"{item_id}: tools must be non-empty"
        tool_names = set()
        for tool in tools:
            assert tool.get("type") == "function", f"{item_id}: tool not a function"
            fn = tool["function"]
            assert isinstance(fn, dict) and fn.get("name"), f"{item_id}: bad function"
            tool_names.add(fn["name"])

        assert isinstance(it["expected"], list), f"{item_id}: expected not a list"
        for call in it["expected"]:
            assert call["name"] in tool_names, f"{item_id}: call names unoffered tool"
            offered = next(t for t in tools if t["function"]["name"] == call["name"])
            props = offered["function"].get("parameters", {}).get("properties", {})
            for arg_name, values in call.get("args", {}).items():
                assert arg_name in props, f"{item_id}: arg {arg_name} not in schema"
                assert (
                    isinstance(values, list) and values
                ), f"{item_id}: arg {arg_name} values must be a non-empty list"
