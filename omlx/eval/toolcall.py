# SPDX-License-Identifier: Apache-2.0
"""ToolCall benchmark: tool-call correctness through oMLX's own parser.

Measures whether a model reliably emits tool calls that oMLX can parse
and that match the request. Outputs are scored on the *parsed* result
(engine-side ``GenerationOutput.tool_calls`` when set, otherwise
``omlx.api.tool_calling.parse_tool_calls``), not on raw text — a model
whose format this server's parser mishandles is, for routing purposes,
bad at tool calls on this server.

Four categories:
- ``simple``: one tool offered, one call expected
- ``multiple``: 2-4 tools offered, pick the right one
- ``parallel``: one prompt needs 2+ calls (any order)
- ``irrelevance``: tools offered, none applicable — correct means NO call

Dataset provenance: converted from the Berkeley Function-Calling
Leaderboard (BFCL) v4 data in github.com/ShishirPatil/gorilla
(Apache-2.0), commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8, fetched
2026-07-09. Source categories simple_python / multiple / parallel /
irrelevance, deterministically sampled (seed 42) to 120/80/40/60 = 300
items. Conversion: BFCL parameter schemas mapped to OpenAI
function-calling format (type dict->object, float->number,
list/tuple->array, recursively; non-standard "optional" keys stripped);
ground-truth args keep BFCL's list-of-acceptable-values shape; an
acceptable-values list containing "" marks the arg optional (moved to
"optional", accepted with any value); irrelevance items expect zero
calls.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..api.tool_calling import parse_tool_calls
from .base import THINKING_MAX_TOKENS, THINKING_MIN_TOKENS, BaseBenchmark
from .datasets import load_jsonl, stratified_sample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def _values_equal(parsed: Any, acceptable: Any) -> bool:
    """Compare a parsed arg value against one acceptable value.

    Normalization: strings are stripped (case kept); ints/floats compare
    numerically (1 == 1.0); bools exact; lists element-wise in order;
    dicts key-wise. A string that JSON-decodes to the acceptable value's
    type is decoded first (models often quote numbers).
    """
    # bool before int/float: bool is an int subclass in Python
    if isinstance(acceptable, bool):
        if isinstance(parsed, str):
            parsed = _try_json_decode(parsed)
        return isinstance(parsed, bool) and parsed == acceptable
    if isinstance(acceptable, (int, float)):
        if isinstance(parsed, str):
            parsed = _try_json_decode(parsed)
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            return False
        return float(parsed) == float(acceptable)
    if isinstance(acceptable, str):
        return isinstance(parsed, str) and parsed.strip() == acceptable.strip()
    if isinstance(acceptable, list):
        if isinstance(parsed, str):
            parsed = _try_json_decode(parsed)
        if not isinstance(parsed, list) or len(parsed) != len(acceptable):
            return False
        return all(_values_equal(p, a) for p, a in zip(parsed, acceptable))
    if isinstance(acceptable, dict):
        if isinstance(parsed, str):
            parsed = _try_json_decode(parsed)
        if not isinstance(parsed, dict) or set(parsed) != set(acceptable):
            return False
        return all(_values_equal(parsed[k], v) for k, v in acceptable.items())
    return bool(parsed == acceptable)  # None and anything else: exact


def _try_json_decode(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _call_matches(parsed: dict, expected: dict) -> bool:
    """One parsed call vs one expected call.

    Matches iff the function name is equal (exact), every required arg is
    present with an acceptable value, and no arg falls outside
    required + optional.
    """
    if parsed.get("name") != expected.get("name"):
        return False
    parsed_args = parsed.get("args") or {}
    if not isinstance(parsed_args, dict):
        return False
    required: dict = expected.get("args") or {}
    optional = set(expected.get("optional") or [])
    if set(parsed_args) - set(required) - optional:
        return False
    for arg_name, acceptable_values in required.items():
        if arg_name not in parsed_args:
            return False
        if not any(_values_equal(parsed_args[arg_name], v) for v in acceptable_values):
            return False
    return True


def _match_calls(parsed_calls: list[dict], expected_calls: list[dict]) -> bool:
    """Greedy 1:1 matching of parsed calls onto expected calls, order free.

    Correct iff every expected call is matched and no unmatched extra
    parsed calls remain. ``expected_calls == []`` (irrelevance) is correct
    iff zero calls were parsed.
    """
    unmatched = list(parsed_calls)
    for expected in expected_calls:
        for i, candidate in enumerate(unmatched):
            if _call_matches(candidate, expected):
                unmatched.pop(i)
                break
        else:
            return False
    return not unmatched


def _canonicalize_calls(calls: list[dict]) -> str:
    """Serialize parsed calls to a canonical JSON string ("[]" when none)."""
    return json.dumps(
        [{"name": c.get("name", ""), "args": c.get("args") or {}} for c in calls],
        sort_keys=True,
        ensure_ascii=False,
    )


def _decode_arguments(arguments: Any) -> dict:
    """Coerce a parser's arguments payload (dict or JSON string) to a dict."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        decoded = _try_json_decode(arguments)
        if isinstance(decoded, dict):
            return decoded
    return {}


class ToolCallBenchmark(BaseBenchmark):
    """Tool-call correctness scored through oMLX's own parsing path."""

    name = "toolcall"
    quick_size = 50

    def __init__(self) -> None:
        # Engines already warned about a tools-less chat template this run
        self._warned_template_engines: set[int] = set()

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load the bundled toolcall dataset, stratified across categories."""
        items = load_jsonl(DATA_DIR / "toolcall.jsonl")
        for item in items:
            # Derived display field: base run() reads item["answer"] for the
            # per-question "expected" column
            item["answer"] = json.dumps(
                item.get("expected") or [], sort_keys=True, ensure_ascii=False
            )
        logger.info(f"ToolCall: loaded {len(items)} questions")
        if sample_size == 0:
            return items
        return stratified_sample(items, sample_size, key="category")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        # No system prompt: the chat template injects tool instructions
        # from the tools kwarg; adding our own would double-instruct.
        return [{"role": "user", "content": item["question"]}]

    def get_max_tokens(self) -> int:
        return 512

    def get_category(self, item: dict) -> str | None:
        return str(item["category"])

    def get_question_text(self, item: dict) -> str:
        return str(item.get("question", ""))

    def extract_answer(self, response: str, item: dict) -> str:
        # _eval_single already canonicalized the parsed calls
        return response

    def check_answer(self, predicted: str, item: dict) -> bool:
        try:
            parsed_calls = json.loads(predicted)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(parsed_calls, list):
            return False
        return _match_calls(parsed_calls, item.get("expected") or [])

    def _warn_once_if_template_lacks_tools(self, engine: Any) -> None:
        key = id(engine)
        if key in self._warned_template_engines:
            return
        self._warned_template_engines.add(key)
        template = getattr(getattr(engine, "tokenizer", None), "chat_template", None)
        if isinstance(template, str) and "tool" not in template:
            logger.warning(
                "toolcall: model chat template does not appear to support "
                "tools; the model will likely score 0 on this benchmark"
            )

    def _parsed_calls(self, output: Any, engine: Any, item: dict) -> list[dict]:
        """Normalize parsed tool calls to [{"name": str, "args": dict}]."""
        engine_calls = getattr(output, "tool_calls", None)
        if engine_calls:
            # Engine-side parse (e.g. Harmony): list of OpenAI-shaped dicts
            normalized = []
            for tc in engine_calls:
                fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                if not isinstance(fn, dict):
                    continue
                normalized.append(
                    {
                        "name": fn.get("name", ""),
                        "args": _decode_arguments(fn.get("arguments", {})),
                    }
                )
            return normalized
        _, tool_calls = parse_tool_calls(output.text, engine.tokenizer, item["tools"])
        if not tool_calls:
            return []
        return [
            {
                "name": tc.function.name,
                "args": _decode_arguments(tc.function.arguments),
            }
            for tc in tool_calls
        ]

    async def _eval_single(
        self,
        engine: Any,
        item: dict,
        index: int,
        sampling_kwargs: dict | None = None,
        enable_thinking: bool = False,
    ) -> tuple[int, dict, str, str, str, dict[str, Any]]:
        """Evaluate one item, passing tools and scoring the parsed calls.

        Mirrors BaseBenchmark._eval_single's kwargs discipline exactly,
        adds tools= to the chat call, and returns the canonical JSON of
        the parsed calls in the response-text slot (raw model text stays
        in the raw slot for thinking auto-detection and debuggability).

        The trailing empty dict is the diagnostics slot upstream added in
        0.5.2; toolcall only runs against local engines, so it is always {}.
        """
        messages = self.format_prompt(item)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        kwargs = dict(sampling_kwargs or {})
        # Force benchmark-controlled params (override model settings)
        max_tokens = self.get_max_tokens()
        # Harmony models (gpt_oss) use analysis + final channels;
        # analysis can consume the entire budget before final is emitted
        if getattr(engine, "model_type", None) == "gpt_oss":
            max_tokens = max(max_tokens * 4, 8192)
        elif enable_thinking:
            max_tokens = min(max(max_tokens, THINKING_MIN_TOKENS), THINKING_MAX_TOKENS)
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
        kwargs["presence_penalty"] = 0.0
        kwargs["repetition_penalty"] = 1.0
        # Merge enable_thinking into any existing chat_template_kwargs
        ct_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        ct_kwargs["enable_thinking"] = enable_thinking
        kwargs["chat_template_kwargs"] = ct_kwargs
        self._warn_once_if_template_lacks_tools(engine)
        try:
            output = await engine.chat(
                messages=messages,
                tools=item["tools"],
                **kwargs,
            )
            raw_text = output.text
            canonical = _canonicalize_calls(self._parsed_calls(output, engine, item))
            return index, item, canonical, prompt_text, raw_text, {}
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            return index, item, "", prompt_text, "", {}
