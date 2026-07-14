# SPDX-License-Identifier: Apache-2.0
"""Tests for the multi-turn classification window (loop-state phase C).

Covers build_classify_window() itself, the profiler wiring (what text the
router engine actually sees with the window on/off), and the shadow-labeler
fallback on tool_result-only turns. No model files or real inference.
"""

import asyncio
from types import SimpleNamespace

from omlx.routing.service import RoutingService, build_classify_window
from omlx.settings import RoutingClassifyWindowSettings, RoutingSettings

ANALYSIS = (
    "Domain: Programming | Complexity: 2 | Math: False | Code: True | "
    "Route: small model | Justification: simple."
)


class _RecordingEngine:
    """Fake engine that records the classify prompt it was given."""

    def __init__(self, response_text: str = ANALYSIS) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=self.response_text)


def make_settings(tmp_path, *, window=None, shadow=False) -> RoutingSettings:
    data = {
        "enabled": True,
        "targets": {"small": "small-model", "big": "big-model"},
        "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
        "shadow_labeler": {"enabled": shadow},
    }
    if window is not None:
        data["classify_window"] = window
    return RoutingSettings.from_dict(data)


# ---------------------------------------------------------------------------
# build_classify_window
# ---------------------------------------------------------------------------


def test_window_single_user_message():
    out = build_classify_window(
        [{"role": "user", "content": "hello"}], max_turns=6, max_chars=4000
    )
    assert out == "User: hello"


def test_window_chronological_with_role_prefixes():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "fix the failing test"},
        {"role": "assistant", "content": "I updated the parser."},
        {"role": "user", "content": "make it faster"},
    ]
    out = build_classify_window(messages, max_turns=6, max_chars=4000)
    assert out == (
        "User: fix the failing test\n"
        "Assistant: I updated the parser.\n"
        "User: make it faster"
    )


def test_window_skips_tool_payloads_and_tool_role():
    messages = [
        {"role": "user", "content": "run the tests"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running them now."},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
            ],
        },
        {"role": "tool", "content": "3 passed"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "3 passed"}
            ],
        },
    ]
    out = build_classify_window(messages, max_turns=6, max_chars=4000)
    # The tool_result-only user turn contributes nothing; tool payloads and
    # role="tool" messages are excluded — but the earlier text survives.
    assert out == "User: run the tests\nAssistant: Running them now."
    assert "3 passed" not in out
    assert "tool_use" not in out


def test_window_max_turns_keeps_newest():
    messages = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    out = build_classify_window(messages, max_turns=2, max_chars=4000)
    assert out == "User: turn 8\nUser: turn 9"


def test_window_max_chars_drops_oldest_first():
    messages = [
        {"role": "user", "content": "x" * 200},
        {"role": "user", "content": "recent question"},
    ]
    out = build_classify_window(messages, max_turns=6, max_chars=50)
    assert out == "User: recent question"


def test_window_newest_always_included_even_over_budget():
    out = build_classify_window(
        [{"role": "user", "content": "y" * 300}], max_turns=6, max_chars=10
    )
    assert out.startswith("User: yyy")


def test_window_elides_long_messages():
    long_text = "a" * 5000
    out = build_classify_window(
        [{"role": "user", "content": long_text}], max_turns=6, max_chars=4000
    )
    assert "characters of pasted content elided" in out
    assert len(out) < 2000


def test_window_empty_messages():
    assert build_classify_window([], max_turns=6, max_chars=4000) == ""
    assert (
        build_classify_window(
            [{"role": "system", "content": "sys"}], max_turns=6, max_chars=4000
        )
        == ""
    )


def test_window_attribute_style_messages():
    messages = [
        SimpleNamespace(role="user", content="attr question"),
        SimpleNamespace(
            role="assistant",
            content=[SimpleNamespace(type="text", text="attr answer")],
        ),
    ]
    out = build_classify_window(messages, max_turns=6, max_chars=4000)
    assert out == "User: attr question\nAssistant: attr answer"


def test_window_clamps_bad_config():
    out = build_classify_window(
        [{"role": "user", "content": "hi"}], max_turns=0, max_chars=-5
    )
    assert out == "User: hi"


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------


def test_settings_default_off_round_trip():
    s = RoutingSettings()
    assert s.classify_window.enabled is False
    assert s.classify_window.max_turns == 6
    assert s.classify_window.max_chars == 4000
    assert s.classify_window.tier_from_newest is False
    d = s.to_dict()
    assert d["classify_window"] == {
        "enabled": False,
        "max_turns": 6,
        "max_chars": 4000,
        "tier_from_newest": False,
    }
    again = RoutingSettings.from_dict(d)
    assert again.classify_window == s.classify_window


def test_settings_from_dict_custom():
    w = RoutingClassifyWindowSettings.from_dict(
        {"enabled": True, "max_turns": 3, "max_chars": 1500, "tier_from_newest": True}
    )
    assert w == RoutingClassifyWindowSettings(
        enabled=True, max_turns=3, max_chars=1500, tier_from_newest=True
    )


def test_settings_missing_block_defaults():
    s = RoutingSettings.from_dict({"enabled": True})
    assert s.classify_window.enabled is False
    assert s.classify_window.tier_from_newest is False


def test_settings_tier_from_newest_round_trip():
    w = RoutingClassifyWindowSettings(tier_from_newest=True)
    d = w.to_dict()
    assert d["tier_from_newest"] is True
    assert RoutingClassifyWindowSettings.from_dict(d).tier_from_newest is True
    assert RoutingClassifyWindowSettings.from_dict({}).tier_from_newest is False


# ---------------------------------------------------------------------------
# Service wiring: what the profiler actually sees
# ---------------------------------------------------------------------------

MULTI_TURN = [
    {"role": "user", "content": "write a sort function"},
    {"role": "assistant", "content": "Here is quicksort."},
    {"role": "user", "content": "make it faster"},
]


async def test_profiler_sees_last_user_only_when_window_off(tmp_path):
    service = RoutingService(make_settings(tmp_path))
    engine = _RecordingEngine()

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 1
    assert "make it faster" in engine.prompts[0]
    assert "quicksort" not in engine.prompts[0]
    await service.close()


async def test_profiler_sees_window_when_enabled(tmp_path):
    service = RoutingService(make_settings(tmp_path, window={"enabled": True}))
    engine = _RecordingEngine()

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    prompt = engine.prompts[0]
    assert "User: write a sort function" in prompt
    assert "Assistant: Here is quicksort." in prompt
    assert "User: make it faster" in prompt
    await service.close()


async def test_window_falls_back_to_last_user_when_empty(tmp_path):
    service = RoutingService(make_settings(tmp_path, window={"enabled": True}))
    engine = _RecordingEngine()

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    # No user/assistant text anywhere: window is empty, falls back to
    # _last_user_content ("") — same input the profiler saw pre-window.
    await service.route_chat_request(
        messages=[{"role": "system", "content": "sys"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    assert engine.prompts == ["Task: \nAnalysis: "]
    await service.close()


# ---------------------------------------------------------------------------
# Shadow labeler: coverage gap on tool_result-only turns
# ---------------------------------------------------------------------------

TOOL_RESULT_TAIL = [
    {"role": "user", "content": "run the tests"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running them now."},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "3 passed"}
        ],
    },
]


async def test_shadow_labels_tool_result_turn_with_window(tmp_path):
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True}, shadow=True)
    )
    seen: list[str] = []

    async def fake_classify(text):
        seen.append(text)
        return {"provider": "apple_fm", "label": "SIMPLE", "reason": "r", "ms": 1.0}

    service._shadow.classify = fake_classify
    await service.route_chat_request(
        messages=TOOL_RESULT_TAIL, has_tools=True, request_id="r1", stream=False
    )
    await asyncio.gather(*service._shadow_tasks)
    assert seen == ["User: run the tests\nAssistant: Running them now."]
    await service.close()


async def test_shadow_keeps_last_user_text_when_present(tmp_path):
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True}, shadow=True)
    )
    seen: list[str] = []

    async def fake_classify(text):
        seen.append(text)
        return {"provider": "apple_fm", "label": "SIMPLE", "reason": "r", "ms": 1.0}

    service._shadow.classify = fake_classify
    engine = _RecordingEngine()

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    await asyncio.gather(*service._shadow_tasks)
    # Labels stay comparable with the pre-window corpus: last user text
    # is used as-is whenever it exists.
    assert seen == ["make it faster"]
    await service.close()


async def test_shadow_still_skips_when_window_off_and_no_text(tmp_path):
    service = RoutingService(make_settings(tmp_path, shadow=True))

    async def fake_classify(text):
        raise AssertionError("should not label a text-less turn with window off")

    service._shadow.classify = fake_classify
    await service.route_chat_request(
        messages=TOOL_RESULT_TAIL, has_tools=True, request_id="r1", stream=False
    )
    assert not service._shadow_tasks
    await service.close()


# ---------------------------------------------------------------------------
# #13a: tier_from_newest — window poisons complexity, so re-derive it from
# only the newest user text while axis (domain/math/code) keeps the window.
# ---------------------------------------------------------------------------

WINDOW_ANALYSIS = (
    "Domain: Programming | Complexity: 5 | Math: False | Code: True | "
    "Route: big model | Justification: whole transcript has code."
)
NEWEST_ANALYSIS = (
    "Domain: Programming | Complexity: 1 | Math: False | Code: True | "
    "Route: small model | Justification: just a confirmation."
)


class _SequencedEngine:
    """Fake engine returning one response per call, in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        idx = len(self.prompts) - 1
        text = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        return SimpleNamespace(text=text)


class _RaisingSecondEngine:
    """First call succeeds with the window analysis; second call raises."""

    def __init__(self, first_response: str) -> None:
        self.first_response = first_response
        self.prompts: list[str] = []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        if len(self.prompts) == 1:
            return SimpleNamespace(text=self.first_response)
        raise RuntimeError("boom")


async def test_tier_from_newest_off_is_single_classify(tmp_path):
    """Flag off: byte-identical to today — one classify call, window features."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": False})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    decision = await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 1
    assert decision.features.complexity == 5
    await service.close()


async def test_tier_from_newest_splits_when_newest_differs_from_window(tmp_path):
    """Flag on + window on + newest != window text: two classifies, merged
    features (complexity from newest, domain/math/code from window)."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS, NEWEST_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    decision = await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 2
    # First call sees the window (all three turns); second sees newest only.
    assert "write a sort function" in engine.prompts[0]
    assert "write a sort function" not in engine.prompts[1]
    assert "make it faster" in engine.prompts[1]
    # complexity comes from the newest-text classify (1), axis fields from
    # the window classify (domain/code from WINDOW_ANALYSIS, both agree here
    # but we assert on the actual merged object to be explicit).
    assert decision.features.complexity == 1
    assert decision.features.domain == "Programming"
    assert decision.features.code is True
    await service.close()


async def test_tier_from_newest_skips_second_call_when_newest_equals_window(
    tmp_path, monkeypatch
):
    """When the window text happens to equal the newest user text verbatim
    (e.g. window building degenerates to the bare last-user string), the
    guard must skip the redundant second classify."""
    import omlx.routing.service as service_mod

    monkeypatch.setattr(
        service_mod, "build_classify_window", lambda messages, **kw: "hello"
    )
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    assert len(engine.prompts) == 1
    await service.close()


# ---------------------------------------------------------------------------
# #13a2: tier source must be the newest HUMAN text — on agentic traffic the
# newest user-role message is a tool_result blob, so walk back to the newest
# user turn that carries genuine text (cc-live-test 2026-07-13).
# ---------------------------------------------------------------------------

# An agent loop: human task prompt, assistant tool call, then a user turn that
# is nothing but the tool's output. The naive "newest user message" is the
# tool_result blob (which flattens to ""); the tier classify must instead see
# the human's task prompt.
AGENTIC_TURN = [
    {"role": "user", "content": "fix the failing test in the bucket module"},
    {"role": "assistant", "content": "Let me run the suite."},
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "E   assert 4 == 5\n1 failed in 0.42s\n" + "x" * 400,
            }
        ],
    },
]


async def test_tier_from_newest_skips_tool_result_only_turn(tmp_path):
    """#13a2: the newest user turn is a tool_result blob → the second classify
    must run on the human task prompt, never on the tool output."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS, NEWEST_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    decision = await service.route_chat_request(
        messages=AGENTIC_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 2
    # Second classify sees the human ask, not the tool output or the
    # assistant turn.
    assert "fix the failing test" in engine.prompts[1]
    assert "1 failed" not in engine.prompts[1]
    assert "Let me run the suite" not in engine.prompts[1]
    # Complexity comes from the newest-human-text classify.
    assert decision.features.complexity == 1
    await service.close()


async def test_tier_from_newest_uses_text_block_in_mixed_turn(tmp_path):
    """A user turn carrying both a human text block and a tool_result: the
    tier classify sees the human text, never the tool_result content."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS, NEWEST_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    messages = [
        {"role": "user", "content": "add error handling"},
        {"role": "assistant", "content": "Done."},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "OK " * 200},
                {"type": "text", "text": "are you sure?"},
            ],
        },
    ]
    await service.route_chat_request(
        messages=messages, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 2
    assert "are you sure?" in engine.prompts[1]
    assert "OK OK" not in engine.prompts[1]
    await service.close()


async def test_tier_from_newest_no_human_text_keeps_window(tmp_path):
    """Pure tool_result conversation (no human text anywhere): the guard finds
    no newest-human text, so it never fires the second classify and keeps the
    window's complexity."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _SequencedEngine([WINDOW_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    decision = await service.route_chat_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t3", "content": "output"}
                ],
            }
        ],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    # No second classify; complexity stays the window's (5).
    assert len(engine.prompts) == 1
    assert decision.features.complexity == 5
    await service.close()


async def test_tier_from_newest_noop_when_window_disabled(tmp_path):
    """Flag on but window off: split is meaningless without a window, so
    behavior is a single classify on the last user message."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": False, "tier_from_newest": True})
    )
    engine = _SequencedEngine([NEWEST_ANALYSIS])

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 1
    await service.close()


async def test_tier_from_newest_second_classify_fails_open_to_window_complexity(
    tmp_path,
):
    """Second classify raising must not fail the route — keep the window's
    complexity and still route successfully."""
    service = RoutingService(
        make_settings(tmp_path, window={"enabled": True, "tier_from_newest": True})
    )
    engine = _RaisingSecondEngine(WINDOW_ANALYSIS)

    async def getter(model_id):
        return engine

    service.set_engine_getter(getter)
    decision = await service.route_chat_request(
        messages=MULTI_TURN, has_tools=False, request_id="r1", stream=False
    )
    assert len(engine.prompts) == 2
    assert decision.features.complexity == 5  # window's, not lost to fail-open
    assert decision.rule_fired != "error"  # route succeeded, not a global fail-open
    await service.close()
