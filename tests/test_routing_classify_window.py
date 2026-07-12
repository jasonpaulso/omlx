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
    d = s.to_dict()
    assert d["classify_window"] == {
        "enabled": False,
        "max_turns": 6,
        "max_chars": 4000,
    }
    again = RoutingSettings.from_dict(d)
    assert again.classify_window == s.classify_window


def test_settings_from_dict_custom():
    w = RoutingClassifyWindowSettings.from_dict(
        {"enabled": True, "max_turns": 3, "max_chars": 1500}
    )
    assert w == RoutingClassifyWindowSettings(enabled=True, max_turns=3, max_chars=1500)


def test_settings_missing_block_defaults():
    s = RoutingSettings.from_dict({"enabled": True})
    assert s.classify_window.enabled is False


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
