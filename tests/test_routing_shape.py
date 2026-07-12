# SPDX-License-Identifier: Apache-2.0
"""Tests for the shape-based modality pre-route (M4.1 + part-type fix).

The part-type classification exists because of production evidence
(2026-07-11): Anthropic-protocol tool_use/tool_result/thinking blocks
tripped the old any-non-text check and routed 95% of a real Claude Code
session to the vision target, bypassing the classifier entirely.
"""

from types import SimpleNamespace

import pytest

from omlx.api.anthropic_models import (
    AnthropicMessage,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
)
from omlx.routing.service import RoutingService, _detect_modality
from omlx.settings import RoutingSettings


class FakeEngine:
    async def generate(self, **kwargs):
        return SimpleNamespace(
            text=(
                "Domain: general | Complexity: 1 | Math: False | Code: False "
                "| Route: small model"
            )
        )


def make_service(tmp_path, *, vision=None, audio=None):
    targets = {"small": "small-model", "big": "big-model"}
    if vision:
        targets["vision"] = vision
    if audio:
        targets["audio"] = audio
    svc = RoutingService(
        RoutingSettings.from_dict(
            {
                "enabled": True,
                "targets": targets,
                "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
            }
        )
    )

    async def getter(model_id):
        return FakeEngine()

    svc.set_engine_getter(getter)
    return svc


IMAGE_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
    ],
}

AUDIO_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Transcribe this."},
        {"type": "input_audio", "input_audio": {"data": "xx", "format": "wav"}},
    ],
}


def agent_messages():
    """A conversation shaped like the captured 2026-07-11 Claude Code
    session: Anthropic pydantic models with thinking/tool_use/tool_result
    blocks and no media parts. Exactly what server.py passes to routing.
    """
    return [
        AnthropicMessage(role="user", content="Fix the failing test in taskq."),
        AnthropicMessage(
            role="assistant",
            content=[
                ContentBlockThinking(thinking="Need to read the test file."),
                ContentBlockText(text="Let me look at the test."),
                ContentBlockToolUse(
                    id="toolu_1", name="Read", input={"file_path": "test_q.py"}
                ),
            ],
        ),
        AnthropicMessage(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="toolu_1",
                    content=[{"type": "text", "text": "def test_pop(): ..."}],
                )
            ],
        ),
        AnthropicMessage(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    id="toolu_2", name="Bash", input={"command": "pytest -q"}
                )
            ],
        ),
        AnthropicMessage(
            role="user",
            content=[ContentBlockToolResult(tool_use_id="toolu_2", content="1 failed")],
        ),
    ]


class TestDetection:
    def test_dict_image_part(self):
        assert _detect_modality([IMAGE_MSG]) == "vision"

    def test_pydantic_style_part(self):
        part = SimpleNamespace(type="image_url", image_url=SimpleNamespace(url="u"))
        msg = SimpleNamespace(role="user", content=[part])
        assert _detect_modality([msg]) == "vision"

    def test_plain_string_content(self):
        assert _detect_modality([{"role": "user", "content": "hi"}]) is None

    def test_text_only_parts(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert _detect_modality([msg]) is None

    def test_untyped_parts_treated_as_text(self):
        msg = {"role": "user", "content": [{"text": "legacy shape"}]}
        assert _detect_modality([msg]) is None

    def test_earlier_turn_image_detected(self):
        msgs = [
            IMAGE_MSG,
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": "what breed?"},
        ]
        assert _detect_modality(msgs) == "vision"

    def test_anthropic_image_block(self):
        msg = {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
        }
        assert _detect_modality([msg]) == "vision"

    def test_document_block_is_vision(self):
        msg = {
            "role": "user",
            "content": [{"type": "document", "source": {"type": "base64"}}],
        }
        assert _detect_modality([msg]) == "vision"

    def test_input_audio_part(self):
        assert _detect_modality([AUDIO_MSG]) == "audio"

    def test_vision_wins_over_audio(self):
        assert _detect_modality([AUDIO_MSG, IMAGE_MSG]) == "vision"

    # -- the 2026-07-11 regression: agent control-flow blocks are text-flow --

    def test_tool_flow_blocks_are_text_flow(self):
        assert _detect_modality(agent_messages()) is None

    def test_dict_tool_flow_blocks_are_text_flow(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "redacted_thinking", "data": "xx"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
        ]
        assert _detect_modality(msgs) is None

    def test_unknown_part_type_fails_open(self):
        msg = {"role": "user", "content": [{"type": "server_tool_use", "id": "x"}]}
        assert _detect_modality([msg]) is None

    def test_image_inside_tool_result_is_vision(self):
        # A browser tool returning a screenshot: the enclosing block is
        # text-flow but its nested content carries real image parts.
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "data": "x"}},
                    ],
                }
            ],
        }
        assert _detect_modality([msg]) == "vision"

    def test_text_only_tool_result_content_list(self):
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "42"}],
                }
            ],
        }
        assert _detect_modality([msg]) is None


@pytest.mark.asyncio
async def test_image_routes_to_vision_target(tmp_path):
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=[IMAGE_MSG], has_tools=False, request_id="r", stream=False
    )
    assert d.target == "vlm-model"
    assert d.rule_fired == "shape:vision"
    assert d.classify_ms < 50  # no classifier call
    await svc.close()


@pytest.mark.asyncio
async def test_audio_routes_to_audio_target(tmp_path):
    svc = make_service(tmp_path, vision="vlm-model", audio="audio-model")
    d = await svc.route_chat_request(
        messages=[AUDIO_MSG], has_tools=False, request_id="r", stream=False
    )
    assert d.target == "audio-model"
    assert d.rule_fired == "shape:audio"
    await svc.close()


@pytest.mark.asyncio
async def test_no_audio_target_falls_through(tmp_path):
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=[AUDIO_MSG], has_tools=False, request_id="r", stream=False
    )
    assert d.target == "small-model"
    assert d.rule_fired != "shape:audio"
    await svc.close()


@pytest.mark.asyncio
async def test_shape_beats_tools_override(tmp_path):
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=[IMAGE_MSG], has_tools=True, request_id="r", stream=False
    )
    assert d.target == "vlm-model"
    assert d.rule_fired == "shape:vision"
    await svc.close()


@pytest.mark.asyncio
async def test_no_vision_target_falls_through(tmp_path):
    svc = make_service(tmp_path)
    d = await svc.route_chat_request(
        messages=[IMAGE_MSG], has_tools=False, request_id="r", stream=False
    )
    assert d.target == "small-model"  # classified on text parts as before
    assert d.rule_fired != "shape:vision"
    await svc.close()


@pytest.mark.asyncio
async def test_text_request_unaffected(tmp_path):
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="r",
        stream=False,
    )
    assert d.target == "small-model"
    await svc.close()


@pytest.mark.asyncio
async def test_agent_traffic_not_eaten_by_shape_gate(tmp_path):
    """2026-07-11 regression: a tool-using agent conversation must NOT
    fire shape:vision. With tools present the agentic override applies;
    the request reaches the normal decision chain, not the vision target.
    """
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=agent_messages(), has_tools=True, request_id="r", stream=False
    )
    assert d.rule_fired == "override:tools"
    assert d.target == "big-model"
    await svc.close()


@pytest.mark.asyncio
async def test_agent_traffic_without_tools_reaches_classifier(tmp_path):
    """Same shape minus the tools flag: the classifier itself must run
    (the old gate returned classify_ms=0 / features=None on these)."""
    svc = make_service(tmp_path, vision="vlm-model")
    d = await svc.route_chat_request(
        messages=agent_messages(), has_tools=False, request_id="r", stream=False
    )
    assert d.rule_fired not in ("shape:vision", "override:tools")
    assert d.features is not None
    assert d.target == "small-model"
    await svc.close()
