# SPDX-License-Identifier: Apache-2.0
"""Tests for the shape-based vision pre-route (M4.1)."""

from types import SimpleNamespace

import pytest

from omlx.routing.service import RoutingService, _has_non_text_parts
from omlx.settings import RoutingSettings


class FakeEngine:
    async def generate(self, **kwargs):
        return SimpleNamespace(
            text=(
                "Domain: general | Complexity: 1 | Math: False | Code: False "
                "| Route: small model"
            )
        )


def make_service(tmp_path, *, vision=None):
    targets = {"small": "small-model", "big": "big-model"}
    if vision:
        targets["vision"] = vision
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


class TestDetection:
    def test_dict_image_part(self):
        assert _has_non_text_parts([IMAGE_MSG])

    def test_pydantic_style_part(self):
        part = SimpleNamespace(type="image_url", image_url=SimpleNamespace(url="u"))
        msg = SimpleNamespace(role="user", content=[part])
        assert _has_non_text_parts([msg])

    def test_plain_string_content(self):
        assert not _has_non_text_parts([{"role": "user", "content": "hi"}])

    def test_text_only_parts(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert not _has_non_text_parts([msg])

    def test_untyped_parts_treated_as_text(self):
        msg = {"role": "user", "content": [{"text": "legacy shape"}]}
        assert not _has_non_text_parts([msg])

    def test_earlier_turn_image_detected(self):
        msgs = [
            IMAGE_MSG,
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": "what breed?"},
        ]
        assert _has_non_text_parts(msgs)


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
