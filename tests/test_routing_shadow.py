# SPDX-License-Identifier: Apache-2.0
"""Tests for the Apple FM shadow labeler (omlx/routing/shadow.py)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omlx.routing.service import RoutingService
from omlx.routing.shadow import ShadowLabeler, elide
from omlx.settings import RoutingSettings, RoutingShadowLabelerSettings


class TestElide:
    def test_short_text_untouched(self):
        assert elide("hello") == "hello"

    def test_long_text_keeps_head_and_tail(self):
        text = "H" * 500 + "M" * 5000 + "T" * 300
        out = elide(text)
        assert out.startswith("H" * 500)
        assert out.endswith("T" * 300)
        assert "5000 characters" in out
        assert len(out) < 1000


class TestSettings:
    def test_defaults_off(self):
        s = RoutingSettings.from_dict({})
        assert s.shadow_labeler.enabled is False

    def test_round_trip(self):
        s = RoutingShadowLabelerSettings.from_dict(
            {"enabled": True, "use_case": "general", "timeout_s": 5.0}
        )
        assert s.to_dict() == {
            "enabled": True,
            "use_case": "general",
            "timeout_s": 5.0,
        }


def fake_proc(stdout: bytes, returncode: int = 0):
    async def communicate(_input=None):
        return stdout, b""

    return SimpleNamespace(communicate=communicate, returncode=returncode)


def cli_backend(which="/usr/bin/fm"):
    """Force the CLI path regardless of whether apple-fm-sdk is installed."""
    return (
        patch("omlx.routing.shadow._fm", None),
        patch("omlx.routing.shadow.shutil.which", return_value=which),
    )


class TestClassifyCli:
    @pytest.mark.asyncio
    async def test_no_backend_returns_none(self):
        labeler = ShadowLabeler()
        with (
            patch("omlx.routing.shadow._fm", None),
            patch("omlx.routing.shadow.shutil.which", return_value=None),
        ):
            assert await labeler.classify("some text") is None

    @pytest.mark.asyncio
    async def test_valid_label(self):
        labeler = ShadowLabeler()
        out = json.dumps({"label": "MODERATE", "reason": "multi-step"}).encode()
        fm_none, which = cli_backend()
        with (
            fm_none,
            which,
            patch(
                "omlx.routing.shadow.asyncio.create_subprocess_exec",
                return_value=fake_proc(out),
            ),
        ):
            rec = await labeler.classify("write a parser")
        assert rec["provider"] == "apple_fm"
        assert rec["backend"] == "cli"
        assert rec["label"] == "MODERATE"
        assert rec["reason"] == "multi-step"
        assert rec["ms"] >= 0

    @pytest.mark.asyncio
    async def test_invalid_label_returns_none(self):
        labeler = ShadowLabeler()
        out = json.dumps({"label": "BANANAS"}).encode()
        fm_none, which = cli_backend()
        with (
            fm_none,
            which,
            patch(
                "omlx.routing.shadow.asyncio.create_subprocess_exec",
                return_value=fake_proc(out),
            ),
        ):
            assert await labeler.classify("x") is None

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_none(self):
        labeler = ShadowLabeler()
        fm_none, which = cli_backend()
        with (
            fm_none,
            which,
            patch(
                "omlx.routing.shadow.asyncio.create_subprocess_exec",
                return_value=fake_proc(b"", returncode=2),
            ),
        ):
            assert await labeler.classify("x") is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        labeler = ShadowLabeler(timeout_s=0.01)

        async def hang(_input=None):
            await asyncio.sleep(10)

        proc = SimpleNamespace(communicate=hang, returncode=None)
        fm_none, which = cli_backend()
        with (
            fm_none,
            which,
            patch(
                "omlx.routing.shadow.asyncio.create_subprocess_exec",
                return_value=proc,
            ),
        ):
            assert await labeler.classify("x") is None

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self):
        labeler = ShadowLabeler()
        assert await labeler.classify("") is None


class FakeSession:
    def __init__(self, instructions=None):
        self.instructions = instructions

    async def respond(self, prompt, json_schema=None, options=None):
        # Real SDK (0.2.1): GeneratedContent.value is a METHOD, not a
        # property — the fake mirrors that (it silently broke extraction
        # once already).
        return SimpleNamespace(
            value=lambda: {"label": "SIMPLE", "reason": "short synthesis"}
        )


def fake_sdk():
    return SimpleNamespace(
        LanguageModelSession=FakeSession,
        GenerationOptions=lambda **kw: SimpleNamespace(**kw),
        SamplingMode=SimpleNamespace(greedy=lambda: "greedy"),
    )


class TestClassifySdk:
    @pytest.mark.asyncio
    async def test_sdk_preferred_over_cli(self):
        labeler = ShadowLabeler()
        with patch("omlx.routing.shadow._fm", fake_sdk()):
            rec = await labeler.classify("summarize this")
        assert rec["backend"] == "sdk"
        assert rec["label"] == "SIMPLE"

    @pytest.mark.asyncio
    async def test_sdk_error_returns_none(self):
        class BoomSession(FakeSession):
            async def respond(self, *a, **kw):
                raise RuntimeError("rate limited")

        sdk = fake_sdk()
        sdk.LanguageModelSession = BoomSession
        labeler = ShadowLabeler()
        with patch("omlx.routing.shadow._fm", sdk):
            assert await labeler.classify("x") is None


class FakeEngine:
    async def generate(self, **kwargs):
        return SimpleNamespace(
            text=(
                "Domain: general | Complexity: 1 | Math: False | Code: False "
                "| Route: small model"
            )
        )


def make_service(tmp_path, *, shadow_enabled=True):
    svc = RoutingService(
        RoutingSettings.from_dict(
            {
                "enabled": True,
                "targets": {"small": "small-model", "big": "big-model"},
                "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
                "shadow_labeler": {"enabled": shadow_enabled},
            }
        )
    )

    async def getter(model_id):
        return FakeEngine()

    svc.set_engine_getter(getter)
    return svc


@pytest.mark.asyncio
async def test_shadow_label_lands_in_telemetry_row(tmp_path):
    svc = make_service(tmp_path)

    async def fake_classify(text):
        return {"provider": "apple_fm", "label": "TRIVIAL", "reason": "r", "ms": 1.0}

    svc._shadow.classify = fake_classify
    await svc.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    await asyncio.gather(*svc._shadow_tasks)
    svc.record_outcome("r1", completion_tokens=1, finish_reason="stop", gen_ms=1.0)
    await svc.close()
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["shadow"]["label"] == "TRIVIAL"


@pytest.mark.asyncio
async def test_shadow_disabled_spawns_no_tasks(tmp_path):
    svc = make_service(tmp_path, shadow_enabled=False)
    await svc.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    assert svc._shadow is None
    assert not svc._shadow_tasks
    await svc.close()


@pytest.mark.asyncio
async def test_shadow_failure_leaves_row_clean(tmp_path):
    svc = make_service(tmp_path)

    async def fake_classify(text):
        return None

    svc._shadow.classify = fake_classify
    await svc.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    await asyncio.gather(*svc._shadow_tasks)
    svc.record_outcome("r1", completion_tokens=1, finish_reason="stop", gen_ms=1.0)
    await svc.close()
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["shadow"] is None
