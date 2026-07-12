# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.routing.service.RoutingService: glue + telemetry.

Uses a fake engine/getter throughout -- no model files or real inference.
"""

import asyncio
import json

from omlx.routing.service import RoutingService
from omlx.settings import RoutingSettings


def make_settings(tmp_path, **overrides) -> RoutingSettings:
    settings = RoutingSettings()
    settings.classify_timeout_s = overrides.get("classify_timeout_s", 3.0)
    settings.telemetry.enabled = overrides.get("telemetry_enabled", True)
    settings.telemetry.path = str(tmp_path / "routing_decisions.jsonl")
    if "fail_open_target" in overrides:
        settings.policy.fail_open_target = overrides["fail_open_target"]
    return settings


class _FakeGenerationOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeEngine:
    def __init__(self, response_text: str = "", delay: float = 0.0) -> None:
        self.response_text = response_text
        self.delay = delay

    async def generate(self, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        return _FakeGenerationOutput(self.response_text)


BIG_ANALYSIS = (
    "Domain: Programming | Complexity: 4 | Math: False | Code: True | "
    "Route: big model | Justification: complex."
)
SMALL_ANALYSIS = (
    "Domain: Geography | Complexity: 1 | Math: False | Code: False | "
    "Route: small model | Justification: trivial."
)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def test_route_chat_request_escalates_on_complexity(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(BIG_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "implement a distributed lock"}],
        has_tools=False,
        request_id="req-1",
        stream=False,
    )

    assert decision.target == settings.targets["big"]
    assert decision.rule_fired == "complexity>=4"
    assert decision.override is None
    assert decision.features.complexity == 4
    await service.close()


async def test_route_chat_request_stays_small_on_low_complexity(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(SMALL_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "capital of Portugal?"}],
        has_tools=False,
        request_id="req-2",
        stream=False,
    )

    assert decision.target == settings.targets["small"]
    await service.close()


async def test_tools_override_skips_classification(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)

    async def _getter(model_id):
        raise AssertionError("engine getter should not be called when tools override")

    service.set_engine_getter(_getter)

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "call a tool"}],
        has_tools=True,
        request_id="req-3",
        stream=False,
    )

    assert decision.target == settings.targets["big"]
    assert decision.rule_fired == "override:tools"
    assert decision.override == "tools"
    await service.close()


async def test_turns_override(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)

    async def _getter(model_id):
        raise AssertionError("engine getter should not be called on turns override")

    service.set_engine_getter(_getter)

    messages = [{"role": "user", "content": f"turn {i}"} for i in range(5)]
    decision = await service.route_chat_request(
        messages=messages,
        has_tools=False,
        request_id="req-4",
        stream=False,
    )

    assert decision.rule_fired == "override:turns"
    await service.close()


async def test_no_engine_getter_fails_open(tmp_path):
    settings = make_settings(tmp_path, fail_open_target="big")
    service = RoutingService(settings)
    # set_engine_getter never called

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="req-5",
        stream=False,
    )

    assert decision.target == settings.targets["big"]
    assert decision.rule_fired == "fail_open:error"
    await service.close()


async def test_engine_error_fails_open(tmp_path):
    settings = make_settings(tmp_path, fail_open_target="small")

    async def _getter(model_id):
        raise RuntimeError("engine blew up")

    service = RoutingService(settings)
    service.set_engine_getter(_getter)

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="req-6",
        stream=False,
    )

    assert decision.target == settings.targets["small"]
    assert decision.rule_fired == "fail_open:error"
    await service.close()


async def test_classify_timeout_fails_open(tmp_path):
    settings = make_settings(tmp_path, classify_timeout_s=0.05, fail_open_target="big")
    service = RoutingService(settings)
    slow_engine = _FakeEngine(BIG_ANALYSIS, delay=1.0)
    service.set_engine_getter(lambda model_id: _get(slow_engine))

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="req-7",
        stream=False,
    )

    assert decision.target == settings.targets["big"]
    assert decision.rule_fired == "fail_open:timeout"
    await service.close()


async def test_telemetry_row_written_with_outcome(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(BIG_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    await service.route_chat_request(
        messages=[{"role": "user", "content": "implement a distributed lock"}],
        has_tools=False,
        request_id="req-8",
        stream=True,
        endpoint="chat",
    )
    service.record_outcome(
        "req-8",
        completion_tokens=512,
        finish_reason="stop",
        gen_ms=8000.0,
        ttft_ms=1234.56,
        decode_ms=6765.44,
        prompt_tokens=4096,
        cached_tokens=3072,
    )
    await service.close()

    rows = read_jsonl(settings.telemetry.path)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "req-8"
    assert row["endpoint"] == "chat"
    assert row["stream"] is True
    assert row["target"] == settings.targets["big"]
    assert row["rule_fired"] == "complexity>=4"
    assert row["outcome"] == {
        "completion_tokens": 512,
        "finish_reason": "stop",
        "gen_ms": 8000.0,
        "ttft_ms": 1234.6,
        "decode_ms": 6765.4,
        "prompt_tokens": 4096,
        "cached_tokens": 3072,
    }


async def test_outcome_timing_fields_default_to_none(tmp_path):
    """Non-streaming paths pass no timing/token detail; row keys still present."""
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(BIG_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    await service.route_chat_request(
        messages=[{"role": "user", "content": "implement a distributed lock"}],
        has_tools=False,
        request_id="req-9",
        stream=False,
        endpoint="chat",
    )
    service.record_outcome(
        "req-9", completion_tokens=64, finish_reason="stop", gen_ms=500.0
    )
    await service.close()

    rows = read_jsonl(settings.telemetry.path)
    assert len(rows) == 1
    outcome = rows[0]["outcome"]
    assert outcome["ttft_ms"] is None
    assert outcome["decode_ms"] is None
    assert outcome["prompt_tokens"] is None
    assert outcome["cached_tokens"] is None


async def test_telemetry_flushes_pending_rows_without_outcome_on_close(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(SMALL_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    await service.route_chat_request(
        messages=[{"role": "user", "content": "capital of Portugal?"}],
        has_tools=False,
        request_id="req-9",
        stream=False,
    )
    # No record_outcome call before close().
    await service.close()

    rows = read_jsonl(settings.telemetry.path)
    assert len(rows) == 1
    assert rows[0]["outcome"] is None


async def test_telemetry_disabled_writes_nothing(tmp_path):
    settings = make_settings(tmp_path, telemetry_enabled=False)
    service = RoutingService(settings)
    engine = _FakeEngine(SMALL_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    await service.route_chat_request(
        messages=[{"role": "user", "content": "hi"}],
        has_tools=False,
        request_id="req-10",
        stream=False,
    )
    service.record_outcome(
        "req-10", completion_tokens=1, finish_reason="stop", gen_ms=1.0
    )
    await service.close()

    assert not (tmp_path / "routing_decisions.jsonl").exists()


async def test_record_outcome_for_unknown_request_id_is_noop(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    # No prior route_chat_request for this id.
    service.record_outcome(
        "never-routed", completion_tokens=1, finish_reason="stop", gen_ms=1.0
    )
    await service.close()

    assert not (tmp_path / "routing_decisions.jsonl").exists()


async def test_orphaned_pending_rows_flushed_with_null_outcome(tmp_path):
    """A 507/disconnected request's row is older than the threshold and
    flushed on the next decision, without waiting for close()."""
    import datetime as dt

    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(SMALL_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    old_ts = (
        dt.datetime.now(dt.timezone.utc)  # noqa: UP017 - mypy targets py310
        - dt.timedelta(seconds=601)
    ).isoformat()
    service._pending["orphan-1"] = {
        "ts": old_ts,
        "request_id": "orphan-1",
        "outcome": None,
    }

    await service.route_chat_request(
        messages=[{"role": "user", "content": "hi"}],
        has_tools=False,
        request_id="req-fresh",
        stream=False,
    )

    # The orphan was flushed to the queue (removed from pending); the fresh
    # row is untouched, still pending until close/outcome.
    assert "orphan-1" not in service._pending
    assert "req-fresh" in service._pending

    await service.close()
    rows = read_jsonl(settings.telemetry.path)
    ids = {r["request_id"] for r in rows}
    assert "orphan-1" in ids
    assert "req-fresh" in ids
    orphan_row = next(r for r in rows if r["request_id"] == "orphan-1")
    assert orphan_row["outcome"] is None


async def test_fresh_pending_rows_not_flushed(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(SMALL_ANALYSIS)
    service.set_engine_getter(lambda model_id: _get(engine))

    await service.route_chat_request(
        messages=[{"role": "user", "content": "first"}],
        has_tools=False,
        request_id="req-first",
        stream=False,
    )
    await service.route_chat_request(
        messages=[{"role": "user", "content": "second"}],
        has_tools=False,
        request_id="req-second",
        stream=False,
    )

    # Both are fresh (well under 600s old); neither should be flushed early.
    assert "req-first" in service._pending
    assert "req-second" in service._pending
    await service.close()


class _FakePart:
    """Stands in for a pydantic content-part model (attribute access)."""

    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        self.text = text


class _FakeMessage:
    """Stands in for a pydantic ChatMessage model (attribute access)."""

    def __init__(self, role: str, content) -> None:
        self.role = role
        self.content = content


async def test_route_chat_request_accepts_pydantic_style_messages(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    engine = _FakeEngine(BIG_ANALYSIS)
    captured_prompt = {}

    async def _getter(model_id):
        return engine

    async def _generate_spy(**kwargs):
        captured_prompt["prompt"] = kwargs["prompt"]
        return await _FakeEngine.generate(engine, **kwargs)

    engine.generate = _generate_spy
    service.set_engine_getter(_getter)

    messages = [
        _FakeMessage("system", "be helpful"),
        _FakeMessage(
            "user",
            [
                _FakePart("text", "implement a distributed lock"),
                _FakePart("image_url", None),
            ],
        ),
    ]
    decision = await service.route_chat_request(
        messages=messages,
        has_tools=False,
        request_id="req-11",
        stream=False,
    )

    assert decision.target == settings.targets["big"]
    assert "implement a distributed lock" in captured_prompt["prompt"]
    await service.close()


async def test_turns_override_counts_pydantic_style_messages(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)

    async def _getter(model_id):
        raise AssertionError("engine getter should not be called on turns override")

    service.set_engine_getter(_getter)

    messages = [_FakeMessage("user", f"turn {i}") for i in range(5)]
    decision = await service.route_chat_request(
        messages=messages,
        has_tools=False,
        request_id="req-12",
        stream=False,
    )

    assert decision.rule_fired == "override:turns"
    await service.close()


def test_settings_is_public_attribute(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    assert service.settings is settings


def test_header_value_format(tmp_path):
    from omlx.routing.service import RouteDecision

    decision = RouteDecision(
        target="gpt-oss-120b-Fable-5-Distilled",
        rule_fired="complexity>=4",
        override=None,
        features=None,
        raw_analysis=None,
        classify_ms=210.4,
    )
    assert isinstance(decision.header_value, str)
    assert (
        decision.header_value
        == "gpt-oss-120b-Fable-5-Distilled; rule=complexity>=4; classify_ms=210"
    )


async def _get(engine):
    return engine
