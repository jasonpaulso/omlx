# SPDX-License-Identifier: Apache-2.0
"""Service-level tests for M3 table dispatch integration."""

import json

import pytest

from omlx.routing.service import RoutingService
from omlx.settings import RoutingSettings


class FakeEngine:
    def __init__(self, text):
        self._text = text

    async def generate(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(text=self._text)


ANALYSIS = (
    "Domain: coding | Complexity: 2 | Math: False | Code: True | Route: small model"
)


def make_settings(
    tmp_path,
    *,
    table_enabled=True,
    default_target=None,
    max_interactive_ttft_s=None,
):
    return RoutingSettings.from_dict(
        {
            "enabled": True,
            "targets": {"small": "small-model", "big": "big-model"},
            "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
            "table_dispatch": {
                "enabled": table_enabled,
                "default_target": default_target,
                "max_interactive_ttft_s": max_interactive_ttft_s,
            },
        }
    )


def make_service(settings, models, resident=frozenset(), analysis=ANALYSIS):
    svc = RoutingService(settings)

    async def getter(model_id):
        return FakeEngine(analysis)

    svc.set_engine_getter(getter)
    svc.set_table_sources(lambda: models, lambda: set(resident))
    return svc


MODELS = {
    "coder-model": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"code": 0.9, "knowledge": 0.6},
        "evals": [],
    },
    "general-model": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"code": 0.5, "knowledge": 0.85},
        "evals": [],
    },
}


async def route(svc, **over):
    kwargs = dict(
        messages=[{"role": "user", "content": "write a parser"}],
        has_tools=False,
        request_id="r1",
        stream=False,
    )
    kwargs.update(over)
    return await svc.route_chat_request(**kwargs)


@pytest.mark.asyncio
async def test_table_dispatch_picks_axis_leader(tmp_path):
    svc = make_service(make_settings(tmp_path), MODELS)
    d = await route(svc)
    assert d.target == "coder-model"
    assert d.rule_fired == "table:code"
    assert d.candidates and d.candidates[0][0] == "coder-model"
    await svc.close()
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["candidates_considered"][0][0] == "coder-model"


@pytest.mark.asyncio
async def test_table_disabled_uses_binary(tmp_path):
    svc = make_service(make_settings(tmp_path, table_enabled=False), MODELS)
    d = await route(svc)
    assert d.target == "small-model"  # binary policy on the same features
    assert d.candidates is None
    await svc.close()


@pytest.mark.asyncio
async def test_empty_table_falls_back_to_binary(tmp_path):
    svc = make_service(make_settings(tmp_path), {})
    d = await route(svc)
    assert d.target == "small-model"
    assert d.rule_fired != "table:code"
    await svc.close()


@pytest.mark.asyncio
async def test_override_routes_to_generalist_spine(tmp_path):
    svc = make_service(make_settings(tmp_path, default_target="general-model"), MODELS)
    d = await route(svc, has_tools=True)
    assert d.target == "general-model"
    assert d.rule_fired == "override:tools"
    await svc.close()


@pytest.mark.asyncio
async def test_override_without_default_target_uses_big(tmp_path):
    svc = make_service(make_settings(tmp_path), MODELS)
    d = await route(svc, has_tools=True)
    assert d.target == "big-model"
    await svc.close()


@pytest.mark.asyncio
async def test_broken_sources_fall_back_to_binary(tmp_path):
    svc = make_service(make_settings(tmp_path), MODELS)

    def boom():
        raise RuntimeError("store exploded")

    svc.set_table_sources(boom, boom)
    d = await route(svc)
    assert d.target == "small-model"  # never 5xx, never raise
    await svc.close()


@pytest.mark.asyncio
async def test_enable_routing_gate_skips_disabled_model(tmp_path):
    svc = make_service(make_settings(tmp_path), MODELS)
    # Only general-model opted in; the code leader (coder-model) is gated out.
    svc.set_table_sources(
        lambda: MODELS, lambda: set(), enabled_getter=lambda: {"general-model"}
    )
    d = await route(svc)
    assert d.target == "general-model"
    assert d.rule_fired == "table:code"
    assert d.disabled == ["coder-model"]
    await svc.close()
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["disabled"] == ["coder-model"]


@pytest.mark.asyncio
async def test_enable_routing_empty_set_is_noop(tmp_path):
    svc = make_service(make_settings(tmp_path), MODELS)
    # Nobody opted in -> gate inert -> axis leader still wins, nothing gated.
    svc.set_table_sources(lambda: MODELS, lambda: set(), enabled_getter=lambda: set())
    d = await route(svc)
    assert d.target == "coder-model"
    assert d.disabled is None
    await svc.close()


AGENTIC_MODELS = {
    "tool-leader": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"agentic": 0.91, "code": 0.7},
        "evals": [],
    },
    "tool-runner": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"agentic": 0.88, "code": 0.9},
        "evals": [],
    },
}


@pytest.mark.asyncio
async def test_override_dispatches_on_agentic_axis(tmp_path):
    svc = make_service(
        make_settings(tmp_path, default_target="general-model"),
        AGENTIC_MODELS,
    )
    d = await route(svc, has_tools=True)
    assert d.target == "tool-leader"
    assert d.rule_fired == "table:agentic"
    assert d.override == "tools"
    assert d.candidates[0][0] == "tool-leader"
    await svc.close()


@pytest.mark.asyncio
async def test_override_agentic_prefers_resident_within_epsilon(tmp_path):
    models = {
        "cold-leader": dict(AGENTIC_MODELS["tool-leader"]),
        "warm-second": {
            "role": "chat",
            "health": {"status": "ok"},
            "categories": {"agentic": 0.90},
            "evals": [],
        },
    }
    svc = make_service(make_settings(tmp_path), models, resident={"warm-second"})
    d = await route(svc, has_tools=True)
    assert d.target == "warm-second"
    assert d.rule_fired == "table:agentic"
    await svc.close()


@pytest.mark.asyncio
async def test_turns_override_also_dispatches_agentic(tmp_path):
    svc = make_service(make_settings(tmp_path), AGENTIC_MODELS)
    msgs = [{"role": "user", "content": f"turn {i}"} for i in range(4)]
    d = await route(svc, messages=msgs)
    assert d.override == "turns"
    assert d.target == "tool-leader"
    assert d.rule_fired == "table:agentic"
    await svc.close()


@pytest.mark.asyncio
async def test_override_agentic_respects_enable_routing_gate(tmp_path):
    svc = make_service(make_settings(tmp_path), AGENTIC_MODELS)
    svc.set_table_sources(
        lambda: AGENTIC_MODELS,
        lambda: set(),
        enabled_getter=lambda: {"tool-runner"},
    )
    d = await route(svc, has_tools=True)
    assert d.target == "tool-runner"
    assert d.disabled == ["tool-leader"]
    await svc.close()


# --- M8: est_ttft gate, threaded end-to-end from the request messages -------

TTFT_MODELS = {
    "slow-31b": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"agentic": 0.92, "code": 0.92, "knowledge": 0.92},
        "evals": [],
        "prefill": {"24576": 230.0, "measured_at": "x"},
    },
    "fast-35b": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"agentic": 0.88, "code": 0.88, "knowledge": 0.88},
        "evals": [],
        "prefill": {"24576": 1240.0, "measured_at": "x"},
    },
}


@pytest.mark.asyncio
async def test_ttft_gate_skips_slow_prefill_leader_on_long_prompt(tmp_path):
    # ~24k-token prompt (chars/4) with both models resident: the slow-prefill
    # agentic leader is gated out, the fast one is dispatched, and the skip is
    # recorded on the decision and in telemetry.
    svc = make_service(
        make_settings(tmp_path, max_interactive_ttft_s=30.0),
        TTFT_MODELS,
        resident={"slow-31b", "fast-35b"},
    )
    big = "x" * (24576 * 4)
    d = await route(svc, has_tools=True, messages=[{"role": "user", "content": big}])
    assert d.target == "fast-35b"
    assert d.rule_fired == "table:agentic"
    assert d.slow_ttft == ["slow-31b"]
    await svc.close()

    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["slow_ttft"] == ["slow-31b"]


@pytest.mark.asyncio
async def test_ttft_gate_inert_on_short_prompt(tmp_path):
    # Same gate on, tiny prompt: est_ttft well under budget -> leader wins,
    # nothing gated.
    svc = make_service(
        make_settings(tmp_path, max_interactive_ttft_s=30.0),
        TTFT_MODELS,
        resident={"slow-31b", "fast-35b"},
    )
    d = await route(svc, has_tools=True, messages=[{"role": "user", "content": "hi"}])
    assert d.target == "slow-31b"
    assert d.slow_ttft is None
    await svc.close()


def test_table_dispatch_settings_ttft_round_trip():
    from omlx.settings import RoutingTableDispatchSettings

    s = RoutingTableDispatchSettings(max_interactive_ttft_s=20.0)
    assert s.to_dict()["max_interactive_ttft_s"] == 20.0
    assert (
        RoutingTableDispatchSettings.from_dict(s.to_dict()).max_interactive_ttft_s
        == 20.0
    )
    # Absent key defaults to None (gate inert / backward compatible).
    assert RoutingTableDispatchSettings.from_dict({}).max_interactive_ttft_s is None
