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


def make_settings(tmp_path, *, table_enabled=True, default_target=None):
    return RoutingSettings.from_dict(
        {
            "enabled": True,
            "targets": {"small": "small-model", "big": "big-model"},
            "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
            "table_dispatch": {
                "enabled": table_enabled,
                "default_target": default_target,
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
    svc = make_service(
        make_settings(tmp_path, default_target="general-model"), MODELS
    )
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
