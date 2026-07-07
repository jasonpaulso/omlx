# SPDX-License-Identifier: Apache-2.0
"""Tests for P0-C fit-aware dispatch (omlx/routing/table.py, service.py)."""

import json

from omlx.routing.profiler import RouterFeatures
from omlx.routing.service import RoutingService
from omlx.routing.table import choose
from omlx.settings import RoutingSettings


def feats(complexity=2, math=False, code=False):
    return RouterFeatures(
        domain=None, complexity=complexity, math=math, code=code, route_token=None
    )


def entry(
    role="chat",
    healthy=True,
    categories=None,
    size_gb=None,
):
    return {
        "role": role,
        "health": {"status": "ok" if healthy else "unhealthy"},
        "categories": categories or {},
        "evals": [],
        "size_gb": size_gb,
    }


class TestChooseFitFilter:
    def test_fit_excludes_oversize_candidate_and_records_unfit(self):
        models = {
            "huge": entry(categories={"knowledge": 0.95}, size_gb=80.0),
            "small": entry(categories={"knowledge": 0.80}, size_gb=8.0),
        }
        c = choose(feats(), models, set(), escalate_at=4, fit_budget_gb=20.0)
        assert c.target == "small"
        assert c.unfit == ["huge"]
        assert all(mid != "huge" for mid, _ in c.candidates)

    def test_missing_size_gb_not_filtered(self):
        models = {
            "unknown_size": entry(categories={"knowledge": 0.95}, size_gb=None),
            "small": entry(categories={"knowledge": 0.80}, size_gb=8.0),
        }
        c = choose(feats(), models, set(), escalate_at=4, fit_budget_gb=20.0)
        assert c.target == "unknown_size"
        assert c.unfit == []

    def test_missing_fit_budget_no_filtering(self):
        models = {
            "huge": entry(categories={"knowledge": 0.95}, size_gb=80.0),
            "small": entry(categories={"knowledge": 0.80}, size_gb=8.0),
        }
        c = choose(feats(), models, set(), escalate_at=4, fit_budget_gb=None)
        assert c.target == "huge"
        assert c.unfit == []

    def test_old_behavior_byte_identical_when_budget_absent(self):
        models = {
            "huge": entry(categories={"knowledge": 0.95}, size_gb=80.0),
            "small": entry(categories={"knowledge": 0.80}, size_gb=8.0),
        }
        no_kwarg = choose(feats(), models, set(), escalate_at=4)
        explicit_none = choose(
            feats(), models, set(), escalate_at=4, fit_budget_gb=None
        )
        assert no_kwarg == explicit_none

    def test_fit_filter_applies_in_escalation_tier(self):
        models = {
            "frontier": entry(categories={"code": 0.9, "knowledge": 0.9}, size_gb=80.0),
            "coder": entry(categories={"code": 0.7, "knowledge": 0.5}, size_gb=8.0),
        }
        c = choose(
            feats(complexity=5, code=True),
            models,
            set(),
            escalate_at=4,
            fit_budget_gb=20.0,
        )
        assert c.target == "coder"
        assert c.unfit == ["frontier"]

    def test_exact_boundary_fits(self):
        models = {"m": entry(categories={"knowledge": 0.8}, size_gb=20.0)}
        c = choose(feats(), models, set(), escalate_at=4, fit_budget_gb=20.0)
        assert c.target == "m"
        assert c.unfit == []


MODELS_WITH_SIZE = {
    "coder-model": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"code": 0.9, "knowledge": 0.6},
        "evals": [],
        "size_gb": 80.0,
    },
    "general-model": {
        "role": "chat",
        "health": {"status": "ok"},
        "categories": {"code": 0.5, "knowledge": 0.85},
        "evals": [],
        "size_gb": 8.0,
    },
}

ANALYSIS = (
    "Domain: coding | Complexity: 2 | Math: False | Code: True | Route: small model"
)


class FakeEngine:
    def __init__(self, text):
        self._text = text

    async def generate(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(text=self._text)


def make_settings(tmp_path):
    return RoutingSettings.from_dict(
        {
            "enabled": True,
            "targets": {"small": "small-model", "big": "big-model"},
            "telemetry": {"enabled": True, "path": str(tmp_path / "t.jsonl")},
            "table_dispatch": {"enabled": True},
        }
    )


def make_service(settings, models, fit_budget_getter=None):
    svc = RoutingService(settings)

    async def getter(model_id):
        return FakeEngine(ANALYSIS)

    svc.set_engine_getter(getter)
    svc.set_table_sources(lambda: models, set, fit_budget_getter)
    return svc


class TestServiceFitWiring:
    async def test_fit_budget_getter_excludes_oversize_and_telemetry_carries_unfit(
        self, tmp_path
    ):
        svc = make_service(
            make_settings(tmp_path), MODELS_WITH_SIZE, fit_budget_getter=lambda: 20.0
        )
        decision = await svc.route_chat_request(
            messages=[{"role": "user", "content": "write a parser"}],
            has_tools=False,
            request_id="r1",
            stream=False,
        )
        assert decision.target == "general-model"
        assert decision.unfit == ["coder-model"]
        await svc.close()
        rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
        assert rows[0]["unfit"] == ["coder-model"]

    async def test_fit_budget_getter_absent_old_behavior(self, tmp_path):
        svc = make_service(make_settings(tmp_path), MODELS_WITH_SIZE)
        decision = await svc.route_chat_request(
            messages=[{"role": "user", "content": "write a parser"}],
            has_tools=False,
            request_id="r1",
            stream=False,
        )
        assert decision.target == "coder-model"
        assert decision.unfit is None
        await svc.close()
        rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
        assert rows[0]["unfit"] is None

    async def test_set_table_sources_two_arg_call_still_works(self, tmp_path):
        """Backward compat: pre-existing 2-positional-arg call sites must not break."""
        svc = RoutingService(make_settings(tmp_path))

        async def getter(model_id):
            return FakeEngine(ANALYSIS)

        svc.set_engine_getter(getter)
        svc.set_table_sources(lambda: MODELS_WITH_SIZE, set)  # no third arg
        decision = await svc.route_chat_request(
            messages=[{"role": "user", "content": "write a parser"}],
            has_tools=False,
            request_id="r1",
            stream=False,
        )
        assert decision.target == "coder-model"
        await svc.close()
