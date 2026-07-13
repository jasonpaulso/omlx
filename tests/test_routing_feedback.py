# SPDX-License-Identifier: Apache-2.0
"""Tests for the M6.0 routing feedback ingest:

RoutingService.record_feedback / join_feedback (omlx/routing/service.py)
POST /v1/feedback (omlx/api/feedback_routes.py)
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api import feedback_routes
from omlx.routing.service import (
    RouteDecision,
    RoutingService,
    join_feedback,
    read_telemetry_tail,
)
from omlx.settings import RoutingSettings


def make_service(tmp_path, **overrides) -> RoutingService:
    data = {
        "enabled": True,
        "telemetry": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")},
    }
    data.update(overrides)
    return RoutingService(RoutingSettings.from_dict(data))


def decision(target="model-a", rule="small") -> RouteDecision:
    return RouteDecision(
        target=target,
        rule_fired=rule,
        override=None,
        features=None,
        raw_analysis=None,
        classify_ms=1.0,
    )


def record(service, request_id, target="model-a", rule="small"):
    service._record_decision(decision(target, rule), request_id, "chat", False)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRecordFeedback:
    async def test_round_trip_writes_feedback_row(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1")
        svc.record_feedback(
            "r1", score=0.9, label="good", tags=["accepted"], source="client"
        )
        await svc.close()

        rows = read_jsonl(svc.settings.telemetry.path)
        feedback_rows = [r for r in rows if r.get("kind") == "feedback"]
        assert len(feedback_rows) == 1
        fb = feedback_rows[0]
        assert fb["request_id"] == "r1"
        assert fb["score"] == 0.9
        assert fb["label"] == "good"
        assert fb["tags"] == ["accepted"]
        assert fb["source"] == "client"

    async def test_live_ring_attach(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1")
        svc.record_feedback("r1", score=0.5)

        rows = svc.recent_decisions(limit=10)
        assert len(rows) == 1
        assert rows[0]["request_id"] == "r1"
        assert len(rows[0]["feedback"]) == 1
        assert rows[0]["feedback"][0]["score"] == 0.5
        await svc.close()

    async def test_flushed_id_still_records_without_ring_entry(self, tmp_path):
        svc = make_service(tmp_path)
        # No matching decision in _recent at all -- request_id unknown here.
        svc.record_feedback("ghost-id", score=0.1, label="bad")
        await svc.close()

        rows = read_jsonl(svc.settings.telemetry.path)
        feedback_rows = [r for r in rows if r.get("kind") == "feedback"]
        assert len(feedback_rows) == 1
        assert feedback_rows[0]["request_id"] == "ghost-id"

    async def test_telemetry_disabled_is_noop(self, tmp_path):
        svc = make_service(
            tmp_path,
            telemetry={"enabled": False, "path": str(tmp_path / "decisions.jsonl")},
        )
        svc.record_feedback("r1", score=0.5)  # must not raise
        assert not (tmp_path / "decisions.jsonl").exists()
        assert svc.recent_decisions(limit=10) == []

    async def test_enqueue_failure_swallowed(self, tmp_path, monkeypatch):
        svc = make_service(tmp_path)
        record(svc, "r1")

        def _boom(row):
            raise RuntimeError("write failed")

        monkeypatch.setattr(svc, "_enqueue", _boom)
        svc.record_feedback("r1", score=0.5)  # must not raise
        monkeypatch.undo()
        await svc.close()


class TestJoinFeedback:
    def test_attaches_feedback_to_matching_decision(self):
        rows = [
            {"request_id": "r1", "target": "m"},
            {"request_id": "r2", "target": "m"},
            {"kind": "feedback", "request_id": "r1", "score": 0.7},
        ]
        joined = join_feedback(rows)
        by_id = {r["request_id"]: r for r in joined}
        assert len(by_id["r1"]["feedback"]) == 1
        assert by_id["r1"]["feedback"][0]["score"] == 0.7
        assert "feedback" not in by_id["r2"]

    def test_feedback_for_unknown_id_dropped(self):
        rows = [
            {"request_id": "r1", "target": "m"},
            {"kind": "feedback", "request_id": "unknown", "score": 0.7},
        ]
        joined = join_feedback(rows)
        assert len(joined) == 1
        assert "feedback" not in joined[0]


class TestRecentDecisionsExcludesFeedback:
    def test_feedback_rows_in_file_tail_not_surfaced_as_decisions(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"request_id": "old1", "target": "m"}) + "\n")
            f.write(
                json.dumps({"kind": "feedback", "request_id": "old1", "score": 0.5})
                + "\n"
            )
        svc = make_service(tmp_path)
        rows = svc.recent_decisions(limit=10)
        assert [r["request_id"] for r in rows] == ["old1"]
        assert all(r.get("kind") != "feedback" for r in rows)

    def test_read_telemetry_tail_still_includes_feedback_kind(self, tmp_path):
        # Sanity: read_telemetry_tail itself is kind-agnostic; the filtering
        # lives in recent_decisions.
        path = tmp_path / "decisions.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "feedback", "request_id": "r1"}) + "\n")
        rows = read_telemetry_tail(path, 10)
        assert rows[0]["kind"] == "feedback"


class _FakeRoutingService:
    def __init__(self):
        self.calls = []

    def record_feedback(self, request_id, **kwargs):
        self.calls.append((request_id, kwargs))


@pytest.fixture
def feedback_client(monkeypatch):
    fake = _FakeRoutingService()
    app = FastAPI()
    app.include_router(feedback_routes.router)
    feedback_routes.set_routing_service_getter(lambda: fake)
    yield TestClient(app), fake
    feedback_routes.set_routing_service_getter(None)


class TestFeedbackEndpoint:
    def test_valid_score_only_returns_202(self, feedback_client):
        client, fake = feedback_client
        r = client.post("/v1/feedback", json={"request_id": "r1", "score": 0.8})
        assert r.status_code == 202
        assert r.json() == {"recorded": True, "reason": None}
        assert len(fake.calls) == 1
        request_id, kwargs = fake.calls[0]
        assert request_id == "r1"
        assert kwargs == {
            "score": 0.8,
            "label": None,
            "tags": None,
            "comment": None,
            "source": "client",
        }

    def test_empty_signal_returns_422(self, feedback_client):
        client, _fake = feedback_client
        r = client.post("/v1/feedback", json={"request_id": "r1"})
        assert r.status_code == 422

    def test_score_out_of_range_returns_422(self, feedback_client):
        client, _fake = feedback_client
        r = client.post("/v1/feedback", json={"request_id": "r1", "score": 1.5})
        assert r.status_code == 422

    def test_routing_disabled_returns_202_recorded_false(self, feedback_client):
        client, fake = feedback_client
        feedback_routes.set_routing_service_getter(lambda: None)
        r = client.post("/v1/feedback", json={"request_id": "r1", "label": "good"})
        assert r.status_code == 202
        assert r.json() == {"recorded": False, "reason": "routing_disabled"}
        assert fake.calls == []


# #8 (routed response carries x-omlx-request-id header): skipped. Exercising
# that requires the full chat-completion request path (server.py app +
# model loading or a deep fake engine harness), which none of the existing
# routing tests set up -- they test RoutingService directly. Building that
# scaffolding is out of scope for this feedback-ingest test file.
