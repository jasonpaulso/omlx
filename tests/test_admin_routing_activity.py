# SPDX-License-Identifier: Apache-2.0
"""Tests for the Router tab admin endpoints:

GET  /admin/api/routing/activity
POST /admin/api/routing/settings
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.settings import GlobalSettings


class _FakeRoutingService:
    def __init__(self):
        self.last_limit = None
        self.rows = [{"request_id": "r1", "target": "model-a", "outcome": None}]

    def recent_decisions(self, limit=100):
        self.last_limit = limit
        return self.rows

    @property
    def pending_count(self):
        return 1

    def shadow_status(self):
        return {"enabled": True, "backend": "sdk"}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    gs = GlobalSettings(base_path=tmp_path)
    gs.routing.telemetry.path = str(tmp_path / "decisions.jsonl")
    state = SimpleNamespace(routing_service=None)

    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: state)

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app), gs, state


class TestActivity:
    def test_no_service_falls_back_to_file_tail(self, harness, tmp_path):
        client, gs, _state = harness
        with open(gs.routing.telemetry.path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"request_id": "from-file", "target": "m"}) + "\n")
        r = client.get("/admin/api/routing/activity")
        assert r.status_code == 200
        body = r.json()
        assert body["service_active"] is False
        assert body["pending_count"] == 0
        assert body["shadow"] == {"enabled": False, "backend": None}
        assert [d["request_id"] for d in body["decisions"]] == ["from-file"]
        assert body["config"] == gs.routing.to_dict()

    def test_active_service_serves_buffer_and_status(self, harness):
        client, _gs, state = harness
        svc = _FakeRoutingService()
        state.routing_service = svc
        r = client.get("/admin/api/routing/activity?limit=25")
        assert r.status_code == 200
        body = r.json()
        assert body["service_active"] is True
        assert body["pending_count"] == 1
        assert body["shadow"] == {"enabled": True, "backend": "sdk"}
        assert body["decisions"] == svc.rows
        assert svc.last_limit == 25

    def test_limit_clamped(self, harness):
        client, _gs, state = harness
        svc = _FakeRoutingService()
        state.routing_service = svc
        client.get("/admin/api/routing/activity?limit=99999")
        assert svc.last_limit == 256
        client.get("/admin/api/routing/activity?limit=0")
        assert svc.last_limit == 1


class TestUpdateSettings:
    def test_round_trip_saves_and_flags_restart(self, harness, tmp_path):
        client, gs, _state = harness
        payload = gs.routing.to_dict()
        payload["enabled"] = True
        payload["table_dispatch"]["residency_epsilon"] = 0.05
        payload["targets"]["vision"] = "vlm-model"
        r = client.post("/admin/api/routing/settings", json=payload)
        assert r.status_code == 200
        assert r.json() == {"success": True, "restart_required": True}
        assert gs.routing.enabled is True
        assert gs.routing.table_dispatch.residency_epsilon == 0.05
        assert gs.routing.targets["vision"] == "vlm-model"
        # Persisted to the settings file.
        saved = json.loads((tmp_path / "settings.json").read_text())
        assert saved["routing"]["enabled"] is True

    def test_empty_string_targets_dropped(self, harness):
        client, gs, _state = harness
        payload = gs.routing.to_dict()
        payload["targets"]["vision"] = ""
        payload["targets"]["audio"] = "   "
        r = client.post("/admin/api/routing/settings", json=payload)
        assert r.status_code == 200
        assert "vision" not in gs.routing.targets
        assert "audio" not in gs.routing.targets

    def test_empty_virtual_model_id_rejected(self, harness):
        client, gs, _state = harness
        before = gs.routing.to_dict()
        payload = gs.routing.to_dict()
        payload["virtual_model_id"] = "  "
        r = client.post("/admin/api/routing/settings", json=payload)
        assert r.status_code == 400
        assert gs.routing.to_dict() == before

    def test_non_object_body_rejected(self, harness):
        client, _gs, _state = harness
        r = client.post("/admin/api/routing/settings", json=[1, 2, 3])
        assert r.status_code == 400
