# SPDX-License-Identifier: Apache-2.0
"""Tests for /admin/api/suitability/table on-disk filtering.

The table endpoint surfaces only models present in the engine pool's
roster (the same discovery view the Models page uses); the store keeps
records for absent models so scores survive a re-download.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.admin import suitability as suitability_module
from omlx.routing.store import SuitabilityStore


class _FakePool:
    def __init__(self, model_ids):
        self.model_ids = list(model_ids)

    def get_model_ids(self):
        return list(self.model_ids)


def _make_store(tmp_path) -> SuitabilityStore:
    store = SuitabilityStore(str(tmp_path / "suitability.json"))
    store.load()
    for model_id in ("model-a", "model-gone"):
        store.ensure_model(model_id)
        store.set_role(model_id, "chat", source="user")
        store.record_eval(
            model_id,
            bench="toolcall",
            accuracy=0.8 if model_id == "model-a" else 0.9,
            n=30,
            baseline=True,
            thinking=False,
            time_s=40.0,
        )
    return store


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    pool = _FakePool(["model-a"])  # model-gone is scored but off disk

    monkeypatch.setattr(suitability_module, "_store", store)
    admin_routes._get_engine_pool = lambda: pool

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app), store, pool


class TestSuitabilityTableOnDiskFilter:
    def test_absent_model_hidden_from_models_and_rankings(self, client):
        c, store, _pool = client
        r = c.get("/admin/api/suitability/table")
        assert r.status_code == 200
        body = r.json()
        assert set(body["models"]) == {"model-a"}
        for _axis, ranked in body["rankings"].items():
            assert all(mid != "model-gone" for mid, _score in ranked)
        # agentic axis still ranks the on-disk model
        assert ["model-a", 0.8] in body["rankings"]["agentic"]

    def test_store_retains_absent_model_record(self, client):
        c, store, _pool = client
        c.get("/admin/api/suitability/table")
        assert "model-gone" in store.all_models()
        assert store.all_models()["model-gone"]["categories"]["agentic"] == 0.9

    def test_redownloaded_model_reappears_with_scores(self, client):
        c, _store, pool = client
        pool.model_ids.append("model-gone")
        body = c.get("/admin/api/suitability/table").json()
        assert set(body["models"]) == {"model-a", "model-gone"}
        assert ["model-gone", 0.9] in body["rankings"]["agentic"]

    def test_no_pool_fails_open_unfiltered(self, client):
        c, _store, _pool = client
        admin_routes._get_engine_pool = lambda: None
        body = c.get("/admin/api/suitability/table").json()
        assert set(body["models"]) == {"model-a", "model-gone"}


class TestSuitabilityClearEndpoint:
    def test_clear_wipes_scores_and_returns_entry(self, client):
        c, store, _pool = client
        r = c.post("/admin/api/suitability/clear", json={"model_id": "model-a"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["model"]["evals"] == []
        assert body["model"]["categories"] == {}
        assert body["model"]["cleared_at"]
        assert store.all_models()["model-a"]["categories"] == {}

    def test_cleared_model_drops_out_of_rankings(self, client):
        c, _store, _pool = client
        c.post("/admin/api/suitability/clear", json={"model_id": "model-a"})
        body = c.get("/admin/api/suitability/table").json()
        # Still listed (it's on disk), just unranked until re-benched.
        assert set(body["models"]) == {"model-a"}
        assert body["rankings"]["agentic"] == []

    def test_clear_leaves_other_models_alone(self, client):
        c, store, _pool = client
        c.post("/admin/api/suitability/clear", json={"model_id": "model-a"})
        assert store.all_models()["model-gone"]["categories"]["agentic"] == 0.9

    def test_clear_unknown_model_404s(self, client):
        c, _store, _pool = client
        r = c.post("/admin/api/suitability/clear", json={"model_id": "nope"})
        assert r.status_code == 404

    def test_clear_without_model_id_400s(self, client):
        c, _store, _pool = client
        assert c.post("/admin/api/suitability/clear", json={}).status_code == 400

    def test_clear_works_for_offdisk_model(self, client):
        # A model can be cleared while its weights are absent — the table
        # hides it, but the record (and the stale scores) still exist.
        c, store, _pool = client
        r = c.post("/admin/api/suitability/clear", json={"model_id": "model-gone"})
        assert r.status_code == 200
        assert store.all_models()["model-gone"]["categories"] == {}
