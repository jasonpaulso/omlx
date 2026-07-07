# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /admin/api/models/{model_id}/unload during an active bench.

Incident: three manual admin unloads fired mid-benchmark, each running a
deep reset while the bench's engine (a different model) was mid-generation;
the bench's await never resolved. The unload endpoint's deep-reset side
effects aren't provably scoped to the unloaded model, so this guard blocks
ANY manual unload while an accuracy benchmark run is active.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.accuracy_benchmark as accuracy_benchmark
import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — triggers set_admin_getters

MODEL_ID = "test-model"


def _run_unload(model_id: str = MODEL_ID):
    return asyncio.run(admin_routes.unload_model(model_id, is_admin=True))


def _loaded_pool(model_id: str = MODEL_ID):
    pool = MagicMock(spec=[])
    entry = MagicMock()
    entry.engine = MagicMock()
    pool.get_entry = MagicMock(return_value=entry)
    pool._unload_engine = AsyncMock(return_value=None)
    return pool


class TestUnloadBlockedDuringActiveBench:
    def test_unload_returns_409_when_bench_active(self, monkeypatch):
        monkeypatch.setattr(accuracy_benchmark, "_queue_running", True)
        monkeypatch.setattr(accuracy_benchmark, "_current_model", "other-model")
        pool = _loaded_pool()

        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=pool),
            pytest.raises(HTTPException) as exc_info,
        ):
            _run_unload()

        assert exc_info.value.status_code == 409
        assert "other-model" in exc_info.value.detail
        pool._unload_engine.assert_not_called()

    def test_unload_blocked_for_any_model_not_just_the_active_one(self, monkeypatch):
        """The guard isn't scoped to the model the bench is using — unload of
        an unrelated, idle model is blocked too, since the deep-reset side
        effects aren't provably scoped to the unloaded model.
        """
        monkeypatch.setattr(accuracy_benchmark, "_queue_running", True)
        monkeypatch.setattr(accuracy_benchmark, "_current_model", "bench-model")
        pool = _loaded_pool("unrelated-model")

        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=pool),
            pytest.raises(HTTPException) as exc_info,
        ):
            _run_unload("unrelated-model")

        assert exc_info.value.status_code == 409
        pool._unload_engine.assert_not_called()


class TestUnloadAllowedWhenIdle:
    def test_unload_proceeds_when_no_bench_running(self, monkeypatch):
        monkeypatch.setattr(accuracy_benchmark, "_queue_running", False)
        monkeypatch.setattr(accuracy_benchmark, "_current_model", None)
        pool = _loaded_pool()

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = _run_unload()

        assert result["status"] == "ok"
        assert result["model_id"] == MODEL_ID
        pool._unload_engine.assert_called_once_with(MODEL_ID)

    def test_unload_404_when_model_not_found_still_applies(self, monkeypatch):
        """Existing behavior (model not found) must survive the new guard."""
        monkeypatch.setattr(accuracy_benchmark, "_queue_running", False)
        monkeypatch.setattr(accuracy_benchmark, "_current_model", None)
        pool = MagicMock(spec=[])
        pool.get_entry = MagicMock(return_value=None)

        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=pool),
            pytest.raises(HTTPException) as exc_info,
        ):
            _run_unload()

        assert exc_info.value.status_code == 404
