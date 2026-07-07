# SPDX-License-Identifier: Apache-2.0
"""Tests for the suitability sweep orchestrator (omlx/admin/suitability.py)."""

from types import SimpleNamespace

import pytest

from omlx.admin import accuracy_benchmark as ab
from omlx.admin import suitability


class FakePool:
    def __init__(self, sizes: dict[str, int]):
        self._sizes = sizes

    def get_entry(self, model_id):
        size = self._sizes.get(model_id)
        if size is None:
            return None
        return SimpleNamespace(estimated_size=size)


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    """init_suitability against a tmp store path and a fake pool."""
    pool = FakePool({"big-model": 60_000_000_000, "small-model": 2_000_000_000})
    store = suitability.init_suitability(
        pool, path=str(tmp_path / "suitability.json")
    )
    yield store, pool
    suitability.shutdown_suitability()


def _result(model_id="big-model", bench="mmlu", baseline=True, **over):
    data = {
        "model_id": model_id,
        "benchmark": bench,
        "accuracy": 0.75,
        "thinking_used": False,
        "total": 40,
        "correct": 30,
        "time_s": 120.0,
        "baseline": baseline,
        "run_id": "run-1",
        "load_s": 9.5,
        "question_results": [
            {"id": 1, "time_s": 2.0},
            {"id": 2, "time_s": 4.0},
            {"id": 3, "time_s": 3.0},
        ],
    }
    data.update(over)
    return data


class TestInit:
    def test_init_registers_sinks(self, store_env):
        assert ab._result_sink is suitability._harvest_result
        assert ab._run_status_sink is suitability._on_run_status

    def test_shutdown_unregisters(self, tmp_path):
        suitability.init_suitability(
            FakePool({}), path=str(tmp_path / "s.json")
        )
        suitability.shutdown_suitability()
        assert ab._result_sink is None
        assert ab._run_status_sink is None
        assert suitability.get_store() is None


class TestHarvest:
    def test_result_recorded_with_provenance(self, store_env):
        store, _ = store_env
        suitability._harvest_result(_result())
        entry = store.get_model("big-model")
        assert entry is not None
        rec = entry["evals"][-1]
        assert rec["bench"] == "mmlu"
        assert rec["baseline"] is True
        assert rec["n"] == 40
        assert rec["load_s"] == 9.5
        assert rec["run_id"] == "run-1"
        assert rec["median_q_time_s"] == 3.0
        assert rec["source"] == "suitability_sweep"
        assert entry["size_gb"] == 60.0
        assert entry["categories"]["knowledge"] == 0.75

    def test_manual_bench_tagged_and_excluded_from_scores(self, store_env):
        store, _ = store_env
        suitability._harvest_result(_result(baseline=False, accuracy=0.99))
        entry = store.get_model("big-model")
        assert entry["evals"][-1]["source"] == "manual_bench"
        assert "knowledge" not in entry.get("categories", {})

    def test_malformed_result_ignored(self, store_env):
        store, _ = store_env
        suitability._harvest_result({"accuracy": 0.5})  # no model/bench
        assert store.all_models() == {}

    def test_error_run_marks_unhealthy_and_unranked(self, store_env):
        store, _ = store_env
        suitability._harvest_result(_result())
        suitability._on_run_status("big-model", "error", "OOM during load")
        entry = store.get_model("big-model")
        assert entry["health"]["status"] == "unhealthy"
        assert entry["health"]["last_error"]["message"] == "OOM during load"
        assert store.ranked("knowledge") == []

    def test_cancelled_run_not_unhealthy(self, store_env):
        store, _ = store_env
        suitability._harvest_result(_result())
        suitability._on_run_status("big-model", "cancelled", None)
        assert store.get_model("big-model")["health"]["status"] == "ok"

    def test_sinks_inactive_after_shutdown(self, tmp_path):
        suitability.init_suitability(FakePool({}), path=str(tmp_path / "s.json"))
        suitability.shutdown_suitability()
        # Must be no-ops, not crashes
        suitability._harvest_result(_result())
        suitability._on_run_status("m", "error", "x")


class TestSweep:
    def test_sweep_enqueues_baseline_requests(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        status = suitability.start_sweep(
            ["big-model", "small-model"], {"mmlu_pro": 30}, pool
        )
        # Companion (<5GB) is excluded from standalone evals, not benched
        assert [q.model_id for q in queued] == ["big-model"]
        assert all(q.baseline_mode for q in queued)
        assert all(q.benchmarks == {"mmlu_pro": 30} for q in queued)
        assert status["queued"] == ["big-model"]
        assert status["skipped"] == {"small-model": "draft_companion"}
        assert store.get_model("small-model")["role"] == "draft_companion"
        assert store.get_model("big-model")["role"] == "chat"

    def test_user_role_override_restores_eligibility(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        store.set_role("small-model", "chat", source="user")
        status = suitability.start_sweep(["small-model"], {"mmlu": 10}, pool)
        assert status["queued"] == ["small-model"]
        assert status["skipped"] == {}
