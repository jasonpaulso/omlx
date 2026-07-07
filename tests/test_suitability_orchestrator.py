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
    store = suitability.init_suitability(pool, path=str(tmp_path / "suitability.json"))
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
        suitability.init_suitability(FakePool({}), path=str(tmp_path / "s.json"))
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


class TestSweepOnlyMissing:
    def test_only_missing_skips_fully_covered_model(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        suitability._harvest_result(
            _result(model_id="big-model", bench="mmlu_pro", baseline=True)
        )
        suitability._harvest_result(
            _result(model_id="big-model", bench="livecodebench", baseline=True)
        )

        status = suitability.start_sweep(
            ["big-model"],
            {"mmlu_pro": 30, "livecodebench": 10},
            pool,
            only_missing=True,
        )
        assert queued == []
        assert status["queued"] == []
        assert status["skipped"] == {"big-model": "already_scored"}

    def test_only_missing_queues_partially_covered_model(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        # Only one of the two requested benches has a baseline record.
        suitability._harvest_result(
            _result(model_id="big-model", bench="mmlu_pro", baseline=True)
        )

        status = suitability.start_sweep(
            ["big-model"],
            {"mmlu_pro": 30, "livecodebench": 10},
            pool,
            only_missing=True,
        )
        assert [q.model_id for q in queued] == ["big-model"]
        assert status["queued"] == ["big-model"]
        assert status["skipped"] == {}

    def test_only_missing_false_ignores_coverage(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        suitability._harvest_result(
            _result(model_id="big-model", bench="mmlu_pro", baseline=True)
        )

        status = suitability.start_sweep(
            ["big-model"], {"mmlu_pro": 30}, pool, only_missing=False
        )
        assert status["queued"] == ["big-model"]
        assert status["skipped"] == {}

    def test_only_missing_non_chat_still_skipped_by_role(self, store_env, monkeypatch):
        store, pool = store_env
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})

        status = suitability.start_sweep(
            ["small-model"], {"mmlu_pro": 30}, pool, only_missing=True
        )
        assert status["skipped"] == {"small-model": "draft_companion"}


class TestSettingsDeltaRescore:
    def _mock_queue(self, monkeypatch):
        queued = []
        monkeypatch.setattr(ab, "add_to_queue", queued.append)
        monkeypatch.setattr(ab, "start_next_from_queue", lambda p: "run-x")
        monkeypatch.setattr(ab, "get_queue_status", lambda: {"queue": []})
        return queued

    def test_queues_baseline_then_variant_when_no_baseline(
        self, store_env, monkeypatch
    ):
        store, pool = store_env
        queued = self._mock_queue(monkeypatch)
        result = suitability.start_delta_rescore(
            "big-model", "mmlu_pro", 30, {"mtp_enabled": True}, "mtp", pool
        )
        # baseline run first (baseline_mode, no override), then variant run
        assert len(queued) == 2
        assert queued[0].baseline_mode and queued[0].settings_override is None
        assert queued[1].settings_override == {"mtp_enabled": True}
        assert queued[1].variant_label == "mtp"
        assert not queued[1].baseline_mode
        assert result["queued"] == ["baseline:mmlu_pro", "mtp:mmlu_pro"]

    def test_skips_baseline_when_present(self, store_env, monkeypatch):
        store, pool = store_env
        suitability._harvest_result(
            _result(model_id="big-model", bench="mmlu_pro", baseline=True)
        )
        queued = self._mock_queue(monkeypatch)
        suitability.start_delta_rescore(
            "big-model", "mmlu_pro", 30, {"mtp_enabled": True}, "mtp", pool
        )
        assert len(queued) == 1
        assert queued[0].variant_label == "mtp"

    def test_ensure_baseline_false_queues_only_variant(self, store_env, monkeypatch):
        store, pool = store_env
        queued = self._mock_queue(monkeypatch)
        suitability.start_delta_rescore(
            "big-model",
            "mmlu_pro",
            30,
            {"mtp_enabled": True},
            "mtp",
            pool,
            ensure_baseline=False,
        )
        assert len(queued) == 1

    def test_non_chat_rejected(self, store_env, monkeypatch):
        store, pool = store_env
        self._mock_queue(monkeypatch)
        result = suitability.start_delta_rescore(
            "small-model", "mmlu_pro", 30, {"mtp_enabled": True}, "mtp", pool
        )
        assert "error" in result and result["queued"] == []

    def test_variant_result_tagged_and_excluded_from_scores(
        self, store_env, monkeypatch
    ):
        store, pool = store_env
        suitability._harvest_result(
            _result(bench="mmlu_pro", baseline=True, accuracy=0.75)
        )
        suitability._harvest_result(
            _result(bench="mmlu_pro", baseline=False, variant="mtp", accuracy=0.74)
        )
        entry = store.get_model("big-model")
        evals = entry["evals"]
        variant_recs = [e for e in evals if e.get("variant") == "mtp"]
        assert len(variant_recs) == 1
        assert variant_recs[0]["source"] == "settings_delta"
        assert variant_recs[0]["baseline"] is False
        # Category score still reflects the baseline (0.75), not the variant.
        assert entry["categories"]["knowledge"] == 0.75


class TestComputeDeltas:
    def test_delta_computed(self, store_env):
        store, pool = store_env
        suitability._harvest_result(
            _result(bench="mmlu_pro", baseline=True, accuracy=0.70)
        )
        suitability._harvest_result(
            _result(
                bench="mmlu_pro",
                baseline=False,
                variant="mtp",
                accuracy=0.68,
                question_results=[{"id": 1, "time_s": 1.0}, {"id": 2, "time_s": 1.0}],
            )
        )
        deltas = suitability.compute_deltas("big-model")
        assert len(deltas) == 1
        d = deltas[0]
        assert d["bench"] == "mmlu_pro" and d["variant"] == "mtp"
        assert d["accuracy_delta"] == round(0.68 - 0.70, 4)
        # baseline median 3.0 (2,3,4), variant median 1.0 -> faster by 2.0
        assert d["speed_delta_s"] == -2.0

    def test_variant_without_baseline_omitted(self, store_env):
        store, pool = store_env
        suitability._harvest_result(
            _result(bench="mmlu_pro", baseline=False, variant="mtp")
        )
        assert suitability.compute_deltas("big-model") == []

    def test_latest_baseline_wins(self, store_env):
        # Set dates explicitly via record_eval (the harvest sink stamps its
        # own timestamp, so date ordering must be exercised at the store).
        store, pool = store_env
        store.ensure_model("big-model")
        store.record_eval(
            "big-model",
            bench="mmlu_pro",
            accuracy=0.5,
            n=40,
            baseline=True,
            thinking=False,
            time_s=1.0,
            date="2026-01-01T00:00:00+00:00",
        )
        store.record_eval(
            "big-model",
            bench="mmlu_pro",
            accuracy=0.9,
            n=40,
            baseline=True,
            thinking=False,
            time_s=1.0,
            date="2026-02-01T00:00:00+00:00",
        )
        store.record_eval(
            "big-model",
            bench="mmlu_pro",
            accuracy=0.8,
            n=40,
            baseline=False,
            thinking=False,
            time_s=1.0,
            variant="mtp",
            date="2026-02-02T00:00:00+00:00",
        )
        d = suitability.compute_deltas("big-model")[0]
        assert d["baseline_accuracy"] == 0.9  # latest baseline, not 0.5
