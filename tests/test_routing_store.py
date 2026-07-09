# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.routing.store.SuitabilityStore and the role/axis taxonomy.

All tmp_path-based; no models or real inference needed.
"""

import json

from omlx.routing.store import (
    CATEGORY_AXES,
    DISPATCH_AXES,
    SuitabilityStore,
    classify_role,
)


def make_store(tmp_path, name: str = "suitability.json") -> SuitabilityStore:
    return SuitabilityStore(tmp_path / name)


# --- classify_role -----------------------------------------------------


def test_classify_role_draft_name_patterns():
    assert classify_role("qwen-dflash-3b", None) == "draft_companion"
    assert classify_role("llama-70b-assistant", None) == "draft_companion"
    assert classify_role("some-draft-model", None) == "draft_companion"


def test_classify_role_mtp_name_is_not_a_draft_signal():
    # Full chat models keep their MTP heads; "mtp" in the name must not
    # flag them as companions. Extracted drafter heads are caught by size.
    assert classify_role("Qwen3.6-27B-oQ4e-mtp", 16.0) == "chat"
    assert classify_role("model-mtp-head", None) == "chat"
    assert classify_role("qwen3-mtp-drafter", 1.2) == "draft_companion"


def test_classify_role_embedding_reranker_router_names():
    assert classify_role("bge-embed-large", 1.0) == "embedding"
    assert classify_role("bge-rerank-base", 1.0) == "reranker"
    assert classify_role("supra-router-51m", 0.1) == "router"


def test_classify_role_size_threshold():
    assert classify_role("some-chat-model", 4.9) == "draft_companion"
    assert classify_role("some-chat-model", 5.0) == "chat"
    assert classify_role("some-chat-model", None) == "chat"


def test_classify_role_name_pattern_wins_over_size():
    # A large model with a draft-pattern name is still a draft companion.
    assert classify_role("big-draft-70b", 40.0) == "draft_companion"


def test_category_axes_mapping():
    assert CATEGORY_AXES["gsm8k"] == "math"
    assert CATEGORY_AXES["mathqa"] == "math"
    assert CATEGORY_AXES["humaneval"] == "code"
    assert CATEGORY_AXES["mbpp"] == "code"
    assert CATEGORY_AXES["livecodebench"] == "code"
    for bench in ("mmlu", "mmlu_pro", "kmmlu", "cmmlu", "jmmlu", "truthfulqa"):
        assert CATEGORY_AXES[bench] == "knowledge"
    for bench in ("arc_challenge", "hellaswag", "winogrande"):
        assert CATEGORY_AXES[bench] == "reasoning"
    assert CATEGORY_AXES["bbq"] == "safety"
    assert CATEGORY_AXES["safetybench"] == "safety"


def test_toolcall_maps_to_agentic_axis():
    assert CATEGORY_AXES["toolcall"] == "agentic"


def test_dispatch_axes_include_agentic():
    assert DISPATCH_AXES == ("agentic", "code", "knowledge", "math")


# --- persistence roundtrip ----------------------------------------------


def test_load_missing_file_is_empty_store(tmp_path):
    store = make_store(tmp_path)
    store.load()
    assert store.all_models() == {}


def test_save_then_load_roundtrips(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "big-chat-70b",
        bench="gsm8k",
        accuracy=0.9,
        n=100,
        baseline=True,
        thinking=False,
        time_s=10.0,
    )

    store2 = make_store(tmp_path)
    store2.load()
    entry = store2.get_model("big-chat-70b")
    assert entry is not None
    assert entry["categories"]["math"] == 0.9
    assert entry["evals"][0]["bench"] == "gsm8k"


def test_save_stamps_host_once(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("chat-model", size_gb=10.0)
    store.save()
    raw = json.loads((tmp_path / "suitability.json").read_text())
    assert "hostname" in raw["host"]
    first_host = raw["host"]

    store.save()
    raw2 = json.loads((tmp_path / "suitability.json").read_text())
    assert raw2["host"] == first_host


def test_save_is_atomic_no_leftover_tmp_file(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("chat-model")
    store.save()
    assert not (tmp_path / "suitability.json.tmp").exists()
    assert (tmp_path / "suitability.json").exists()


def test_load_corrupt_file_tolerates_and_resets(tmp_path):
    path = tmp_path / "suitability.json"
    path.write_text("{not valid json")
    store = SuitabilityStore(path)
    store.load()
    assert store.all_models() == {}
    # Store still usable and writable after tolerating corruption.
    store.ensure_model("m")
    store.save()
    assert store.get_model("m") is not None


def test_load_malformed_structure_resets(tmp_path):
    path = tmp_path / "suitability.json"
    path.write_text(json.dumps({"not": "a valid schema"}))
    store = SuitabilityStore(path)
    store.load()
    assert store.all_models() == {}


def test_future_version_loads_read_only(tmp_path):
    path = tmp_path / "suitability.json"
    path.write_text(
        json.dumps(
            {
                "version": 999,
                "host": {},
                "models": {"m": {"role": "chat", "categories": {"math": 0.5}}},
            }
        )
    )
    store = SuitabilityStore(path)
    store.load()
    # Data is still readable best-effort.
    assert store.get_model("m")["categories"]["math"] == 0.5

    # But writes are refused, so the on-disk file is untouched.
    store.ensure_model("new-model")
    store.save()
    on_disk = json.loads(path.read_text())
    assert "new-model" not in on_disk["models"]
    assert on_disk["version"] == 999


# --- role heuristics + ensure_model / set_role --------------------------


def test_ensure_model_creates_with_heuristic_role(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("small-dflash-1b", size_gb=0.9)
    entry = store.get_model("small-dflash-1b")
    assert entry["role"] == "draft_companion"
    assert entry["role_source"] == "heuristic"
    assert entry["size_gb"] == 0.9


def test_ensure_model_refreshes_size_gb(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("chat-model", size_gb=3.0)
    assert store.get_model("chat-model")["role"] == "draft_companion"

    store.ensure_model("chat-model", size_gb=12.0)
    entry = store.get_model("chat-model")
    assert entry["size_gb"] == 12.0
    assert entry["role"] == "chat"


def test_set_role_user_override_never_clobbered_by_ensure_model(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("ambiguous-model", size_gb=2.0)
    assert store.get_model("ambiguous-model")["role"] == "draft_companion"

    store.set_role("ambiguous-model", "chat")
    entry = store.get_model("ambiguous-model")
    assert entry["role"] == "chat"
    assert entry["role_source"] == "user"

    # Even with a new size that would heuristically say draft_companion,
    # the user's role sticks.
    store.ensure_model("ambiguous-model", size_gb=1.0)
    entry = store.get_model("ambiguous-model")
    assert entry["role"] == "chat"
    assert entry["role_source"] == "user"


def test_set_role_creates_model_if_absent(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.set_role("new-model", "reranker")
    entry = store.get_model("new-model")
    assert entry["role"] == "reranker"
    assert entry["role_source"] == "user"


# --- record_eval / category derivation ----------------------------------


def test_record_eval_appends_and_derives_category(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    entry = store.get_model("m")
    assert len(entry["evals"]) == 1
    assert entry["evals"][0]["axis"] == "math"
    assert entry["categories"] == {"math": 0.8}
    assert entry["health"]["status"] == "ok"


def test_record_eval_unknown_bench_maps_to_other_axis(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="some_new_bench",
        accuracy=0.5,
        n=10,
        baseline=True,
        thinking=False,
        time_s=1.0,
    )
    entry = store.get_model("m")
    assert entry["evals"][0]["axis"] == "other"
    assert entry["categories"] == {"other": 0.5}


def test_category_derivation_uses_latest_per_bench(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.5,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
        date="2026-01-01T00:00:00+00:00",
    )
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.9,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
        date="2026-02-01T00:00:00+00:00",
    )
    entry = store.get_model("m")
    assert len(entry["evals"]) == 2
    assert entry["categories"]["math"] == 0.9


def test_category_derivation_ties_broken_by_list_order(tmp_path):
    store = make_store(tmp_path)
    store.load()
    same_date = "2026-01-01T00:00:00+00:00"
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.5,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
        date=same_date,
    )
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.7,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
        date=same_date,
    )
    # Later entry in list order wins the tie.
    assert store.get_model("m")["categories"]["math"] == 0.7


def test_category_derivation_ignores_non_baseline(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.9,
        n=50,
        baseline=False,
        thinking=False,
        time_s=5.0,
    )
    entry = store.get_model("m")
    assert entry["categories"] == {}
    assert len(entry["evals"]) == 1


def test_category_derivation_bench_with_only_non_baseline_contributes_nothing(
    tmp_path,
):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.9,
        n=50,
        baseline=False,
        thinking=False,
        time_s=5.0,
    )
    store.record_eval(
        "m",
        bench="humaneval",
        accuracy=0.6,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    entry = store.get_model("m")
    assert entry["categories"] == {"code": 0.6}


def test_category_derivation_multi_bench_axis_mean(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.record_eval(
        "m",
        bench="mathqa",
        accuracy=0.6,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    entry = store.get_model("m")
    assert entry["categories"]["math"] == 0.7


def test_record_eval_updates_perf_load_s(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
        load_s=12.5,
    )
    assert store.get_model("m")["perf"]["load_s"] == 12.5


def test_record_eval_full_provenance_fields(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=True,
        time_s=640.2,
        median_q_time_s=4.1,
        load_s=14.2,
        source="suitability_sweep",
        run_id="run-123",
    )
    record = store.get_model("m")["evals"][0]
    assert record["n"] == 50
    assert record["thinking"] is True
    assert record["median_q_time_s"] == 4.1
    assert record["source"] == "suitability_sweep"
    assert record["run_id"] == "run-123"
    assert "date" in record


# --- record_unhealthy ----------------------------------------------------


def test_record_unhealthy_sets_status_and_error(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_unhealthy("m", phase="load", message="OOM")
    entry = store.get_model("m")
    assert entry["health"]["status"] == "unhealthy"
    assert entry["health"]["last_error"]["phase"] == "load"
    assert entry["health"]["last_error"]["message"] == "OOM"
    assert "ts" in entry["health"]["last_error"]


def test_record_unhealthy_preserves_existing_evals_and_categories(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.record_unhealthy("m", phase="generate", message="timeout")
    entry = store.get_model("m")
    assert entry["health"]["status"] == "unhealthy"
    assert entry["categories"] == {"math": 0.8}
    assert len(entry["evals"]) == 1


def test_record_unhealthy_excludes_model_from_ranked(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.9,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.set_role("m", "chat")
    assert store.ranked("math") == [("m", 0.9)]

    store.record_unhealthy("m", phase="load", message="OOM")
    assert store.ranked("math") == []


# --- ranked ---------------------------------------------------------------


def test_ranked_orders_descending_by_score(tmp_path):
    store = make_store(tmp_path)
    store.load()
    for model_id, acc in (("a", 0.5), ("b", 0.9), ("c", 0.7)):
        store.record_eval(
            model_id,
            bench="gsm8k",
            accuracy=acc,
            n=50,
            baseline=True,
            thinking=False,
            time_s=5.0,
        )
        store.set_role(model_id, "chat")
    assert store.ranked("math") == [("b", 0.9), ("c", 0.7), ("a", 0.5)]


def test_ranked_filters_by_role(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "chat-model",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.set_role("chat-model", "chat")
    store.record_eval(
        "draft-model",
        bench="gsm8k",
        accuracy=0.95,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.set_role("draft-model", "draft_companion")

    assert store.ranked("math") == [("chat-model", 0.8)]
    assert store.ranked("math", roles=("draft_companion",)) == [("draft-model", 0.95)]
    assert store.ranked("math", roles=("chat", "draft_companion")) == [
        ("draft-model", 0.95),
        ("chat-model", 0.8),
    ]


def test_ranked_excludes_models_without_score_for_axis(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.8,
        n=50,
        baseline=True,
        thinking=False,
        time_s=5.0,
    )
    store.set_role("m", "chat")
    assert store.ranked("code") == []


def test_ranked_baseline_only_false_uses_latest_regardless_of_baseline(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m",
        bench="gsm8k",
        accuracy=0.9,
        n=50,
        baseline=False,
        thinking=False,
        time_s=5.0,
    )
    store.set_role("m", "chat")

    # Default (baseline_only=True) sees nothing: no baseline record exists.
    assert store.ranked("math") == []
    # baseline_only=False picks up the non-baseline record.
    assert store.ranked("math", baseline_only=False) == [("m", 0.9)]


# --- get_model / all_models ------------------------------------------------


def test_get_model_missing_returns_none(tmp_path):
    store = make_store(tmp_path)
    store.load()
    assert store.get_model("nope") is None


def test_all_models_returns_all_entries(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.ensure_model("a")
    store.ensure_model("b")
    assert set(store.all_models().keys()) == {"a", "b"}


# --- sample-size authority (n beats freshness) ------------------------------


def test_larger_n_beats_newer_smaller_run(tmp_path):
    """A quick n=4 spot-check must not displace an n=100 run's score."""
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m", bench="mmlu", accuracy=0.60, n=100, baseline=True,
        thinking=False, time_s=1.0, date="2026-07-01T00:00:00+00:00",
    )
    store.record_eval(
        "m", bench="mmlu", accuracy=1.00, n=4, baseline=True,
        thinking=False, time_s=1.0, date="2026-07-02T00:00:00+00:00",
    )
    assert store.get_model("m")["categories"]["knowledge"] == 0.60


def test_equal_n_newer_wins(tmp_path):
    store = make_store(tmp_path)
    store.load()
    store.record_eval(
        "m", bench="mmlu", accuracy=0.60, n=8, baseline=True,
        thinking=False, time_s=1.0, date="2026-07-01T00:00:00+00:00",
    )
    store.record_eval(
        "m", bench="mmlu", accuracy=0.80, n=8, baseline=True,
        thinking=False, time_s=1.0, date="2026-07-02T00:00:00+00:00",
    )
    assert store.get_model("m")["categories"]["knowledge"] == 0.80
