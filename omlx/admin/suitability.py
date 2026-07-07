# SPDX-License-Identifier: Apache-2.0
"""Suitability sweep orchestration.

Bridges the accuracy-benchmark queue and the persistent SuitabilityStore:
sweeps enqueue baseline-mode runs through the existing queue (eviction
politeness comes free), and sink hooks harvest every completed result —
including manual bench runs — into the store. Load/serve failures are
recorded as first-class unhealthy entries, not missing rows.
"""

import logging
import statistics
from typing import Any

from ..routing.store import SuitabilityStore

logger = logging.getLogger(__name__)

_store: SuitabilityStore | None = None
_engine_pool: Any = None


def init_suitability(engine_pool: Any, path: str | None = None) -> SuitabilityStore:
    """Create the store, register harvest sinks, and return the store.

    Called once from the server lifespan. Always on: the store only writes
    when benchmark runs complete, so idle cost is zero.
    """
    global _store, _engine_pool
    from .accuracy_benchmark import set_result_sink, set_run_status_sink

    _engine_pool = engine_pool
    _store = SuitabilityStore(path) if path else SuitabilityStore()
    _store.load()
    set_result_sink(_harvest_result)
    set_run_status_sink(_on_run_status)
    return _store


def get_store() -> SuitabilityStore | None:
    return _store


def shutdown_suitability() -> None:
    global _store, _engine_pool
    from .accuracy_benchmark import set_result_sink, set_run_status_sink

    set_result_sink(None)
    set_run_status_sink(None)
    _store = None
    _engine_pool = None


def _model_size_gb(model_id: str) -> float | None:
    if _engine_pool is None:
        return None
    try:
        entry = _engine_pool.get_entry(model_id)
        size_bytes = getattr(entry, "estimated_size", None) if entry else None
        return round(size_bytes / 1e9, 2) if size_bytes else None
    except Exception:  # size is best-effort metadata only
        return None


def _harvest_result(result_data: dict) -> None:
    """Record one completed benchmark result into the store."""
    if _store is None:
        return
    model_id = result_data.get("model_id")
    bench = result_data.get("benchmark")
    if not model_id or not bench:
        return
    baseline = bool(result_data.get("baseline", False))
    q_times = [
        q.get("time_s")
        for q in result_data.get("question_results", [])
        if isinstance(q.get("time_s"), (int, float))
    ]
    _store.ensure_model(model_id, size_gb=_model_size_gb(model_id))
    _store.record_eval(
        model_id,
        bench=bench,
        accuracy=float(result_data.get("accuracy", 0.0)),
        n=int(result_data.get("total", 0)),
        baseline=baseline,
        thinking=bool(result_data.get("thinking_used", False)),
        time_s=float(result_data.get("time_s", 0.0)),
        median_q_time_s=(round(statistics.median(q_times), 3) if q_times else None),
        load_s=result_data.get("load_s"),
        source="suitability_sweep" if baseline else "manual_bench",
        run_id=result_data.get("run_id"),
    )


def _on_run_status(model_id: str, status: str, error_message: str | None) -> None:
    """Record failed runs as unhealthy. Cancellations are not failures."""
    if _store is None or status != "error":
        return
    _store.record_unhealthy(
        model_id,
        phase="bench_run",
        message=error_message or "benchmark run failed",
    )


def start_sweep(
    models: list[str],
    benchmarks: dict[str, int],
    engine_pool: Any,
    batch_size: int = 1,
) -> dict:
    """Enqueue baseline-mode benchmark runs for each model; returns queue status.

    The existing queue serializes runs and evicts all models before each,
    so sweep politeness is inherited, not reimplemented.
    """
    from .accuracy_benchmark import (
        AccuracyBenchmarkRequest,
        add_to_queue,
        get_queue_status,
        start_next_from_queue,
    )

    skipped: dict[str, str] = {}
    eligible: list[str] = []
    for model_id in models:
        role = "chat"
        if _store is not None:
            _store.ensure_model(model_id, size_gb=_model_size_gb(model_id))
            entry = _store.get_model(model_id)
            role = entry.get("role", "chat") if entry else "chat"
        # Companions/embedders/etc. are never benched standalone (a draft
        # model scored standalone is a category error, not a data point).
        # User role overrides via /api/suitability/role change eligibility.
        if role == "chat":
            eligible.append(model_id)
        else:
            skipped[model_id] = role

    for model_id in eligible:
        add_to_queue(
            AccuracyBenchmarkRequest(
                model_id=model_id,
                benchmarks=benchmarks,
                batch_size=batch_size,
                baseline_mode=True,
            )
        )
    if eligible:
        start_next_from_queue(engine_pool)
    logger.info(
        "Suitability sweep queued: %d model(s) x %s (skipped non-chat: %s)",
        len(eligible),
        list(benchmarks),
        skipped or "none",
    )
    return {"queued": eligible, "skipped": skipped, **get_queue_status()}
