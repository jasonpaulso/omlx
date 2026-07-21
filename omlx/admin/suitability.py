# SPDX-License-Identifier: Apache-2.0
"""Suitability sweep orchestration.

Bridges the accuracy-benchmark queue and the persistent SuitabilityStore:
sweeps enqueue baseline-mode runs through the existing queue (eviction
politeness comes free), and sink hooks harvest every completed result —
including manual bench runs — into the store. Load/serve failures are
recorded as first-class unhealthy entries, not missing rows.
"""

import hashlib
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from ..routing.store import SuitabilityStore, stale_records

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


_FINGERPRINT_TTL_S = 30.0
_fingerprint_cache: dict[str, tuple[float, str | None]] = {}


def _fingerprint_dir(path: Path) -> str | None:
    """Hash (name, size, mtime_ns) over every file in a model directory.

    Nanosecond mtime, not whole seconds: a re-quantize that happens to
    produce the same byte count within the same second would otherwise
    fingerprint identically.
    """
    parts: list[str] = []
    try:
        for f in sorted(path.rglob("*")):
            if not f.is_file():
                continue
            st = f.stat()
            parts.append(f"{f.relative_to(path)}:{st.st_size}:{st.st_mtime_ns}")
    except OSError as e:
        logger.debug("weights fingerprint: stat failed for %s: %s", path, e)
        return None
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def weights_fingerprint(model_id: str) -> str | None:
    """Identity stamp for the weights currently on disk, or None if unknown.

    Stamped onto every eval and prefill record so the suitability table can
    tell measurements of *these* weights from measurements of whatever used
    to live at this model id — a re-quantize or a re-download of an updated
    upstream release keeps the id and replaces the tensors, and nothing else
    about the record would show it.

    Cheap by construction: a stat walk (no file reads), memoized for
    `_FINGERPRINT_TTL_S` because the table endpoint polls every 5s. Size and
    mtime, not content — so a byte-identical copy that lands with fresh
    mtimes reads as changed. That errs toward a stale badge on unchanged
    weights, which costs a re-bench; the reverse would hide the thing this
    exists to surface.

    None means "can't tell" (no pool, unknown id, unreadable dir) and is
    never treated as stale.
    """
    if _engine_pool is None:
        return None
    now = time.monotonic()
    cached = _fingerprint_cache.get(model_id)
    if cached and now - cached[0] < _FINGERPRINT_TTL_S:
        return cached[1]
    fingerprint: str | None = None
    try:
        entry = _engine_pool.get_entry(model_id)
        model_path = getattr(entry, "model_path", None) if entry else None
        if model_path:
            fingerprint = _fingerprint_dir(Path(model_path))
    except Exception as e:  # noqa: BLE001 - staleness hinting is best-effort
        logger.debug("weights fingerprint: failed for %s: %s", model_id, e)
        fingerprint = None
    _fingerprint_cache[model_id] = (now, fingerprint)
    return fingerprint


def model_staleness(model_id: str, entry: dict) -> dict[str, Any]:
    """Which of a model's stored measurements predate the weights on disk.

    Returns ``{"current": <fingerprint|None>, "stale": bool, "records":
    [<bench>, ...]}``. Records stamped before fingerprinting existed carry
    None and are reported as unknown, not stale — otherwise every row
    predating this feature would light up at once.
    """
    current = weights_fingerprint(model_id)
    if current is None:
        return {"current": None, "stale": False, "records": []}
    stale = stale_records(entry, current)
    return {"current": current, "stale": bool(stale), "records": stale}


def _harvest_result(result_data: dict) -> None:
    """Record one completed benchmark result into the store."""
    if _store is None:
        return
    model_id = result_data.get("model_id")
    bench = result_data.get("benchmark")
    if not model_id or not bench:
        return
    baseline = bool(result_data.get("baseline", False))
    variant = result_data.get("variant")
    q_times = [
        q.get("time_s")
        for q in result_data.get("question_results", [])
        if isinstance(q.get("time_s"), (int, float))
    ]
    if variant:
        source = "settings_delta"
    elif baseline:
        source = "suitability_sweep"
    else:
        source = "manual_bench"
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
        source=source,
        run_id=result_data.get("run_id"),
        variant=variant,
        weights_fingerprint=weights_fingerprint(model_id),
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


def _has_baseline_for_bench(entry: dict, bench: str) -> bool:
    return any(
        rec.get("bench") == bench and rec.get("baseline")
        for rec in entry.get("evals", [])
    )


def start_sweep(
    models: list[str],
    benchmarks: dict[str, int],
    engine_pool: Any,
    batch_size: int = 1,
    only_missing: bool = False,
) -> dict:
    """Enqueue baseline-mode benchmark runs for each model; returns queue status.

    The existing queue serializes runs and evicts all models before each,
    so sweep politeness is inherited, not reimplemented.

    `only_missing` (P1-E gap-fill): additionally skip models that already
    have a baseline eval record for every bench in `benchmarks` -- a
    partially-covered model is still queued. Makes resuming a crashed
    sweep a single call instead of a hand-picked model list.
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
        entry = None
        if _store is not None:
            _store.ensure_model(model_id, size_gb=_model_size_gb(model_id))
            entry = _store.get_model(model_id)
            role = entry.get("role", "chat") if entry else "chat"
        # Companions/embedders/etc. are never benched standalone (a draft
        # model scored standalone is a category error, not a data point).
        # User role overrides via /api/suitability/role change eligibility.
        if role != "chat":
            skipped[model_id] = role
            continue
        if (
            only_missing
            and entry is not None
            and all(_has_baseline_for_bench(entry, bench) for bench in benchmarks)
        ):
            skipped[model_id] = "already_scored"
            continue
        eligible.append(model_id)

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


def start_delta_rescore(
    model_id: str,
    benchmark: str,
    sample_size: int,
    settings_override: dict,
    variant_label: str,
    engine_pool: Any,
    batch_size: int = 1,
    ensure_baseline: bool = True,
) -> dict:
    """Queue a settings-delta rescore for one model on one benchmark (M4.3).

    Runs the model with stock defaults plus one flipped load-time knob
    (`settings_override`, labeled `variant_label`) and stores the result as
    a `baseline=False` variant eval. When `ensure_baseline` and no baseline
    eval exists yet for this bench, a baseline run is queued first so the two
    are diffable. Reuses the accuracy queue's serialization + eviction.
    """
    from .accuracy_benchmark import (
        AccuracyBenchmarkRequest,
        add_to_queue,
        get_queue_status,
        start_next_from_queue,
    )

    role = "chat"
    entry = None
    if _store is not None:
        _store.ensure_model(model_id, size_gb=_model_size_gb(model_id))
        entry = _store.get_model(model_id)
        role = entry.get("role", "chat") if entry else "chat"
    if role != "chat":
        return {"error": f"model role is {role!r}, not chat", "queued": []}

    queued: list[str] = []
    need_baseline = ensure_baseline and not (
        entry is not None and _has_baseline_for_bench(entry, benchmark)
    )
    if need_baseline:
        add_to_queue(
            AccuracyBenchmarkRequest(
                model_id=model_id,
                benchmarks={benchmark: sample_size},
                batch_size=batch_size,
                baseline_mode=True,
            )
        )
        queued.append(f"baseline:{benchmark}")

    add_to_queue(
        AccuracyBenchmarkRequest(
            model_id=model_id,
            benchmarks={benchmark: sample_size},
            batch_size=batch_size,
            settings_override=settings_override,
            variant_label=variant_label,
        )
    )
    queued.append(f"{variant_label}:{benchmark}")
    start_next_from_queue(engine_pool)
    logger.info(
        "Settings-delta rescore queued: %s on %s (variant=%s, baseline_first=%s)",
        model_id,
        benchmark,
        variant_label,
        need_baseline,
    )
    return {"queued": queued, "model_id": model_id, **get_queue_status()}


def _speed_delta(baseline: float | None, variant: float | None) -> float | None:
    """variant - baseline (negative = the variant is faster). None if either
    side is missing a measurement."""
    if baseline is None or variant is None:
        return None
    return round(variant - baseline, 3)


def compute_deltas(model_id: str) -> list[dict]:
    """Diff each (bench, variant) against the latest baseline eval for that
    bench. Latest-by-date within each group. Returns one row per variant that
    has a comparable baseline; variants without a baseline are omitted.
    """
    if _store is None:
        return []
    entry = _store.get_model(model_id)
    if not entry:
        return []

    baselines: dict[str, dict] = {}
    variants: dict[tuple[str, str], dict] = {}
    for rec in entry.get("evals", []):
        bench = rec.get("bench")
        if not bench:
            continue
        date = rec.get("date") or ""
        if rec.get("baseline"):
            cur = baselines.get(bench)
            if cur is None or date > (cur.get("date") or ""):
                baselines[bench] = rec
        elif rec.get("variant"):
            key = (bench, rec["variant"])
            cur = variants.get(key)
            if cur is None or date > (cur.get("date") or ""):
                variants[key] = rec

    rows: list[dict] = []
    for (bench, variant), vrec in variants.items():
        brec = baselines.get(bench)
        if brec is None:
            continue
        rows.append(
            {
                "bench": bench,
                "axis": vrec.get("axis"),
                "variant": variant,
                "baseline_accuracy": brec.get("accuracy"),
                "variant_accuracy": vrec.get("accuracy"),
                "accuracy_delta": round(
                    (vrec.get("accuracy") or 0.0) - (brec.get("accuracy") or 0.0), 4
                ),
                "baseline_median_q_time_s": brec.get("median_q_time_s"),
                "variant_median_q_time_s": vrec.get("median_q_time_s"),
                "speed_delta_s": _speed_delta(
                    brec.get("median_q_time_s"), vrec.get("median_q_time_s")
                ),
                "baseline_load_s": brec.get("load_s"),
                "variant_load_s": vrec.get("load_s"),
                "baseline_n": brec.get("n"),
                "variant_n": vrec.get("n"),
                "baseline_date": brec.get("date"),
                "variant_date": vrec.get("date"),
            }
        )
    return sorted(rows, key=lambda r: (r["bench"], r["variant"]))
