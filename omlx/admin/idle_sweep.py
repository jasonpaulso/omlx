# SPDX-License-Identifier: Apache-2.0
"""Passive idle-time suitability sweeps (M4.4).

A background loop runs gap-fill suitability benchmarks while the server is
quiet. A real request arriving mid-sweep preempts it immediately (via the
engine-pool preemptor) and is served. Strictly opt-in
(``routing.idle_sweep.enabled``): benching evicts and holds models.

Wiring (server lifespan):
- ``engine_pool.set_idle_sweep_preemptor(preempt_idle_sweep)`` so a real
  request aborts an in-flight passive sweep before competing for the GPU.
- ``asyncio.create_task(run_idle_sweep_loop(pool, cfg))`` when enabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from ..settings import RoutingIdleSweepSettings
from . import accuracy_benchmark as ab
from .suitability import start_sweep

logger = logging.getLogger(__name__)

# Beat for polling the sweep to drain (short, so the passive-sweep tag clears
# promptly after a normal drain rather than lingering a full poll interval).
_DRAIN_BEAT_S = 2.0

# In-flight prefill gap-fill pass (M8 auto-probe), tracked the same way the
# accuracy-bench queue tracks its run task, so `preempt_idle_sweep` can
# cancel whichever passive activity currently owns the tag.
_probe_task: asyncio.Task | None = None


def should_start_sweep(
    *,
    now: float,
    last_request: float | None,
    idle_after_s: float,
    benchmark_active: bool,
    any_active_requests: bool,
) -> bool:
    """Pure idle predicate. Idle == quiet for at least ``idle_after_s`` with
    nothing in flight and no bench already running. ``last_request is None``
    (no reference stamp yet) is treated as not-idle so we never sweep before a
    baseline is established (the loop stamps startup time on entry)."""
    if benchmark_active or any_active_requests:
        return False
    if last_request is None:
        return False
    return (now - last_request) >= idle_after_s


async def preempt_idle_sweep() -> None:
    """Abort an in-flight passive idle sweep and await its teardown.

    No-op unless a passive sweep currently owns the bench queue or the
    prefill gap-fill pass is running. Injected into ``engine_pool.get_engine``
    so a real request reclaims the GPU immediately.
    """
    if not ab.is_idle_sweep_active():
        return
    logger.info("Real request arrived; preempting passive idle sweep")
    ab.set_idle_sweep_active(False)
    if _probe_task is not None and not _probe_task.done():
        _probe_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _probe_task
        return
    await ab.cancel_queue_and_wait()


async def run_idle_sweep_loop(
    engine_pool: Any,
    cfg: RoutingIdleSweepSettings,
    enabled_getter: Callable[[], Any] | None = None,
) -> None:
    """Background loop: after ``idle_after_s`` of quiet, kick a gap-fill
    (``only_missing``) suitability sweep over the roster. Runs until cancelled;
    every iteration is best-effort and never crashes the loop.

    ``enabled_getter`` (routing-enabled model ids) is only consulted when
    ``cfg.prefill_probe`` is on; it feeds the M8 prefill gap-fill pass.
    """
    logger.info(
        "Idle-sweep loop started (idle_after_s=%.0f, poll=%.0f, benches=%s, "
        "prefill_probe=%s)",
        cfg.idle_after_s,
        cfg.poll_interval_s,
        dict(cfg.benchmarks),
        cfg.prefill_probe,
    )
    # Treat startup as the first "activity" so the first sweep waits a full
    # idle window rather than firing the instant we boot.
    if engine_pool.last_request_monotonic is None:
        engine_pool._last_request_monotonic = time.monotonic()
    try:
        while True:
            await asyncio.sleep(cfg.poll_interval_s)
            try:
                await self_iteration(engine_pool, cfg, enabled_getter)
            except Exception:  # noqa: BLE001 - loop must survive any iteration
                logger.exception("Idle-sweep loop iteration failed")
    except asyncio.CancelledError:
        logger.info("Idle-sweep loop cancelled")
        # If we were mid-sweep, drop the tag so a later request doesn't try to
        # preempt a queue we no longer own.
        ab.set_idle_sweep_active(False)
        raise


def _still_idle(engine_pool: Any, cfg: RoutingIdleSweepSettings) -> bool:
    return should_start_sweep(
        now=time.monotonic(),
        last_request=engine_pool.last_request_monotonic,
        idle_after_s=cfg.idle_after_s,
        benchmark_active=False,
        any_active_requests=engine_pool.has_any_active_requests(),
    )


async def self_iteration(
    engine_pool: Any,
    cfg: RoutingIdleSweepSettings,
    enabled_getter: Callable[[], Any] | None = None,
) -> None:
    """One idle check; starts and supervises a sweep if the server is quiet."""
    if ab.is_benchmark_active():
        return  # a user-initiated bench (or our own sweep) owns the queue
    if not _still_idle(engine_pool, cfg):
        return

    models = engine_pool.get_model_ids()
    ran_passive_activity = False
    try:
        if models:
            ran_passive_activity = True
            # Tag BEFORE starting so a request racing in during start still
            # preempts.
            ab.set_idle_sweep_active(True)
            try:
                logger.info(
                    "Idle for >= %.0fs; starting gap-fill sweep over %d models",
                    cfg.idle_after_s,
                    len(models),
                )
                start_sweep(
                    models, dict(cfg.benchmarks), engine_pool, only_missing=True
                )
                # Supervise until the sweep drains normally or a request
                # preempts it (preemptor clears the tag). start_next_from_queue
                # sets _queue_running synchronously, so is_benchmark_active() is
                # already True here if anything was queued.
                while ab.is_idle_sweep_active() and ab.is_benchmark_active():
                    await asyncio.sleep(_DRAIN_BEAT_S)
            finally:
                ab.set_idle_sweep_active(False)

        # M8 prefill gap-fill: only if enabled and the box is still idle (the
        # accuracy sweep above may have been preempted, or skipped entirely
        # because there were no models to sweep).
        if (
            cfg.prefill_probe
            and enabled_getter is not None
            and _still_idle(engine_pool, cfg)
        ):
            ran_passive_activity = True
            await run_prefill_gap_fill(engine_pool, cfg, enabled_getter)
    finally:
        # Reset the idle clock once, after all passive idle activity finishes,
        # so we wait another full window before the next sweep instead of
        # spinning no-op gap-fills every poll interval. In a `finally` so a
        # raising start_sweep still resets it (matches the pre-M8 semantics
        # where the reset lived in the accuracy sweep's own finally); left
        # untouched when nothing ran (no models, prefill off/no-op).
        if ran_passive_activity:
            engine_pool._last_request_monotonic = time.monotonic()


async def run_prefill_gap_fill(
    engine_pool: Any,
    cfg: RoutingIdleSweepSettings,
    enabled_getter: Callable[[], Any],
) -> None:
    """Probe M8 prefill throughput for routing-enabled models missing it.

    Targets are the intersection of routing-enabled ids and the on-disk
    roster that have no ``prefill`` record in the suitability store (this
    also catches requantized models, which get a new id and so start with no
    prior probe). Smallest-first by ``size_gb`` to minimize how long a
    preempting request has to wait for the in-flight probe to tear down.
    Runs as its own cancellable task so ``preempt_idle_sweep`` can cancel it
    mid-generation, same as the accuracy-bench queue.
    """
    global _probe_task
    from .suitability import get_store

    store = get_store()
    if store is None:
        return

    try:
        enabled = set(enabled_getter() or ())
    except Exception:  # noqa: BLE001 - never break the idle loop
        logger.exception("Prefill gap-fill: enabled_getter failed")
        return
    if not enabled:
        return
    on_disk = set(engine_pool.get_model_ids())
    all_models = store.all_models()
    targets = [
        mid for mid in enabled & on_disk if not all_models.get(mid, {}).get("prefill")
    ]
    if not targets:
        return
    targets.sort(key=lambda mid: all_models.get(mid, {}).get("size_gb") or 0.0)

    logger.info(
        "Prefill gap-fill: probing %d model(s) missing prefill data", len(targets)
    )
    ab.set_idle_sweep_active(True)
    try:
        _probe_task = asyncio.create_task(
            _run_prefill_gap_fill_pass(engine_pool, cfg, store, targets)
        )
        with contextlib.suppress(asyncio.CancelledError):
            await _probe_task
    finally:
        _probe_task = None
        ab.set_idle_sweep_active(False)


async def _run_prefill_gap_fill_pass(
    engine_pool: Any,
    cfg: RoutingIdleSweepSettings,
    store: Any,
    targets: list[str],
) -> None:
    from ..routing.prefill_probe import run_prefill_probe

    for model_id in targets:
        if not _still_idle(engine_pool, cfg):
            logger.info("Prefill gap-fill: request arrived, stopping pass")
            return
        try:
            await run_prefill_probe(engine_pool, store, model_id)
        except Exception:  # noqa: BLE001 - one bad probe must not kill the pass
            logger.exception("Prefill gap-fill: probe failed for %s", model_id)
