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
import logging
import time
from typing import Any

from ..settings import RoutingIdleSweepSettings
from . import accuracy_benchmark as ab
from .suitability import start_sweep

logger = logging.getLogger(__name__)

# Beat for polling the sweep to drain (short, so the passive-sweep tag clears
# promptly after a normal drain rather than lingering a full poll interval).
_DRAIN_BEAT_S = 2.0


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

    No-op unless a passive sweep currently owns the bench queue. Injected into
    ``engine_pool.get_engine`` so a real request reclaims the GPU immediately.
    """
    if not ab.is_idle_sweep_active():
        return
    logger.info("Real request arrived; preempting passive idle sweep")
    ab.set_idle_sweep_active(False)
    await ab.cancel_queue_and_wait()


async def run_idle_sweep_loop(engine_pool: Any, cfg: RoutingIdleSweepSettings) -> None:
    """Background loop: after ``idle_after_s`` of quiet, kick a gap-fill
    (``only_missing``) suitability sweep over the roster. Runs until cancelled;
    every iteration is best-effort and never crashes the loop."""
    logger.info(
        "Idle-sweep loop started (idle_after_s=%.0f, poll=%.0f, benches=%s)",
        cfg.idle_after_s,
        cfg.poll_interval_s,
        dict(cfg.benchmarks),
    )
    # Treat startup as the first "activity" so the first sweep waits a full
    # idle window rather than firing the instant we boot.
    if engine_pool.last_request_monotonic is None:
        engine_pool._last_request_monotonic = time.monotonic()
    try:
        while True:
            await asyncio.sleep(cfg.poll_interval_s)
            try:
                await self_iteration(engine_pool, cfg)
            except Exception:  # noqa: BLE001 - loop must survive any iteration
                logger.exception("Idle-sweep loop iteration failed")
    except asyncio.CancelledError:
        logger.info("Idle-sweep loop cancelled")
        # If we were mid-sweep, drop the tag so a later request doesn't try to
        # preempt a queue we no longer own.
        ab.set_idle_sweep_active(False)
        raise


async def self_iteration(engine_pool: Any, cfg: RoutingIdleSweepSettings) -> None:
    """One idle check; starts and supervises a sweep if the server is quiet."""
    if ab.is_benchmark_active():
        return  # a user-initiated bench (or our own sweep) owns the queue
    if not should_start_sweep(
        now=time.monotonic(),
        last_request=engine_pool.last_request_monotonic,
        idle_after_s=cfg.idle_after_s,
        benchmark_active=False,
        any_active_requests=engine_pool.has_any_active_requests(),
    ):
        return

    models = engine_pool.get_model_ids()
    if not models:
        return

    # Tag BEFORE starting so a request racing in during start still preempts.
    ab.set_idle_sweep_active(True)
    try:
        logger.info(
            "Idle for >= %.0fs; starting gap-fill sweep over %d models",
            cfg.idle_after_s,
            len(models),
        )
        start_sweep(models, dict(cfg.benchmarks), engine_pool, only_missing=True)
        # Supervise until the sweep drains normally or a request preempts it
        # (preemptor clears the tag). start_next_from_queue sets _queue_running
        # synchronously, so is_benchmark_active() is already True here if
        # anything was queued.
        while ab.is_idle_sweep_active() and ab.is_benchmark_active():
            await asyncio.sleep(_DRAIN_BEAT_S)
    finally:
        ab.set_idle_sweep_active(False)
        # Reset the idle clock so we wait another full window before the next
        # sweep instead of spinning no-op gap-fills every poll interval.
        engine_pool._last_request_monotonic = time.monotonic()
