# SPDX-License-Identifier: Apache-2.0
"""Tests for passive idle-time sweeps (omlx/admin/idle_sweep.py, M4.4)."""

import asyncio

import pytest

from omlx.admin import accuracy_benchmark as ab
from omlx.admin import idle_sweep
from omlx.settings import RoutingIdleSweepSettings, RoutingSettings


class FakePool:
    def __init__(self, last_request=None, active=False, models=("m1", "m2")):
        self._last_request_monotonic = last_request
        self._active = active
        self._models = list(models)

    @property
    def last_request_monotonic(self):
        return self._last_request_monotonic

    def has_any_active_requests(self):
        return self._active

    def get_model_ids(self):
        return list(self._models)


@pytest.fixture(autouse=True)
def _reset_tag():
    ab.set_idle_sweep_active(False)
    yield
    ab.set_idle_sweep_active(False)


class TestConfig:
    def test_defaults_off(self):
        cfg = RoutingIdleSweepSettings()
        assert cfg.enabled is False
        assert cfg.idle_after_s == 600.0
        assert cfg.benchmarks == {"mmlu_pro": 30, "livecodebench": 10}

    def test_roundtrip_through_routing_settings(self):
        r = RoutingSettings.from_dict(
            {
                "idle_sweep": {
                    "enabled": True,
                    "idle_after_s": 120,
                    "benchmarks": {"mmlu": 5},
                }
            }
        )
        assert r.idle_sweep.enabled is True
        assert r.idle_sweep.idle_after_s == 120
        assert r.idle_sweep.benchmarks == {"mmlu": 5}
        assert r.to_dict()["idle_sweep"]["enabled"] is True


class TestPredicate:
    def test_idle_long_enough(self):
        assert idle_sweep.should_start_sweep(
            now=1000,
            last_request=100,
            idle_after_s=600,
            benchmark_active=False,
            any_active_requests=False,
        )

    def test_not_idle_yet(self):
        assert not idle_sweep.should_start_sweep(
            now=1000,
            last_request=900,
            idle_after_s=600,
            benchmark_active=False,
            any_active_requests=False,
        )

    def test_bench_active_blocks(self):
        assert not idle_sweep.should_start_sweep(
            now=1000,
            last_request=100,
            idle_after_s=600,
            benchmark_active=True,
            any_active_requests=False,
        )

    def test_in_flight_requests_block(self):
        assert not idle_sweep.should_start_sweep(
            now=1000,
            last_request=100,
            idle_after_s=600,
            benchmark_active=False,
            any_active_requests=True,
        )

    def test_no_reference_stamp_is_not_idle(self):
        assert not idle_sweep.should_start_sweep(
            now=1000,
            last_request=None,
            idle_after_s=600,
            benchmark_active=False,
            any_active_requests=False,
        )


class TestPreemptor:
    @pytest.mark.asyncio
    async def test_noop_when_no_passive_sweep(self, monkeypatch):
        called = []

        async def fake_cancel(*a, **k):
            called.append(True)

        monkeypatch.setattr(ab, "cancel_queue_and_wait", fake_cancel)
        await idle_sweep.preempt_idle_sweep()
        assert called == []

    @pytest.mark.asyncio
    async def test_aborts_and_clears_tag(self, monkeypatch):
        called = []

        async def fake_cancel(*a, **k):
            called.append(True)

        monkeypatch.setattr(ab, "cancel_queue_and_wait", fake_cancel)
        ab.set_idle_sweep_active(True)
        await idle_sweep.preempt_idle_sweep()
        assert called == [True]
        assert ab.is_idle_sweep_active() is False


class FakeStore:
    def __init__(self, models: dict):
        self._models = models

    def all_models(self):
        return self._models


class TestPrefillConfig:
    def test_defaults_off(self):
        cfg = RoutingIdleSweepSettings()
        assert cfg.prefill_probe is False

    def test_roundtrip_through_routing_settings(self):
        r = RoutingSettings.from_dict({"idle_sweep": {"prefill_probe": True}})
        assert r.idle_sweep.prefill_probe is True
        assert r.to_dict()["idle_sweep"]["prefill_probe"] is True


class TestPrefillGapFill:
    @pytest.mark.asyncio
    async def test_selects_only_enabled_and_missing_prefill(self, monkeypatch):
        probed = []

        async def fake_probe(engine_pool, store, model_id):
            probed.append(model_id)

        monkeypatch.setattr("omlx.routing.prefill_probe.run_prefill_probe", fake_probe)
        store = FakeStore(
            {
                "m1": {"size_gb": 5.0},  # enabled, missing prefill
                "m2": {"size_gb": 1.0, "prefill": {"2048": 100.0}},  # already probed
                "m3": {"size_gb": 2.0},  # not enabled
                "m4": {"size_gb": 3.0},  # enabled but not on disk
            }
        )
        monkeypatch.setattr("omlx.admin.suitability.get_store", lambda: store)
        pool = FakePool(last_request=0.0, models=("m1", "m2", "m3"))
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        await idle_sweep.run_prefill_gap_fill(pool, cfg, lambda: {"m1", "m2", "m4"})
        assert probed == ["m1"]

    @pytest.mark.asyncio
    async def test_sorts_smallest_first(self, monkeypatch):
        probed = []

        async def fake_probe(engine_pool, store, model_id):
            probed.append(model_id)

        monkeypatch.setattr("omlx.routing.prefill_probe.run_prefill_probe", fake_probe)
        store = FakeStore(
            {
                "big": {"size_gb": 40.0},
                "small": {"size_gb": 2.0},
                "mid": {"size_gb": 10.0},
            }
        )
        monkeypatch.setattr("omlx.admin.suitability.get_store", lambda: store)
        pool = FakePool(last_request=0.0, models=("big", "small", "mid"))
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        await idle_sweep.run_prefill_gap_fill(
            pool, cfg, lambda: {"big", "small", "mid"}
        )
        assert probed == ["small", "mid", "big"]

    @pytest.mark.asyncio
    async def test_stops_early_when_no_longer_idle(self, monkeypatch):
        probed = []

        async def fake_probe(engine_pool, store, model_id):
            probed.append(model_id)
            # A request "arrives" after the first probe.
            pool._active = True

        monkeypatch.setattr("omlx.routing.prefill_probe.run_prefill_probe", fake_probe)
        store = FakeStore(
            {
                "m1": {"size_gb": 1.0},
                "m2": {"size_gb": 2.0},
            }
        )
        monkeypatch.setattr("omlx.admin.suitability.get_store", lambda: store)
        pool = FakePool(last_request=0.0, models=("m1", "m2"))
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        await idle_sweep.run_prefill_gap_fill(pool, cfg, lambda: {"m1", "m2"})
        assert probed == ["m1"]


class TestPrefillPreemption:
    @pytest.mark.asyncio
    async def test_preempt_cancels_probe_task(self, monkeypatch):
        started = asyncio.Event()
        cancelled = False

        async def hang_forever():
            nonlocal cancelled
            started.set()
            try:
                await asyncio.sleep(1000)
            except asyncio.CancelledError:
                cancelled = True
                raise

        ab.set_idle_sweep_active(True)
        idle_sweep._probe_task = asyncio.create_task(hang_forever())
        await started.wait()

        await idle_sweep.preempt_idle_sweep()
        assert cancelled is True
        assert ab.is_idle_sweep_active() is False


class TestIteration:
    @pytest.mark.asyncio
    async def test_starts_sweep_when_idle(self, monkeypatch):
        calls = {}

        def fake_start(models, benchmarks, pool, only_missing=False):
            # Tag must be set while the sweep is being kicked.
            calls["tagged"] = ab.is_idle_sweep_active()
            calls["models"] = models
            calls["only_missing"] = only_missing

        monkeypatch.setattr(idle_sweep, "start_sweep", fake_start)
        pool = FakePool(last_request=0.0)  # now - 0 >> idle_after_s
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        await idle_sweep.self_iteration(pool, cfg)
        assert calls["tagged"] is True
        assert calls["models"] == ["m1", "m2"]
        assert calls["only_missing"] is True
        # Tag cleared after drain; idle clock reset (anti-spin).
        assert ab.is_idle_sweep_active() is False
        assert pool._last_request_monotonic is not None

    @pytest.mark.asyncio
    async def test_skips_when_not_idle(self, monkeypatch):
        calls = []
        monkeypatch.setattr(idle_sweep, "start_sweep", lambda *a, **k: calls.append(1))
        pool = FakePool(last_request=None)  # no reference stamp -> not idle
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)
        await idle_sweep.self_iteration(pool, cfg)
        assert calls == []

    @pytest.mark.asyncio
    async def test_skips_when_in_flight(self, monkeypatch):
        calls = []
        monkeypatch.setattr(idle_sweep, "start_sweep", lambda *a, **k: calls.append(1))
        pool = FakePool(last_request=0.0, active=True)
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)
        await idle_sweep.self_iteration(pool, cfg)
        assert calls == []

    @pytest.mark.asyncio
    async def test_prefill_probe_off_never_calls_gap_fill(self, monkeypatch):
        gap_fill_calls = []
        monkeypatch.setattr(idle_sweep, "start_sweep", lambda *a, **k: None)
        monkeypatch.setattr(
            idle_sweep,
            "run_prefill_gap_fill",
            lambda *a, **k: gap_fill_calls.append(1),
        )
        pool = FakePool(last_request=0.0)
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)
        assert cfg.prefill_probe is False

        await idle_sweep.self_iteration(pool, cfg, lambda: {"m1", "m2"})
        assert gap_fill_calls == []

    @pytest.mark.asyncio
    async def test_prefill_probe_on_calls_gap_fill_when_idle(self, monkeypatch):
        gap_fill_calls = []

        async def fake_gap_fill(engine_pool, cfg, enabled_getter):
            gap_fill_calls.append(enabled_getter())

        monkeypatch.setattr(idle_sweep, "start_sweep", lambda *a, **k: None)
        monkeypatch.setattr(idle_sweep, "run_prefill_gap_fill", fake_gap_fill)
        pool = FakePool(last_request=0.0, models=())
        cfg = RoutingIdleSweepSettings(
            enabled=True, idle_after_s=1.0, prefill_probe=True
        )

        await idle_sweep.self_iteration(pool, cfg, lambda: {"m1"})
        assert gap_fill_calls == [{"m1"}]

    @pytest.mark.asyncio
    async def test_idle_clock_untouched_when_nothing_ran(self, monkeypatch):
        # No models on disk and prefill off -> no passive activity -> the idle
        # clock must be left exactly as-is (pre-M8 early-return semantics).
        monkeypatch.setattr(idle_sweep, "start_sweep", lambda *a, **k: None)
        pool = FakePool(last_request=0.0, models=())
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        await idle_sweep.self_iteration(pool, cfg, lambda: {"m1"})
        assert pool._last_request_monotonic == 0.0  # untouched

    @pytest.mark.asyncio
    async def test_idle_clock_reset_when_start_sweep_raises(self, monkeypatch):
        # A raising start_sweep must still reset the idle clock (matches the
        # pre-M8 semantics where the reset lived in the sweep's own finally),
        # else the loop retries every poll interval instead of a full window.
        def boom(*a, **k):
            raise RuntimeError("sweep blew up")

        monkeypatch.setattr(idle_sweep, "start_sweep", boom)
        pool = FakePool(last_request=0.0, models=("m1",))
        cfg = RoutingIdleSweepSettings(enabled=True, idle_after_s=1.0)

        with pytest.raises(RuntimeError):
            await idle_sweep.self_iteration(pool, cfg)
        assert pool._last_request_monotonic != 0.0  # reset despite the raise
        assert ab.is_idle_sweep_active() is False  # tag not leaked
