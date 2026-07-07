# SPDX-License-Identifier: Apache-2.0
"""Tests for passive idle-time sweeps (omlx/admin/idle_sweep.py, M4.4)."""

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
