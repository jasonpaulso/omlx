# SPDX-License-Identifier: Apache-2.0
"""Tests for the M8 prefill-throughput probe.

The salted-prompt builder and the tps math are unit-tested here; the real
engine timing validates only on a loaded box (Studio).
"""

from types import SimpleNamespace

import pytest

from omlx.routing.prefill_probe import (
    DEFAULT_DEPTHS,
    build_salted_prompt,
    probe_prefill,
    run_prefill_probe,
)
from omlx.routing.store import SuitabilityStore


def test_salted_prompt_is_unique_per_salt():
    a = build_salted_prompt(100, "aaaa")
    b = build_salted_prompt(100, "bbbb")
    assert a != b
    # Same salt is reproducible (single probe builds each prompt once).
    assert build_salted_prompt(100, "aaaa") == a


def test_salted_prompt_scales_with_depth():
    short = build_salted_prompt(10, "s")
    long = build_salted_prompt(1000, "s")
    assert len(long) > len(short)
    assert short.startswith("probe ")


class _FakeEngine:
    """Records prompts; returns a fixed prompt_tokens and controllable timing."""

    def __init__(self, *, prompt_tps=0.0, prompt_tokens=None, fail_depths=()):
        self.prompt_tps = prompt_tps
        self.prompt_tokens = prompt_tokens
        # Depths (≈ word counts) to raise on, matched by word count so the
        # marker can't collide with hex filler inside another depth's prompt.
        self.fail_depths = fail_depths
        self.prompts = []

    async def generate(self, *, prompt, max_tokens, temperature):
        self.prompts.append(prompt)
        assert max_tokens == 1
        n_words = prompt.count(" ")
        if any(abs(n_words - d) < 100 for d in self.fail_depths):
            raise RuntimeError("boom at depth")
        pt = self.prompt_tokens if self.prompt_tokens is not None else n_words
        return SimpleNamespace(prompt_tps=self.prompt_tps, prompt_tokens=pt)


@pytest.mark.asyncio
async def test_probe_prefers_native_prompt_tps():
    eng = _FakeEngine(prompt_tps=850.0)
    out = await probe_prefill(eng, depths=(2048,), salt="x")
    assert out[2048] == 850.0


@pytest.mark.asyncio
async def test_probe_wall_clocks_when_no_native_tps():
    # No native prompt_tps -> fall back to prompt_tokens / elapsed. Elapsed is
    # tiny and positive, so tps is a large finite number.
    eng = _FakeEngine(prompt_tps=0.0, prompt_tokens=5000)
    out = await probe_prefill(eng, depths=(8192,), salt="x")
    assert 8192 not in out or out[8192] > 0
    assert out[8192] > 0  # measured something positive


@pytest.mark.asyncio
async def test_probe_skips_failed_depth_keeps_others():
    eng = _FakeEngine(prompt_tps=500.0, fail_depths=(8192,))
    out = await probe_prefill(eng, depths=(2048, 8192, 24576), salt="x")
    assert 8192 not in out
    assert out[2048] == 500.0
    assert out[24576] == 500.0


@pytest.mark.asyncio
async def test_run_prefill_probe_persists(tmp_path):
    store = SuitabilityStore(tmp_path / "s.json")
    store.load()

    class _Pool:
        async def get_engine(self, model_id, *, force_lm, stamp_activity):
            return _FakeEngine(prompt_tps=300.0)

    out = await run_prefill_probe(_Pool(), store, "m", depths=(2048, 24576))
    assert out == {2048: 300.0, 24576: 300.0}
    pf = store.get_model("m")["prefill"]
    assert pf["2048"] == 300.0
    assert pf["24576"] == 300.0


@pytest.mark.asyncio
async def test_run_prefill_probe_returns_none_on_load_failure(tmp_path):
    store = SuitabilityStore(tmp_path / "s.json")
    store.load()

    class _Pool:
        async def get_engine(self, model_id, *, force_lm, stamp_activity):
            raise RuntimeError("cannot load")

    assert await run_prefill_probe(_Pool(), store, "m") is None
    assert store.get_model("m") is None or "prefill" not in (store.get_model("m") or {})


def test_default_depths_sorted_ascending():
    assert list(DEFAULT_DEPTHS) == sorted(DEFAULT_DEPTHS)
