# SPDX-License-Identifier: Apache-2.0
"""Tests for the classification-family profiler (M4.5).

The MLX forward pass (mlx-embeddings load + ModernBERT) is not exercised here;
it needs the model weights on disk and Apple Silicon. These tests cover the
pure capability-score -> RouterFeatures mapping, the profiler factory switch,
and the async classify path with the sync forward stubbed out.
"""

import asyncio
import json
import types

from omlx.routing.profiler import RouterProfiler, make_profiler
from omlx.routing.profiler_capability import (
    DEFAULT_CAPABILITY_MODEL,
    CapabilityProfiler,
    features_from_scores,
)

# Full 6-axis capability vector in the model's label order.
_ALL_AXES = (
    "instruction_following",
    "coding",
    "math_reasoning",
    "world_knowledge",
    "planning_agentic",
    "creative_synthesis",
)


def _scores(**overrides) -> dict[str, float]:
    row = {axis: 0.0 for axis in _ALL_AXES}
    row.update(overrides)
    return row


def test_empty_scores_yield_null_features():
    f = features_from_scores({}, 0.5)
    assert f.domain is None
    assert f.complexity is None
    assert f.math is False
    assert f.code is False
    assert f.route_token is None


def test_code_boolean_at_threshold():
    below = features_from_scores(_scores(coding=0.49), 0.5)
    at = features_from_scores(_scores(coding=0.5), 0.5)
    assert below.code is False
    assert at.code is True


def test_math_boolean_uses_math_reasoning_axis():
    f = features_from_scores(_scores(math_reasoning=0.8), 0.5)
    assert f.math is True
    assert f.code is False


def test_domain_is_argmax_axis():
    f = features_from_scores(_scores(world_knowledge=0.9, coding=0.2), 0.5)
    assert f.domain == "world_knowledge"


def test_complexity_proxy_spans_1_to_5():
    lo = features_from_scores(_scores(), 0.5)  # all zero
    hi = features_from_scores(_scores(coding=1.0), 0.5)
    assert lo.complexity == 1
    assert hi.complexity == 5


def test_complexity_ignores_world_knowledge_and_instruction():
    # A pure factual / instruction prompt should not escalate on complexity;
    # only coding/math/planning drive the tier.
    f = features_from_scores(
        _scores(world_knowledge=1.0, instruction_following=1.0), 0.5
    )
    assert f.complexity == 1


def test_complexity_tracks_planning_axis():
    f = features_from_scores(_scores(planning_agentic=1.0), 0.5)
    assert f.complexity == 5


def test_route_token_always_none():
    f = features_from_scores(_scores(coding=0.9), 0.5)
    assert f.route_token is None


def test_threshold_is_configurable():
    strict = features_from_scores(_scores(coding=0.6), 0.75)
    loose = features_from_scores(_scores(coding=0.6), 0.5)
    assert strict.code is False
    assert loose.code is True


def test_make_profiler_defaults_to_generative():
    settings = types.SimpleNamespace(
        profiler_kind="generative", router_model="Supra-Router-51M"
    )
    profiler = make_profiler(settings)
    assert isinstance(profiler, RouterProfiler)
    assert profiler.needs_engine is True


def test_make_profiler_selects_capability():
    settings = types.SimpleNamespace(
        profiler_kind="capability",
        router_model="massaindustries/modernbert-capability-classifier",
        capability_threshold=0.4,
    )
    profiler = make_profiler(settings)
    assert isinstance(profiler, CapabilityProfiler)
    assert profiler.needs_engine is False
    assert profiler.threshold == 0.4


def test_make_profiler_unknown_kind_falls_back_to_generative():
    settings = types.SimpleNamespace(
        profiler_kind="something-else", router_model="Supra-Router-51M"
    )
    assert isinstance(make_profiler(settings), RouterProfiler)


def test_capability_profiler_defaults_model_when_blank():
    p = CapabilityProfiler("")
    assert p.model_id == DEFAULT_CAPABILITY_MODEL


def test_classify_maps_scores_and_ignores_engine(monkeypatch):
    profiler = CapabilityProfiler("some/model", threshold=0.5)

    def fake_sync(text: str) -> dict[str, float]:
        assert text == "write a python quicksort"
        return _scores(coding=0.92, math_reasoning=0.10)

    monkeypatch.setattr(profiler, "_classify_sync", fake_sync)

    # engine is passed as a sentinel that would explode if touched.
    sentinel = object()
    features, raw = asyncio.run(profiler.classify(sentinel, "write a python quicksort"))

    assert features.code is True
    assert features.math is False
    assert features.domain == "coding"
    assert features.complexity == 5
    # raw is JSON telemetry of the score vector.
    parsed = json.loads(raw)
    assert parsed["coding"] == 0.92
