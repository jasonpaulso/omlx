# SPDX-License-Identifier: Apache-2.0
"""Tests for N-way table dispatch (omlx/routing/table.py)."""

from omlx.routing.profiler import RouterFeatures
from omlx.routing.table import TableChoice, axis_for, choose


def feats(complexity=2, math=False, code=False):
    return RouterFeatures(
        domain=None, complexity=complexity, math=math, code=code, route_token=None
    )


def entry(
    role="chat",
    healthy=True,
    categories=None,
    median_q_time_s=None,
    baseline=True,
):
    evals = []
    if median_q_time_s is not None:
        evals.append(
            {
                "bench": "mmlu",
                "baseline": baseline,
                "median_q_time_s": median_q_time_s,
            }
        )
    return {
        "role": role,
        "health": {"status": "ok" if healthy else "unhealthy"},
        "categories": categories or {},
        "evals": evals,
    }


class TestAxisFor:
    def test_code_beats_math(self):
        assert axis_for(feats(math=True, code=True)) == "code"

    def test_math(self):
        assert axis_for(feats(math=True)) == "math"

    def test_default_knowledge(self):
        assert axis_for(feats()) == "knowledge"
        assert axis_for(None) == "knowledge"


class TestChoose:
    def test_axis_leader_wins(self):
        models = {
            "coder": entry(categories={"code": 0.9, "knowledge": 0.5}),
            "general": entry(categories={"code": 0.6, "knowledge": 0.8}),
        }
        c = choose(feats(code=True), models, set(), escalate_at=4)
        assert c.target == "coder"
        assert c.rule == "table:code"
        assert ("coder", 0.9) in c.candidates

    def test_unhealthy_and_non_chat_excluded(self):
        models = {
            "sick": entry(healthy=False, categories={"code": 0.99}),
            "draft": entry(role="draft_companion", categories={"code": 0.98}),
            "ok": entry(categories={"code": 0.5}),
        }
        c = choose(feats(code=True), models, set(), escalate_at=4)
        assert c.target == "ok"

    def test_thinking_lane_excluded_from_interactive(self):
        models = {
            "thinker": entry(categories={"math": 0.95}, median_q_time_s=120.0),
            "fast": entry(categories={"math": 0.85}, median_q_time_s=4.0),
        }
        c = choose(
            feats(math=True),
            models,
            set(),
            escalate_at=4,
            max_interactive_median_q_time_s=30.0,
        )
        assert c.target == "fast"

    def test_thinking_lane_latency_from_baseline_records_only(self):
        # Non-baseline latency records must not exclude a model
        models = {"m": entry(categories={"math": 0.9})}
        models["m"]["evals"].append(
            {"bench": "gsm8k", "baseline": False, "median_q_time_s": 500.0}
        )
        c = choose(feats(math=True), models, set(), escalate_at=4)
        assert c.target == "m"

    def test_escalation_tier_uses_overall_score(self):
        models = {
            "coder": entry(categories={"code": 0.9, "knowledge": 0.4}),
            "frontier": entry(categories={"code": 0.85, "knowledge": 0.9}),
        }
        c = choose(feats(complexity=5, code=True), models, set(), escalate_at=4)
        assert c.target == "frontier"
        assert c.rule == "table:escalate>=4"

    def test_residency_tiebreak_within_epsilon(self):
        models = {
            "leader": entry(categories={"knowledge": 0.80}),
            "resident": entry(categories={"knowledge": 0.79}),
        }
        c = choose(
            feats(),
            models,
            {"resident"},
            escalate_at=4,
            residency_epsilon=0.02,
        )
        assert c.target == "resident"

    def test_residency_does_not_beat_meaningfully_better_cold(self):
        models = {
            "leader": entry(categories={"knowledge": 0.85}),
            "resident": entry(categories={"knowledge": 0.70}),
        }
        c = choose(
            feats(),
            models,
            {"resident"},
            escalate_at=4,
            residency_epsilon=0.02,
        )
        assert c.target == "leader"

    def test_generalist_fallback_when_axis_empty(self):
        models = {"m": entry(categories={"knowledge": 0.8})}
        c = choose(
            feats(code=True), models, set(), escalate_at=4, default_target="gen"
        )
        assert c.target == "gen"
        assert c.rule == "table:generalist"

    def test_no_candidates_no_generalist_returns_none(self):
        c = choose(feats(), {}, set(), escalate_at=4)
        assert c.target is None
        assert c.rule == "table:no_candidates"

    def test_none_features_dispatches_knowledge(self):
        models = {"m": entry(categories={"knowledge": 0.8})}
        c = choose(None, models, set(), escalate_at=4)
        assert c.target == "m"
        assert c.rule == "table:knowledge"

    def test_candidates_capped_at_five(self):
        models = {
            f"m{i}": entry(categories={"knowledge": 0.5 + i / 100}) for i in range(8)
        }
        c = choose(feats(), models, set(), escalate_at=4)
        assert len(c.candidates) == 5

    def test_choice_dataclass_defaults(self):
        c = TableChoice(None, "x")
        assert c.candidates == []
