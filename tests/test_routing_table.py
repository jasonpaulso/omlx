# SPDX-License-Identifier: Apache-2.0
"""Tests for N-way table dispatch (omlx/routing/table.py)."""

from omlx.routing.profiler import RouterFeatures
from omlx.routing.table import TableChoice, axis_for, choose, choose_override


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
        c = choose(feats(code=True), models, set(), escalate_at=4, default_target="gen")
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
        assert c.disabled == []


def afeats(domain="planning_agentic", math=False, code=False, complexity=2):
    return RouterFeatures(
        domain=domain, complexity=complexity, math=math, code=code, route_token=None
    )


class TestAgenticDispatch:
    def test_axis_for_agentic_domain(self):
        assert axis_for(afeats()) == "agentic"

    def test_axis_for_code_beats_agentic(self):
        assert axis_for(afeats(code=True)) == "code"

    def test_axis_for_math_beats_agentic(self):
        assert axis_for(afeats(math=True)) == "math"

    def test_axis_for_non_agentic_domain_is_knowledge(self):
        assert axis_for(afeats(domain="creative_writing")) == "knowledge"
        assert axis_for(afeats(domain=None)) == "knowledge"

    def test_agentic_axis_leader_wins(self):
        models = {
            "agent": entry(categories={"agentic": 0.9, "knowledge": 0.5}),
            "general": entry(categories={"agentic": 0.6, "knowledge": 0.8}),
        }
        c = choose(afeats(), models, set(), escalate_at=4)
        assert c.target == "agent"
        assert c.rule == "table:agentic"
        assert ("agent", 0.9) in c.candidates

    def test_no_agentic_scores_falls_to_generalist(self):
        models = {"m": entry(categories={"knowledge": 0.8})}
        c = choose(afeats(), models, set(), escalate_at=4, default_target="gen")
        assert c.target == "gen"
        assert c.rule == "table:generalist"


class TestEnableRoutingGate:
    """Per-model enable_routing gate on the ranked candidate pool."""

    def _models(self):
        return {
            "coder": entry(categories={"code": 0.9, "knowledge": 0.5}),
            "general": entry(categories={"code": 0.6, "knowledge": 0.8}),
        }

    def test_empty_enabled_is_noop(self):
        # Nobody opted in -> gate inert -> best model wins as before.
        for enabled in (set(), None):
            c = choose(
                feats(code=True),
                self._models(),
                set(),
                escalate_at=4,
                enabled_ids=enabled,
            )
            assert c.target == "coder"
            assert c.disabled == []

    def test_disabled_leader_skipped_next_enabled_wins(self):
        c = choose(
            feats(code=True),
            self._models(),
            set(),
            escalate_at=4,
            enabled_ids={"general"},
        )
        assert c.target == "general"
        assert c.rule == "table:code"
        assert c.disabled == ["coder"]

    def test_all_disabled_falls_to_default_target_which_bypasses_gate(self):
        # No ranked candidate is enabled -> named default_target wins even
        # though it is not itself in enabled_ids (explicit config = opt-in).
        c = choose(
            feats(code=True),
            self._models(),
            set(),
            escalate_at=4,
            enabled_ids={"nonexistent"},
            default_target="gen-spine",
        )
        assert c.target == "gen-spine"
        assert c.rule == "table:generalist"
        assert sorted(c.disabled) == ["coder", "general"]

    def test_escalation_tier_respects_gate(self):
        models = {
            "big": entry(categories={"code": 0.9, "knowledge": 0.9}),
            "mid": entry(categories={"code": 0.7, "knowledge": 0.7}),
        }
        c = choose(
            feats(complexity=5),
            models,
            set(),
            escalate_at=4,
            enabled_ids={"mid"},
        )
        assert c.target == "mid"
        assert c.rule == "table:escalate>=4"
        assert c.disabled == ["big"]

    def test_enabled_leader_still_wins(self):
        c = choose(
            feats(code=True),
            self._models(),
            set(),
            escalate_at=4,
            enabled_ids={"coder", "general"},
        )
        assert c.target == "coder"
        assert c.disabled == []


def entry_with_load(agentic=None, load_s=None, categories=None, **kw):
    e = entry(categories=categories or ({"agentic": agentic} if agentic else {}), **kw)
    if load_s is not None:
        e["evals"].append(
            {
                "bench": "toolcall",
                "baseline": True,
                "load_s": load_s,
                "date": "2026-07-10T00:00:00+00:00",
            }
        )
    return e


class TestChooseOverride:
    def test_agentic_leader_wins(self):
        models = {
            "toolmaster": entry_with_load(agentic=0.91),
            "generalist": entry_with_load(agentic=0.75),
        }
        c = choose_override(models, set())
        assert c.target == "toolmaster"
        assert c.rule == "table:agentic"
        assert c.candidates[0] == ("toolmaster", 0.91)

    def test_no_agentic_scores_returns_none_target(self):
        models = {"m": entry(categories={"code": 0.9})}
        c = choose_override(models, set())
        assert c.target is None
        assert c.rule == "table:no_candidates"

    def test_gate_and_health_respected(self):
        models = {
            "leader": entry_with_load(agentic=0.95),
            "runner": entry_with_load(agentic=0.9),
            "sick": entry_with_load(agentic=0.99, healthy=False),
        }
        c = choose_override(models, set(), enabled_ids={"runner"})
        assert c.target == "runner"
        assert "leader" in c.disabled

    def test_resident_within_epsilon_beats_cold_leader(self):
        models = {
            "cold-leader": entry_with_load(agentic=0.913),
            "warm-second": entry_with_load(agentic=0.907),
        }
        c = choose_override(models, {"warm-second"})
        assert c.target == "warm-second"


class TestLatencyTiebreak:
    """Cold tie group -> fastest model wins: med-q first, load second."""

    def test_cold_tie_prefers_cheapest_load(self):
        # 122B-class edge: 0.003 score lead never justifies a 22s load.
        models = {
            "huge": entry_with_load(agentic=0.9133, load_s=22.5),
            "fast": entry_with_load(agentic=0.9100, load_s=6.0),
        }
        c = choose_override(models, set())
        assert c.target == "fast"

    def test_cold_tie_medq_beats_load(self):
        # Run-3 edge: cheapest load was the slowest decoder; per-turn
        # latency recurs every request, load is paid once.
        models = {
            "slow-coder": entry_with_load(
                agentic=0.897, load_s=4.8, median_q_time_s=3.4
            ),
            "quick": entry_with_load(agentic=0.907, load_s=6.1, median_q_time_s=0.7),
        }
        c = choose_override(models, set())
        assert c.target == "quick"

    def test_medq_ties_fall_to_load(self):
        models = {
            "a": entry_with_load(agentic=0.91, load_s=22.5, median_q_time_s=0.7),
            "b": entry_with_load(agentic=0.90, load_s=6.0, median_q_time_s=0.7),
        }
        c = choose_override(models, set())
        assert c.target == "b"

    def test_outside_epsilon_leader_wins_despite_load(self):
        models = {
            "huge": entry_with_load(agentic=0.95, load_s=22.5),
            "fast": entry_with_load(agentic=0.90, load_s=1.0),
        }
        c = choose_override(models, set())
        assert c.target == "huge"

    def test_no_latency_data_keeps_leader(self):
        models = {
            "a": entry_with_load(agentic=0.91),
            "b": entry_with_load(agentic=0.90),
        }
        c = choose_override(models, set())
        assert c.target == "a"

    def test_resident_leader_short_circuits(self):
        models = {
            "huge": entry_with_load(agentic=0.913, load_s=22.5),
            "fast": entry_with_load(agentic=0.910, load_s=6.0),
        }
        c = choose_override(models, {"huge"})
        assert c.target == "huge"

    def test_axis_dispatch_also_latency_aware(self):
        # The guard lives in the shared pick, not just the override path.
        models = {
            "huge": entry_with_load(categories={"code": 0.91}, load_s=22.5),
            "fast": entry_with_load(categories={"code": 0.90}, load_s=6.0),
        }
        c = choose(feats(code=True), models, set(), escalate_at=4)
        assert c.target == "fast"
        assert c.rule == "table:code"
