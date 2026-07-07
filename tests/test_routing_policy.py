# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.routing.policy.decide: pure, exhaustive over the
decision order (override -> complexity/math/code thresholds -> route_token
fallback -> fail-open)."""

import pytest

from omlx.routing.policy import decide
from omlx.routing.profiler import RouterFeatures
from omlx.settings import RoutingAgenticOverrideSettings, RoutingPolicySettings


def make_cfg(**overrides) -> RoutingPolicySettings:
    return RoutingPolicySettings(
        escalate_complexity_at=overrides.get("escalate_complexity_at", 4),
        escalate_math_complexity_at=overrides.get("escalate_math_complexity_at", 3),
        escalate_code_complexity_at=overrides.get("escalate_code_complexity_at", 3),
        agentic_override=overrides.get(
            "agentic_override", RoutingAgenticOverrideSettings()
        ),
        fail_open_target=overrides.get("fail_open_target", "big"),
    )


def make_features(
    domain="General",
    complexity=None,
    math=False,
    code=False,
    route_token=None,
) -> RouterFeatures:
    return RouterFeatures(
        domain=domain,
        complexity=complexity,
        math=math,
        code=code,
        route_token=route_token,
    )


class TestOverrides:
    def test_tools_override_wins_regardless_of_features(self):
        cfg = make_cfg()
        target, rule = decide(make_features(complexity=1), "tools", cfg)
        assert (target, rule) == ("big", "override:tools")

    def test_tools_override_wins_with_no_features(self):
        cfg = make_cfg()
        target, rule = decide(None, "tools", cfg)
        assert (target, rule) == ("big", "override:tools")

    def test_turns_override_wins_regardless_of_features(self):
        cfg = make_cfg()
        target, rule = decide(make_features(complexity=1), "turns", cfg)
        assert (target, rule) == ("big", "override:turns")

    def test_no_override_falls_through_to_features(self):
        cfg = make_cfg()
        target, rule = decide(make_features(complexity=1), None, cfg)
        assert target == "small"


class TestComplexityThresholds:
    @pytest.mark.parametrize("complexity", [4, 5, 10])
    def test_complexity_at_or_above_threshold_escalates(self, complexity):
        cfg = make_cfg(escalate_complexity_at=4)
        target, rule = decide(make_features(complexity=complexity), None, cfg)
        assert target == "big"
        assert rule == "complexity>=4"

    @pytest.mark.parametrize("complexity", [0, 1, 2, 3])
    def test_complexity_below_threshold_does_not_escalate(self, complexity):
        cfg = make_cfg(escalate_complexity_at=4)
        target, rule = decide(make_features(complexity=complexity), None, cfg)
        assert target == "small"
        assert rule == "below_thresholds"

    def test_complexity_threshold_is_configurable(self):
        cfg = make_cfg(escalate_complexity_at=2)
        target, rule = decide(make_features(complexity=2), None, cfg)
        assert (target, rule) == ("big", "complexity>=2")


class TestMathEscalation:
    def test_math_at_threshold_escalates(self):
        cfg = make_cfg(escalate_complexity_at=4, escalate_math_complexity_at=3)
        target, rule = decide(make_features(complexity=3, math=True), None, cfg)
        assert (target, rule) == ("big", "math_complexity>=3")

    def test_math_below_threshold_does_not_escalate(self):
        cfg = make_cfg(escalate_complexity_at=4, escalate_math_complexity_at=3)
        target, rule = decide(make_features(complexity=2, math=True), None, cfg)
        assert target == "small"

    def test_math_false_does_not_trigger_math_rule(self):
        cfg = make_cfg(escalate_complexity_at=4, escalate_math_complexity_at=3)
        target, rule = decide(make_features(complexity=3, math=False), None, cfg)
        assert target == "small"


class TestCodeEscalation:
    def test_code_at_threshold_escalates(self):
        cfg = make_cfg(escalate_complexity_at=4, escalate_code_complexity_at=3)
        target, rule = decide(make_features(complexity=3, code=True), None, cfg)
        assert (target, rule) == ("big", "code_complexity>=3")

    def test_code_below_threshold_does_not_escalate(self):
        cfg = make_cfg(escalate_complexity_at=4, escalate_code_complexity_at=3)
        target, rule = decide(make_features(complexity=2, code=True), None, cfg)
        assert target == "small"

    def test_complexity_rule_checked_before_math_and_code(self):
        # complexity alone already clears the top threshold; rule_fired
        # should report the complexity rule, not math/code.
        cfg = make_cfg(
            escalate_complexity_at=3,
            escalate_math_complexity_at=3,
            escalate_code_complexity_at=3,
        )
        target, rule = decide(
            make_features(complexity=3, math=True, code=True), None, cfg
        )
        assert (target, rule) == ("big", "complexity>=3")


class TestRouteTokenFallback:
    def test_falls_back_to_route_token_small_when_complexity_missing(self):
        cfg = make_cfg()
        target, rule = decide(
            make_features(complexity=None, route_token="small"), None, cfg
        )
        assert (target, rule) == ("small", "fallback:route_token")

    def test_falls_back_to_route_token_big_when_complexity_missing(self):
        cfg = make_cfg()
        target, rule = decide(
            make_features(complexity=None, route_token="big"), None, cfg
        )
        assert (target, rule) == ("big", "fallback:route_token")

    def test_unknown_route_token_fails_open(self):
        cfg = make_cfg(fail_open_target="big")
        target, rule = decide(
            make_features(complexity=None, route_token="medium"), None, cfg
        )
        assert (target, rule) == ("big", "fail_open:unparseable")

    def test_no_route_token_fails_open(self):
        cfg = make_cfg(fail_open_target="small")
        target, rule = decide(
            make_features(complexity=None, route_token=None), None, cfg
        )
        assert (target, rule) == ("small", "fail_open:unparseable")


class TestFailOpen:
    def test_no_features_fails_open(self):
        cfg = make_cfg(fail_open_target="big")
        target, rule = decide(None, None, cfg)
        assert (target, rule) == ("big", "fail_open:no_features")

    def test_fail_open_target_is_configurable(self):
        cfg = make_cfg(fail_open_target="small")
        target, rule = decide(None, None, cfg)
        assert target == "small"
