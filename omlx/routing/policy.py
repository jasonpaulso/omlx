# SPDX-License-Identifier: Apache-2.0
"""
Deterministic routing policy: maps router features (plus an agentic
override) to a target model key ("small"/"big"). Pure function, no I/O.
"""

from __future__ import annotations

from omlx.routing.profiler import RouterFeatures
from omlx.settings import RoutingPolicySettings


def decide(
    features: RouterFeatures | None,
    override: str | None,
    cfg: RoutingPolicySettings,
) -> tuple[str, str]:
    """Decide the target model key and the rule that fired.

    Returns (target_key, rule_fired). target_key is "small"/"big" (or
    whatever cfg.fail_open_target is set to); the caller maps it to a
    concrete model id via settings.targets.
    """
    if override == "tools":
        return "big", "override:tools"
    if override == "turns":
        return "big", "override:turns"

    if features is None:
        return cfg.fail_open_target, "fail_open:no_features"

    if features.complexity is not None:
        c = features.complexity
        if c >= cfg.escalate_complexity_at:
            return "big", f"complexity>={cfg.escalate_complexity_at}"
        if features.math and c >= cfg.escalate_math_complexity_at:
            return "big", f"math_complexity>={cfg.escalate_math_complexity_at}"
        if features.code and c >= cfg.escalate_code_complexity_at:
            return "big", f"code_complexity>={cfg.escalate_code_complexity_at}"
        return "small", "below_thresholds"

    if features.route_token in ("small", "big"):
        return features.route_token, "fallback:route_token"

    return cfg.fail_open_target, "fail_open:unparseable"
