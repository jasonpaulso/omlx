# SPDX-License-Identifier: Apache-2.0
"""N-way dispatch: features x suitability store x pool state -> target model.

Pure decision logic. The caller (RoutingService) supplies the store snapshot
and pool state; this module never does I/O. Every choice degrades gracefully:
no table data for the relevant axis -> the configured generalist -> the
caller's binary fallback.

Selection order:
1. Axis from features: code flag -> "code", math flag -> "math",
   planning_agentic domain -> "agentic" (capability profiler only),
   else "knowledge" (the general-prompt axis; mmlu-backed).
2. Candidates: healthy chat models ranked on that axis, minus models whose
   measured median answer latency exceeds the interactive budget (the
   "thinking lane" exclusion -- accuracy ties with 10x latency are not ties).
3. Escalation tier: at complexity >= escalate_at, prefer the overall-best
   model (mean across axes) -- the roster's frontier is dispatched by
   complexity band, not by axis specialization.
4. Residency tiebreak: among candidates within epsilon of the leader's
   score, prefer one that is already resident (loads cost seconds; a
   meaningfully better cold model still wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TableChoice:
    """Outcome of a table dispatch attempt."""

    target: str | None  # None -> caller falls back to binary policy
    rule: str
    candidates: list[tuple[str, float]] = field(default_factory=list)
    unfit: list[str] = field(default_factory=list)  # excluded: can't ever fit
    disabled: list[str] = field(default_factory=list)  # excluded: not opted in budget


def _median_latency_s(entry: dict) -> float | None:
    """Median per-question latency across latest baseline eval records."""
    latest: dict[str, float] = {}
    for rec in entry.get("evals", []):
        if not rec.get("baseline"):
            continue
        val = rec.get("median_q_time_s")
        if isinstance(val, (int, float)):
            latest[rec.get("bench", "?")] = float(val)
    if not latest:
        return None
    values = sorted(latest.values())
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _overall_score(entry: dict) -> float | None:
    cats = entry.get("categories") or {}
    if not cats:
        return None
    return sum(cats.values()) / len(cats)


def axis_for(features) -> str:
    """Map router features to a suitability axis. Code beats math on ties
    (a prompt flagged both is usually a programming task with math in it);
    both beat agentic (a coding-agent prompt scoring high on both stays
    code). Only the capability profiler emits a "planning_agentic" domain,
    so generative (Supra) installs never take the agentic branch."""
    if features is not None and getattr(features, "code", False):
        return "code"
    if features is not None and getattr(features, "math", False):
        return "math"
    if features is not None and getattr(features, "domain", None) == "planning_agentic":
        return "agentic"
    return "knowledge"


def choose(
    features,
    models: dict[str, dict],
    resident_ids: set[str],
    *,
    escalate_at: int,
    residency_epsilon: float = 0.02,
    max_interactive_median_q_time_s: float = 30.0,
    default_target: str | None = None,
    fit_budget_gb: float | None = None,
    enabled_ids: set[str] | None = None,
) -> TableChoice:
    """Pick a target from the suitability table. Never raises.

    `models` is SuitabilityStore.all_models() (or a snapshot); `resident_ids`
    are currently-loaded model ids from the pool. `fit_budget_gb`, when
    given, is the never-fits ceiling (headroom above pinned/resident
    memory) computed by the caller; a candidate whose `size_gb` exceeds it
    can never load and is excluded (recorded in `TableChoice.unfit`).
    Missing size_gb or fit_budget_gb means no filtering (fail-open).

    `enabled_ids`, when non-empty, is the set of models the operator has
    opted in as routing targets (per-model `enable_routing`); a ranked
    candidate outside it is excluded (recorded in `TableChoice.disabled`).
    An empty or None set means no gating -- if nobody has opted in, the
    gate is inert and dispatch behaves as before (fail-open). The gate
    only filters the ranked pool; the caller's `default_target` and
    fail-open path are explicitly-named and bypass it.
    """
    unfit_ids: set[str] = set()
    disabled_ids: set[str] = set()
    gate = bool(enabled_ids)  # empty/None -> gate inert (opt-in not configured)

    def interactive(model_id: str) -> bool:
        lat = _median_latency_s(models.get(model_id, {}))
        return lat is None or lat <= max_interactive_median_q_time_s

    def fits(model_id: str, entry: dict) -> bool:
        if fit_budget_gb is None:
            return True
        size_gb = entry.get("size_gb")
        if size_gb is None or size_gb <= fit_budget_gb:
            return True
        unfit_ids.add(model_id)
        return False

    def enabled(model_id: str) -> bool:
        if not gate or model_id in enabled_ids:  # type: ignore[operator]
            return True
        disabled_ids.add(model_id)
        return False

    def eligible(model_id: str, entry: dict) -> bool:
        return (
            entry.get("role") == "chat"
            and entry.get("health", {}).get("status") == "ok"
            and fits(model_id, entry)
            and interactive(model_id)
            and enabled(model_id)
        )

    # Escalation tier: complexity band beats axis specialization.
    complexity = getattr(features, "complexity", None) if features else None
    if complexity is not None and complexity >= escalate_at:
        scored = [
            (mid, s)
            for mid, entry in models.items()
            if eligible(mid, entry) and (s := _overall_score(entry)) is not None
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        if scored:
            target = _residency_pick(scored, resident_ids, residency_epsilon)
            return TableChoice(
                target,
                f"table:escalate>={escalate_at}",
                scored[:5],
                sorted(unfit_ids),
                sorted(disabled_ids),
            )
        # fall through to axis dispatch if no overall scores exist

    axis = axis_for(features)
    ranked = [
        (mid, score)
        for mid, entry in models.items()
        if eligible(mid, entry)
        and (score := (entry.get("categories") or {}).get(axis)) is not None
    ]
    ranked.sort(key=lambda t: t[1], reverse=True)

    if ranked:
        target = _residency_pick(ranked, resident_ids, residency_epsilon)
        return TableChoice(
            target, f"table:{axis}", ranked[:5], sorted(unfit_ids), sorted(disabled_ids)
        )

    if default_target:
        return TableChoice(
            default_target,
            "table:generalist",
            [],
            sorted(unfit_ids),
            sorted(disabled_ids),
        )

    return TableChoice(
        None, "table:no_candidates", [], sorted(unfit_ids), sorted(disabled_ids)
    )


def _residency_pick(
    ranked: list[tuple[str, float]],
    resident_ids: set[str],
    epsilon: float,
) -> str:
    """Best score wins unless a resident model is within epsilon of it."""
    leader_id, leader_score = ranked[0]
    if leader_id in resident_ids:
        return leader_id
    for model_id, score in ranked[1:]:
        if leader_score - score > epsilon:
            break
        if model_id in resident_ids:
            return model_id
    return leader_id
