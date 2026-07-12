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
   meaningfully better cold model still wins). When nothing in the tie
   group is resident, the cheapest measured load wins — near-tied scores
   never justify paying a 20 s cold load over a 5 s one.

Agentic-override dispatch (choose_override): tool-bearing requests skip
the profiler, but instead of collapsing onto one configured generalist
they rank the same eligible pool on the measured "agentic" axis, with the
identical residency/load tiebreak. The configured default_target remains
the fallback when no agentic scores exist.
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


def _load_cost_s(entry: dict) -> float | None:
    """Measured load time from the most recent baseline eval record."""
    best_date = ""
    load_s: float | None = None
    for rec in entry.get("evals", []):
        if not rec.get("baseline"):
            continue
        val = rec.get("load_s")
        if not isinstance(val, (int, float)):
            continue
        date = rec.get("date") or ""
        if date >= best_date:
            best_date = date
            load_s = float(val)
    return load_s


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
    elig = _Eligibility(
        models,
        max_interactive_median_q_time_s=max_interactive_median_q_time_s,
        fit_budget_gb=fit_budget_gb,
        enabled_ids=enabled_ids,
    )

    # Escalation tier: complexity band beats axis specialization.
    complexity = getattr(features, "complexity", None) if features else None
    if complexity is not None and complexity >= escalate_at:
        scored = [
            (mid, s)
            for mid, entry in models.items()
            if elig.eligible(mid, entry) and (s := _overall_score(entry)) is not None
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        if scored:
            target = _pick(scored, resident_ids, residency_epsilon, models)
            return TableChoice(
                target,
                f"table:escalate>={escalate_at}",
                scored[:5],
                sorted(elig.unfit),
                sorted(elig.disabled),
            )
        # fall through to axis dispatch if no overall scores exist

    axis = axis_for(features)
    ranked = [
        (mid, score)
        for mid, entry in models.items()
        if elig.eligible(mid, entry)
        and (score := (entry.get("categories") or {}).get(axis)) is not None
    ]
    ranked.sort(key=lambda t: t[1], reverse=True)

    if ranked:
        target = _pick(ranked, resident_ids, residency_epsilon, models)
        return TableChoice(
            target,
            f"table:{axis}",
            ranked[:5],
            sorted(elig.unfit),
            sorted(elig.disabled),
        )

    if default_target:
        return TableChoice(
            default_target,
            "table:generalist",
            [],
            sorted(elig.unfit),
            sorted(elig.disabled),
        )

    return TableChoice(
        None, "table:no_candidates", [], sorted(elig.unfit), sorted(elig.disabled)
    )


def choose_override(
    models: dict[str, dict],
    resident_ids: set[str],
    *,
    residency_epsilon: float = 0.02,
    max_interactive_median_q_time_s: float = 30.0,
    fit_budget_gb: float | None = None,
    enabled_ids: set[str] | None = None,
) -> TableChoice:
    """Dispatch an agentic-override request on the measured agentic axis.

    Same eligibility gates and residency/load tiebreak as choose(); ranks
    on categories["agentic"] (the toolcall bench). No agentic-scored
    candidates -> TableChoice(None, ...) and the caller falls back to the
    configured generalist exactly as before this axis existed. Never raises.
    """
    elig = _Eligibility(
        models,
        max_interactive_median_q_time_s=max_interactive_median_q_time_s,
        fit_budget_gb=fit_budget_gb,
        enabled_ids=enabled_ids,
    )
    ranked = [
        (mid, score)
        for mid, entry in models.items()
        if elig.eligible(mid, entry)
        and (score := (entry.get("categories") or {}).get("agentic")) is not None
    ]
    ranked.sort(key=lambda t: t[1], reverse=True)

    if ranked:
        target = _pick(ranked, resident_ids, residency_epsilon, models)
        return TableChoice(
            target,
            "table:agentic",
            ranked[:5],
            sorted(elig.unfit),
            sorted(elig.disabled),
        )
    return TableChoice(
        None, "table:no_candidates", [], sorted(elig.unfit), sorted(elig.disabled)
    )


class _Eligibility:
    """Shared candidate gates; records why models were excluded."""

    def __init__(
        self,
        models: dict[str, dict],
        *,
        max_interactive_median_q_time_s: float,
        fit_budget_gb: float | None,
        enabled_ids: set[str] | None,
    ) -> None:
        self._models = models
        self._max_latency = max_interactive_median_q_time_s
        self._fit_budget_gb = fit_budget_gb
        self._enabled_ids = enabled_ids
        # empty/None -> gate inert (opt-in not configured)
        self._gate = bool(enabled_ids)
        self.unfit: set[str] = set()
        self.disabled: set[str] = set()

    def eligible(self, model_id: str, entry: dict) -> bool:
        return (
            entry.get("role") == "chat"
            and entry.get("health", {}).get("status") == "ok"
            and self._fits(model_id, entry)
            and self._interactive(model_id)
            and self._enabled(model_id)
        )

    def _interactive(self, model_id: str) -> bool:
        lat = _median_latency_s(self._models.get(model_id, {}))
        return lat is None or lat <= self._max_latency

    def _fits(self, model_id: str, entry: dict) -> bool:
        if self._fit_budget_gb is None:
            return True
        size_gb = entry.get("size_gb")
        if size_gb is None or size_gb <= self._fit_budget_gb:
            return True
        self.unfit.add(model_id)
        return False

    def _enabled(self, model_id: str) -> bool:
        if not self._gate or model_id in self._enabled_ids:  # type: ignore[operator]
            return True
        self.disabled.add(model_id)
        return False


def _pick(
    ranked: list[tuple[str, float]],
    resident_ids: set[str],
    epsilon: float,
    models: dict[str, dict],
) -> str:
    """Residency-then-load tiebreak within epsilon of the leader.

    Best score wins unless a resident model is within epsilon of it (loads
    cost seconds). When nothing in the tie group is resident, the cheapest
    measured load wins: a near-tied score never justifies a 20 s cold load
    over a 5 s one. If no tie-group member has load data, the leader wins.
    """
    leader_id, leader_score = ranked[0]
    if leader_id in resident_ids:
        return leader_id
    tie_group = [leader_id]
    for model_id, score in ranked[1:]:
        if leader_score - score > epsilon:
            break
        if model_id in resident_ids:
            return model_id
        tie_group.append(model_id)
    if len(tie_group) > 1:
        loads = [
            (load, mid)
            for mid in tie_group
            if (load := _load_cost_s(models.get(mid, {}))) is not None
        ]
        if loads:
            return min(loads)[1]
    return leader_id
