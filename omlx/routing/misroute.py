# SPDX-License-Identifier: Apache-2.0
"""M6.2 — misroute measurement (read-only).

A misroute is a decision whose routed target demonstrably hurt the user:
a worse answer than a roster sibling would have given (under-route), or the
same answer at needless latency (over-route). No single telemetry field
proves that, so `misroute_report` triangulates three planes, each reported
under its own evidentiary weight and never blended into one opaque score:

1. direct  -- joined feedback: negative rate per rule_fired / target.
2. proxy   -- shadow-label vs profiler-complexity disagreement.
3. cost    -- latency paid on shadow-TRIVIAL rows vs the corpus.

Pure aggregation over already-written telemetry rows; nothing here changes
routing behavior or mutates the corpus. See docs/ROUTING.md, "M6.2 --
misroute measurement (read-only)" for the full spec this module implements,
including the pre-registered M6.3 gate encoded in the `gate` field below.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any

from omlx.routing.service import join_feedback

_MAX_ROWS = 100_000

# Shadow label -> expected profiler complexity (1-5), for the proxy plane.
_EXPECTED_COMPLEXITY = {
    "TRIVIAL": 1,
    "SIMPLE": 2,
    "MODERATE": 3,
    "COMPLEX": 5,
}
_DISAGREEMENT_THRESHOLD = 2

# M6.3 gate thresholds (docs/ROUTING.md, pre-registered before the first
# full-corpus run -- do not tune these after seeing the numbers).
_GATE_MIN_JOINED = 50
_GATE_A_RATE_MULTIPLIER = 2.0
_GATE_A_MIN_NEGATIVE = 10
_GATE_B_UNDER_RATE = 0.15


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load telemetry rows from a routing_decisions.jsonl file.

    Skips unparseable lines. Caps at `_MAX_ROWS`, keeping the tail-most rows
    (most recent) when the file is larger, in original (oldest-first) order.
    Missing file or any I/O problem yields an empty list -- never raises.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines: deque[str] = deque(f, maxlen=_MAX_ROWS)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _min_feedback_score(decision: dict[str, Any]) -> float | None:
    fb = decision.get("feedback")
    if not fb:
        return None
    scores: list[float] = []
    for f in fb:
        score = f.get("score") if isinstance(f, dict) else None
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return min(scores)


def _is_negative(decision: dict[str, Any]) -> bool:
    # min() can legitimately be 0.0 (tool_error/negation), so test against
    # None explicitly -- `score or 1.0` would flip the strongest negatives.
    score = _min_feedback_score(decision)
    return score is not None and score < 0.5


def _segment_stats(joined: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Per-value negative/joined/rate breakdown for `key` (rule_fired/target/endpoint)."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for d in joined:
        value = d.get(key)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(d)
    out: dict[str, dict[str, Any]] = {}
    for value, rows in buckets.items():
        negative = sum(1 for r in rows if _is_negative(r))
        out[value] = {
            "joined": len(rows),
            "negative": negative,
            "rate": _rate(negative, len(rows)),
        }
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _compute_gate(
    *,
    joined_n: int,
    negative_rate: float | None,
    by_rule: dict[str, dict[str, Any]],
    by_target: dict[str, dict[str, Any]],
    under_rate: float | None,
    under_negative_rate: float | None,
) -> dict[str, Any]:
    eligible = joined_n >= _GATE_MIN_JOINED
    if not eligible:
        return {
            "joined_n": joined_n,
            "eligible": False,
            "criterion_a": None,
            "criterion_b": None,
            "met": False,
        }

    baseline = negative_rate or 0.0
    criterion_a = False
    for segment in (*by_rule.values(), *by_target.values()):
        rate = segment.get("rate")
        if rate is None:
            continue
        if (
            segment["negative"] >= _GATE_A_MIN_NEGATIVE
            and rate >= baseline * _GATE_A_RATE_MULTIPLIER
        ):
            criterion_a = True
            break

    # Criterion b: under-route disagreement covers >= 15% of shadowed rows
    # AND those rows' own joined feedback corroborates (their negative rate
    # sits above the corpus baseline).
    criterion_b = bool(
        under_rate is not None
        and under_rate >= _GATE_B_UNDER_RATE
        and under_negative_rate is not None
        and under_negative_rate > baseline
    )

    return {
        "joined_n": joined_n,
        "eligible": True,
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "met": criterion_a or criterion_b,
    }


def misroute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a routing telemetry corpus into the M6.2 misroute report.

    Pure function; tolerant of any corpus vintage (missing shadow/outcome/
    features/sticky fields). Never raises on well-formed JSON row dicts.
    """
    decisions = [r for r in rows if r.get("kind") != "feedback"]
    feedback_rows = [r for r in rows if r.get("kind") == "feedback"]
    implicit_feedback = sum(1 for r in feedback_rows if r.get("source") == "implicit")
    shadowed = [d for d in decisions if d.get("shadow")]
    with_outcome = [d for d in decisions if d.get("outcome")]

    joined = join_feedback(list(rows))
    joined_with_feedback = [d for d in joined if d.get("feedback")]

    # --- direct plane ---
    negative_n = sum(1 for d in joined_with_feedback if _is_negative(d))
    joined_n = len(joined_with_feedback)
    direct: dict[str, Any] = {
        "joined_n": joined_n,
        "negative_n": negative_n,
        "negative_rate": _rate(negative_n, joined_n),
        "by_rule": _segment_stats(joined_with_feedback, "rule_fired"),
        "by_target": _segment_stats(joined_with_feedback, "target"),
    }

    # --- proxy plane ---
    matrix: dict[str, dict[int, int]] = {}
    gaps: list[int] = []
    over_n = 0
    under_n = 0
    disagree_n = 0
    under_request_ids: set[Any] = set()
    for d in shadowed:
        shadow = d.get("shadow") or {}
        label = shadow.get("label")
        features = d.get("features") or {}
        complexity = features.get("complexity")
        if label not in _EXPECTED_COMPLEXITY or complexity is None:
            continue
        matrix.setdefault(label, {})
        matrix[label][complexity] = matrix[label].get(complexity, 0) + 1
        gap = complexity - _EXPECTED_COMPLEXITY[label]
        gaps.append(gap)
        if abs(gap) >= _DISAGREEMENT_THRESHOLD:
            disagree_n += 1
            if gap > 0:
                over_n += 1
            else:
                under_n += 1
                under_request_ids.add(d.get("request_id"))
    shadowed_n = len(shadowed)
    proxy: dict[str, Any] = {
        "shadowed_n": shadowed_n,
        "matrix": matrix,
        "mean_signed_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "disagree_n": disagree_n,
        "disagree_rate": _rate(disagree_n, shadowed_n),
        "over_n": over_n,
        "over_rate": _rate(over_n, shadowed_n),
        "under_n": under_n,
        "under_rate": _rate(under_n, shadowed_n),
    }

    # --- cost plane ---
    trivial_rows = [
        d for d in shadowed if (d.get("shadow") or {}).get("label") == "TRIVIAL"
    ]
    trivial_ttft = [
        d["outcome"]["ttft_ms"]
        for d in trivial_rows
        if d.get("outcome") and d["outcome"].get("ttft_ms") is not None
    ]
    trivial_gen = [
        d["outcome"]["gen_ms"]
        for d in trivial_rows
        if d.get("outcome") and d["outcome"].get("gen_ms") is not None
    ]
    corpus_ttft = [
        d["outcome"]["ttft_ms"]
        for d in with_outcome
        if d["outcome"].get("ttft_ms") is not None
    ]
    corpus_gen = [
        d["outcome"]["gen_ms"]
        for d in with_outcome
        if d["outcome"].get("gen_ms") is not None
    ]
    cost = {
        "trivial_n": len(trivial_rows),
        "trivial_median_ttft_ms": _median(trivial_ttft),
        "trivial_median_gen_ms": _median(trivial_gen),
        "corpus_median_ttft_ms": _median(corpus_ttft),
        "corpus_median_gen_ms": _median(corpus_gen),
    }

    # --- segments ---
    by_endpoint = _segment_stats(joined_with_feedback, "endpoint")
    has_sticky = any(isinstance(d.get("sticky"), dict) for d in decisions)
    sticky_held: dict[str, dict[str, Any]] | None = None
    if has_sticky:
        held_bucket: dict[str, list[dict[str, Any]]] = {"held": [], "fresh": []}
        for d in joined_with_feedback:
            sticky = d.get("sticky")
            if not isinstance(sticky, dict):
                continue
            key = "held" if sticky.get("held") else "fresh"
            held_bucket[key].append(d)
        sticky_held = {}
        for key, rows_ in held_bucket.items():
            negative = sum(1 for r in rows_ if _is_negative(r))
            sticky_held[key] = {
                "joined": len(rows_),
                "negative": negative,
                "rate": _rate(negative, len(rows_)),
            }

    under_joined = [
        d for d in joined_with_feedback if d.get("request_id") in under_request_ids
    ]
    under_negative = sum(1 for d in under_joined if _is_negative(d))
    under_negative_rate = _rate(under_negative, len(under_joined))

    gate = _compute_gate(
        joined_n=joined_n,
        negative_rate=direct["negative_rate"],
        by_rule=direct["by_rule"],
        by_target=direct["by_target"],
        under_rate=proxy["under_rate"],
        under_negative_rate=under_negative_rate,
    )

    return {
        "coverage": {
            "decisions": len(decisions),
            "shadowed": shadowed_n,
            "with_outcome": len(with_outcome),
            "feedback_rows": len(feedback_rows),
            "implicit_feedback": implicit_feedback,
            "joined_decisions": joined_n,
        },
        "direct": direct,
        "proxy": proxy,
        "cost": cost,
        "segments": {
            "by_endpoint": by_endpoint,
            "sticky_held": sticky_held,
        },
        "gate": gate,
    }


def _default_telemetry_path() -> Path:
    try:
        from omlx.settings import GlobalSettings

        return Path(GlobalSettings.load().routing.telemetry.path).expanduser()
    except Exception:  # noqa: BLE001 - CLI convenience only, never fatal
        return Path("~/.omlx/routing_decisions.jsonl").expanduser()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]).expanduser() if argv else _default_telemetry_path()
    rows = load_rows(path)
    report = misroute_report(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
