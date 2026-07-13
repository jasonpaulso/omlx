# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/routing/misroute.py (M6.2 misroute measurement).

Pure aggregation over synthetic telemetry rows; see docs/ROUTING.md, "M6.2
-- misroute measurement (read-only)" for the report contract this exercises.
"""

from __future__ import annotations

import json
import subprocess
import sys

from omlx.routing.misroute import load_rows, misroute_report


def _decision(request_id, **fields):
    row = {"request_id": request_id}
    row.update(fields)
    return row


def _feedback(request_id, score, source="client"):
    return {
        "kind": "feedback",
        "request_id": request_id,
        "score": score,
        "source": source,
    }


class TestEmptyAndNullRates:
    def test_empty_corpus_well_formed(self):
        report = misroute_report([])
        assert report["coverage"] == {
            "decisions": 0,
            "shadowed": 0,
            "with_outcome": 0,
            "feedback_rows": 0,
            "implicit_feedback": 0,
            "joined_decisions": 0,
        }
        assert report["direct"] == {
            "joined_n": 0,
            "negative_n": 0,
            "negative_rate": None,
            "by_rule": {},
            "by_target": {},
        }
        assert report["proxy"]["shadowed_n"] == 0
        assert report["proxy"]["matrix"] == {}
        assert report["proxy"]["mean_signed_gap"] is None
        assert report["proxy"]["disagree_rate"] is None
        assert report["proxy"]["over_rate"] is None
        assert report["proxy"]["under_rate"] is None
        assert report["cost"]["trivial_n"] == 0
        assert report["cost"]["trivial_median_ttft_ms"] is None
        assert report["cost"]["corpus_median_ttft_ms"] is None
        assert report["segments"]["by_endpoint"] == {}
        assert report["segments"]["sticky_held"] is None
        assert report["gate"] == {
            "joined_n": 0,
            "eligible": False,
            "criterion_a": None,
            "criterion_b": None,
            "met": False,
        }

    def test_no_feedback_gives_null_direct_rate(self):
        rows = [_decision("a", rule_fired="r1", target="t1", endpoint="chat")]
        report = misroute_report(rows)
        assert report["direct"]["joined_n"] == 0
        assert report["direct"]["negative_rate"] is None


class TestDirectPlane:
    def test_negative_is_min_joined_score_below_half(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("a", 0.8),
            _feedback("a", 0.3),
        ]
        report = misroute_report(rows)
        assert report["direct"]["joined_n"] == 1
        assert report["direct"]["negative_n"] == 1
        assert report["direct"]["negative_rate"] == 1.0

    def test_zero_score_is_negative(self):
        # tool_error/negation emit exactly 0.0 -- falsy, so a `score or 1.0`
        # default would silently flip the strongest negative signal.
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("a", 0.0, source="implicit"),
        ]
        report = misroute_report(rows)
        assert report["direct"]["negative_n"] == 1
        assert report["direct"]["negative_rate"] == 1.0
        assert report["direct"]["by_target"]["t1"]["negative"] == 1

    def test_positive_feedback_not_negative(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("a", 0.9),
        ]
        report = misroute_report(rows)
        assert report["direct"]["negative_n"] == 0
        assert report["direct"]["negative_rate"] == 0.0

    def test_feedback_rows_never_counted_as_decisions(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("a", 0.9),
            _feedback("b", 0.1),  # unmatched -- no decision "b"
        ]
        report = misroute_report(rows)
        assert report["coverage"]["decisions"] == 1
        assert report["coverage"]["feedback_rows"] == 2

    def test_unmatched_feedback_dropped_from_join(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("orphan", 0.1),
        ]
        report = misroute_report(rows)
        assert report["direct"]["joined_n"] == 0
        assert report["coverage"]["feedback_rows"] == 1

    def test_implicit_feedback_counted(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1", endpoint="chat"),
            _feedback("a", 0.0, source="implicit"),
        ]
        report = misroute_report(rows)
        assert report["coverage"]["implicit_feedback"] == 1


class TestProxyPlane:
    def test_matrix_counts_and_gap_directionality(self):
        rows = [
            # gap = 1 - 1 = 0: agreement
            _decision("a", shadow={"label": "TRIVIAL"}, features={"complexity": 1}),
            # gap = 3 - 1 = 2: disagreement, over
            _decision("b", shadow={"label": "TRIVIAL"}, features={"complexity": 3}),
            # gap = 3 - 5 = -2: disagreement, under
            _decision("c", shadow={"label": "COMPLEX"}, features={"complexity": 3}),
            # gap = 3 - 2 = 1: below the |gap| >= 2 threshold, not disagreement
            _decision("d", shadow={"label": "SIMPLE"}, features={"complexity": 3}),
        ]
        report = misroute_report(rows)
        proxy = report["proxy"]
        assert proxy["shadowed_n"] == 4
        assert proxy["matrix"] == {
            "TRIVIAL": {1: 1, 3: 1},
            "COMPLEX": {3: 1},
            "SIMPLE": {3: 1},
        }
        assert proxy["disagree_n"] == 2
        assert proxy["over_n"] == 1
        assert proxy["under_n"] == 1
        assert proxy["mean_signed_gap"] == (0 + 2 - 2 + 1) / 4

    def test_old_vintage_rows_without_shadow_or_features_dont_crash(self):
        rows = [
            _decision("a", rule_fired="r1", target="t1"),  # no shadow/outcome/features
            _decision("b", shadow={"label": "TRIVIAL"}),  # shadow but no features
            _decision("c", features={"complexity": 3}),  # features but no shadow
        ]
        report = misroute_report(rows)
        assert report["coverage"]["decisions"] == 3
        # "a" has no shadow at all -> not counted in shadowed.
        assert report["proxy"]["shadowed_n"] == 1  # only "b" carries a shadow dict
        assert report["proxy"]["matrix"] == {}  # "b" lacks features.complexity
        # The cost plane only needs shadow.label -- "b" still counts there
        # even without a complexity feature.
        assert report["cost"]["trivial_n"] == 1


class TestCostPlane:
    def test_trivial_vs_corpus_medians(self):
        rows = [
            _decision(
                "a",
                shadow={"label": "TRIVIAL"},
                outcome={"ttft_ms": 100, "gen_ms": 200},
            ),
            _decision(
                "b",
                shadow={"label": "TRIVIAL"},
                outcome={"ttft_ms": 300, "gen_ms": 400},
            ),
            _decision(
                "c",
                shadow={"label": "COMPLEX"},
                outcome={"ttft_ms": 5000, "gen_ms": 6000},
            ),
        ]
        report = misroute_report(rows)
        cost = report["cost"]
        assert cost["trivial_n"] == 2
        assert cost["trivial_median_ttft_ms"] == 200
        assert cost["trivial_median_gen_ms"] == 300
        assert cost["corpus_median_ttft_ms"] == 300
        assert cost["corpus_median_gen_ms"] == 400


class TestGate:
    def _joined_decisions(self, n, negative_ids=()):
        rows = []
        for i in range(n):
            rid = f"d{i}"
            rows.append(_decision(rid, rule_fired="rule", target=f"target-{i}"))
            score = 0.1 if rid in negative_ids else 0.9
            rows.append(_feedback(rid, score))
        return rows

    def test_ineligible_under_50_joined(self):
        rows = self._joined_decisions(10, negative_ids={"d0"})
        report = misroute_report(rows)
        gate = report["gate"]
        assert gate["joined_n"] == 10
        assert gate["eligible"] is False
        assert gate["criterion_a"] is None
        assert gate["criterion_b"] is None
        assert gate["met"] is False

    def test_criterion_a_fires_on_concentrated_negative_segment(self):
        # 50 joined decisions; 10 negative feedback rows all on the same
        # target ("t-bad") -> that segment's rate is well above 2x baseline
        # and has >= 10 negative rows.
        rows = []
        negative_ids = {f"bad{i}" for i in range(10)}
        for i in range(15):
            rid = f"bad{i}" if i < 10 else f"bad-pos{i}"
            rows.append(_decision(rid, rule_fired="r-bad", target="t-bad"))
            rows.append(_feedback(rid, 0.1 if rid in negative_ids else 0.9))
        for i in range(35):
            rid = f"good{i}"
            rows.append(_decision(rid, rule_fired="r-good", target="t-good"))
            rows.append(_feedback(rid, 0.9))
        report = misroute_report(rows)
        gate = report["gate"]
        assert gate["joined_n"] == 50
        assert gate["eligible"] is True
        assert gate["criterion_a"] is True
        assert gate["met"] is True
        # baseline = 10/50 = 0.2; t-bad segment should show the concentration.
        assert report["direct"]["by_target"]["t-bad"]["negative"] == 10
        assert report["direct"]["by_target"]["t-bad"]["rate"] >= 0.4

    def test_criterion_b_fires_on_under_route_corroboration(self):
        # 50 joined decisions, no single target/rule concentrates negatives
        # (keeping criterion_a false), but 10 of them are also shadow
        # under-route disagreements and are all negative -- corroborating
        # the proxy signal against a low corpus baseline.
        rows = []
        for i in range(10):
            rid = f"under{i}"
            rows.append(
                _decision(
                    rid,
                    rule_fired="rule",
                    target=f"target-{i}",  # distinct targets: no segment hits negative>=10
                    shadow={"label": "COMPLEX"},
                    features={"complexity": 3},  # gap = 3-5 = -2 -> under
                )
            )
            rows.append(_feedback(rid, 0.1))  # negative
        for i in range(40):
            rid = f"fine{i}"
            rows.append(
                _decision(
                    rid,
                    rule_fired="rule",
                    target="target-fine",
                    shadow={"label": "TRIVIAL"},
                    features={"complexity": 1},  # gap = 0 -> agreement
                )
            )
            rows.append(_feedback(rid, 0.9))  # positive
        report = misroute_report(rows)
        gate = report["gate"]
        assert gate["joined_n"] == 50
        assert gate["eligible"] is True
        assert gate["criterion_a"] is False
        assert gate["criterion_b"] is True
        assert gate["met"] is True
        assert report["proxy"]["under_rate"] == 10 / 50

    def test_met_false_when_neither_criterion_fires(self):
        rows = self._joined_decisions(50)  # 50 joined, all positive, no shadow
        report = misroute_report(rows)
        gate = report["gate"]
        assert gate["eligible"] is True
        assert gate["criterion_a"] is False
        assert gate["criterion_b"] is False
        assert gate["met"] is False


class TestLoadRows:
    def test_skips_garbage_lines(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        path.write_text(
            '{"request_id": "a"}\n'
            "not json at all\n"
            "\n"
            "123\n"  # valid JSON but not a dict
            '{"request_id": "b"}\n',
            encoding="utf-8",
        )
        rows = load_rows(path)
        assert [r["request_id"] for r in rows] == ["a", "b"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_rows(tmp_path / "nope.jsonl") == []


class TestCLISmoke:
    def test_cli_prints_valid_json(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        path.write_text(
            json.dumps({"request_id": "a", "target": "t1"}) + "\n", encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, "-m", "omlx.routing.misroute", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(result.stdout)
        assert report["coverage"]["decisions"] == 1
