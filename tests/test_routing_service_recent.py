# SPDX-License-Identifier: Apache-2.0
"""Tests for RoutingService's recent-decisions ring buffer (admin Router tab)."""

import json

import pytest

from omlx.routing.service import (
    _RECENT_MAX,
    RouteDecision,
    RoutingService,
    read_telemetry_tail,
)
from omlx.settings import RoutingSettings


def make_service(tmp_path, **overrides) -> RoutingService:
    data = {
        "enabled": True,
        "telemetry": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")},
    }
    data.update(overrides)
    return RoutingService(RoutingSettings.from_dict(data))


def decision(target="model-a", rule="small") -> RouteDecision:
    return RouteDecision(
        target=target,
        rule_fired=rule,
        override=None,
        features=None,
        raw_analysis=None,
        classify_ms=1.0,
    )


def record(service, request_id, target="model-a", rule="small"):
    service._record_decision(decision(target, rule), request_id, "chat", False)


class TestRecentBuffer:
    @pytest.mark.asyncio
    async def test_rows_returned_newest_first(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1", target="model-a")
        record(svc, "r2", target="model-b")
        rows = svc.recent_decisions(limit=10)
        assert [r["request_id"] for r in rows] == ["r2", "r1"]

    @pytest.mark.asyncio
    async def test_outcome_mutation_visible_in_buffer(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1")
        svc.record_outcome(
            "r1",
            completion_tokens=42,
            finish_reason="stop",
            gen_ms=100.0,
            ttft_ms=50.0,
        )
        rows = svc.recent_decisions(limit=10)
        assert rows[0]["outcome"]["completion_tokens"] == 42
        assert rows[0]["outcome"]["ttft_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_limit_clamped(self, tmp_path):
        svc = make_service(tmp_path)
        for i in range(3):
            record(svc, f"r{i}")
        assert len(svc.recent_decisions(limit=0)) == 1
        assert len(svc.recent_decisions(limit=2)) == 2
        # An absurd limit is capped at the buffer size, not an error.
        assert len(svc.recent_decisions(limit=10_000)) == 3

    @pytest.mark.asyncio
    async def test_buffer_capped_at_recent_max(self, tmp_path):
        svc = make_service(tmp_path)
        for i in range(_RECENT_MAX + 10):
            record(svc, f"r{i}")
        rows = svc.recent_decisions(limit=_RECENT_MAX)
        assert len(rows) == _RECENT_MAX
        assert rows[0]["request_id"] == f"r{_RECENT_MAX + 9}"

    @pytest.mark.asyncio
    async def test_telemetry_disabled_records_nothing(self, tmp_path):
        svc = make_service(
            tmp_path,
            telemetry={"enabled": False, "path": str(tmp_path / "decisions.jsonl")},
        )
        record(svc, "r1")
        assert svc.recent_decisions(limit=10) == []

    @pytest.mark.asyncio
    async def test_pending_count(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1")
        record(svc, "r2")
        assert svc.pending_count == 2
        svc.record_outcome("r1", completion_tokens=1, finish_reason="stop", gen_ms=1.0)
        assert svc.pending_count == 1

    @pytest.mark.asyncio
    async def test_shadow_status_disabled(self, tmp_path):
        svc = make_service(tmp_path)
        assert svc.shadow_status() == {"enabled": False, "backend": None}


class TestFileTailMerge:
    def _write_rows(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    @pytest.mark.asyncio
    async def test_file_tail_tops_up_after_restart(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        self._write_rows(
            path,
            [
                {"request_id": "old1", "target": "m"},
                {"request_id": "old2", "target": "m"},
            ],
        )
        svc = make_service(tmp_path)
        record(svc, "live1")
        rows = svc.recent_decisions(limit=10)
        # Live buffer rows first, then file rows newest-first.
        assert [r["request_id"] for r in rows] == ["live1", "old2", "old1"]

    @pytest.mark.asyncio
    async def test_file_duplicates_of_buffered_rows_deduped(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        svc = make_service(tmp_path)
        record(svc, "r1")
        # Simulate the same row already flushed to disk (stale copy).
        self._write_rows(path, [{"request_id": "r1", "target": "stale"}])
        rows = svc.recent_decisions(limit=10)
        assert len(rows) == 1
        assert rows[0]["target"] == "model-a"  # buffer version wins

    @pytest.mark.asyncio
    async def test_no_file_read_when_buffer_fills_limit(self, tmp_path):
        svc = make_service(tmp_path)
        record(svc, "r1")
        record(svc, "r2")
        rows = svc.recent_decisions(limit=2)
        assert [r["request_id"] for r in rows] == ["r2", "r1"]


class TestReadTelemetryTail:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_telemetry_tail(tmp_path / "nope.jsonl", 10) == []

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"request_id": "a"}\nnot json\n[1, 2]\n{"request_id": "b"}\n',
            encoding="utf-8",
        )
        rows = read_telemetry_tail(path, 10)
        assert [r["request_id"] for r in rows] == ["b", "a"]

    def test_limit_takes_last_lines(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            "".join(json.dumps({"request_id": f"r{i}"}) + "\n" for i in range(5)),
            encoding="utf-8",
        )
        rows = read_telemetry_tail(path, 2)
        assert [r["request_id"] for r in rows] == ["r4", "r3"]
