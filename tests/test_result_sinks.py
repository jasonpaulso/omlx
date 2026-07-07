# SPDX-License-Identifier: Apache-2.0
"""Tests for the accuracy benchmark result/run-status observer hooks.

These sinks let an external module (the suitability harvester) observe
completed benchmark results and terminal run status without the benchmark
runner depending on it. Both must be best-effort: a raising sink must
never break a run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.accuracy_benchmark import (
    AccuracyBenchmarkRequest,
    create_run,
    run_accuracy_benchmark,
    set_result_sink,
    set_run_status_sink,
)


def _make_mock_result():
    mock_result = MagicMock()
    mock_result.benchmark_name = "mmlu"
    mock_result.accuracy = 0.75
    mock_result.total_questions = 4
    mock_result.correct_count = 3
    mock_result.time_seconds = 1.0
    mock_result.category_scores = None
    mock_result.thinking_used = False
    return mock_result


def _make_fake_evaluator(mock_result, run_side_effect=None):
    mock_evaluator = MagicMock()
    mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
    if run_side_effect is not None:
        mock_evaluator.run = AsyncMock(side_effect=run_side_effect)
    else:
        mock_evaluator.run = AsyncMock(return_value=mock_result)
    return mock_evaluator


def _make_mock_pool():
    mock_pool = MagicMock()
    mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
    mock_pool.get_engine = AsyncMock(return_value=AsyncMock())
    mock_pool._unload_engine = AsyncMock()
    mock_pool._settings_manager = MagicMock()
    return mock_pool


class TestResultSink:
    def setup_method(self):
        set_result_sink(None)

    def teardown_method(self):
        set_result_sink(None)

    @pytest.mark.asyncio
    async def test_sink_receives_result_data(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        received = []
        set_result_sink(received.append)

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"
        assert len(received) == 1
        assert received[0]["model_id"] == "m"
        assert received[0]["benchmark"] == "mmlu"

    @pytest.mark.asyncio
    async def test_raising_sink_does_not_fail_run(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        def bad_sink(_data):
            raise RuntimeError("sink exploded")

        set_result_sink(bad_sink)

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"
        assert len(run.results) == 1

    @pytest.mark.asyncio
    async def test_no_sink_registered_is_a_noop(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"


class TestRunStatusSink:
    def setup_method(self):
        set_run_status_sink(None)

    def teardown_method(self):
        set_run_status_sink(None)

    @pytest.mark.asyncio
    async def test_sink_sees_completed_status_on_success(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        calls = []
        set_run_status_sink(lambda model_id, status, err: calls.append(
            (model_id, status, err)
        ))

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert calls == [("m", "completed", None)]

    @pytest.mark.asyncio
    async def test_sink_sees_error_status_and_message_on_failure(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        calls = []
        set_run_status_sink(lambda model_id, status, err: calls.append(
            (model_id, status, err)
        ))

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(
            _make_mock_result(), run_side_effect=RuntimeError("eval blew up")
        )
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert len(calls) == 1
        model_id, status, err = calls[0]
        assert model_id == "m"
        assert status == "error"
        assert err == "eval blew up"

    @pytest.mark.asyncio
    async def test_raising_status_sink_does_not_propagate(self):
        req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"mmlu": 100})
        run = create_run(req)

        def bad_sink(_model_id, _status, _err):
            raise RuntimeError("status sink exploded")

        set_run_status_sink(bad_sink)

        mock_pool = _make_mock_pool()
        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        # Must not raise, and the TTL/baseline cleanup below it must still run.
        assert run.status == "completed"
        assert mock_pool._suppress_ttl is False
