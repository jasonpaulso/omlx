# SPDX-License-Identifier: Apache-2.0
"""Tests for accuracy benchmark baseline mode.

Baseline mode forces the target model to load and sample with stock
settings, bypassing custom per-model settings that were shown to corrupt
suitability eval scores (see omlx.model_settings.ModelSettingsManager
baseline bypass and omlx.admin.accuracy_benchmark.run_accuracy_benchmark).
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.accuracy_benchmark import (
    AccuracyBenchmarkRequest,
    create_run,
    run_accuracy_benchmark,
)
from omlx.model_settings import ModelSettings, ModelSettingsManager


class TestModelSettingsManagerBaseline:
    """Tests for the per-id baseline bypass in ModelSettingsManager."""

    def test_get_settings_baseline_returns_stock_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings(
                "model-a",
                ModelSettings(presence_penalty=1.5, dflash_draft_model="draft-x"),
            )

            manager.set_baseline_ids({"model-a"})
            settings = manager.get_settings("model-a")

            assert settings.presence_penalty is None
            assert settings.dflash_draft_model is None
            assert settings == ModelSettings()

    def test_baseline_does_not_affect_other_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("model-a", ModelSettings(presence_penalty=1.5))
            manager.set_settings("model-b", ModelSettings(presence_penalty=0.3))

            manager.set_baseline_ids({"model-a"})

            assert manager.get_settings("model-a").presence_penalty is None
            assert manager.get_settings("model-b").presence_penalty == 0.3

    def test_clear_baseline_ids_restores_custom_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("model-a", ModelSettings(presence_penalty=1.5))

            manager.set_baseline_ids({"model-a"})
            assert manager.get_settings("model-a").presence_penalty is None

            manager.clear_baseline_ids()
            assert manager.get_settings("model-a").presence_penalty == 1.5

    def test_get_settings_for_request_fallback_path_honors_baseline(self):
        """The non-profile fallback in get_settings_for_request delegates to
        get_settings(), so it must observe the same baseline bypass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("model-a", ModelSettings(presence_penalty=1.5))

            manager.set_baseline_ids({"model-a"})
            settings = manager.get_settings_for_request("model-a")

            assert settings.presence_penalty is None

    def test_set_baseline_ids_replaces_active_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("model-a", ModelSettings(presence_penalty=1.5))
            manager.set_settings("model-b", ModelSettings(presence_penalty=0.3))

            manager.set_baseline_ids({"model-a"})
            manager.set_baseline_ids({"model-b"})  # replaces, doesn't union

            assert manager.get_settings("model-a").presence_penalty == 1.5
            assert manager.get_settings("model-b").presence_penalty is None


class TestAccuracyBenchmarkRequestBaselineMode:
    def test_baseline_mode_defaults_false(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        assert req.baseline_mode is False

    def test_baseline_mode_accepts_true(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            baseline_mode=True,
        )
        assert req.baseline_mode is True


def _make_fake_evaluator(mock_result):
    mock_evaluator = MagicMock()
    mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
    mock_evaluator.run = AsyncMock(return_value=mock_result)
    return mock_evaluator


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


class TestRunAccuracyBenchmarkBaselineMode:
    @pytest.mark.asyncio
    async def test_baseline_mode_sets_and_clears_ids_around_load(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            baseline_mode=True,
        )
        run = create_run(req)

        call_order = []

        mock_engine = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool._unload_engine = AsyncMock()
        mock_pool._settings_manager = MagicMock()
        mock_pool._settings_manager.set_baseline_ids = MagicMock(
            side_effect=lambda ids: call_order.append(("set", ids))
        )
        mock_pool._settings_manager.clear_baseline_ids = MagicMock(
            side_effect=lambda: call_order.append(("clear", None))
        )

        async def fake_get_engine(model_id, force_lm=True):
            call_order.append(("get_engine", model_id))
            return mock_engine

        mock_pool.get_engine = AsyncMock(side_effect=fake_get_engine)

        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"
        assert call_order[0] == ("set", {"test-model"})
        assert call_order[1][0] == "get_engine"
        assert call_order[-1] == ("clear", None)

        # sampling_kwargs must be empty in baseline mode
        _, kwargs = evaluator.run.call_args
        assert kwargs["sampling_kwargs"] == {}

    @pytest.mark.asyncio
    async def test_baseline_mode_clears_ids_even_on_failure(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            baseline_mode=True,
        )
        run = create_run(req)

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(side_effect=RuntimeError("load failed"))
        mock_pool._unload_engine = AsyncMock()
        mock_pool._settings_manager = MagicMock()

        with patch.dict("omlx.eval.BENCHMARKS", {}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "error"
        mock_pool._settings_manager.set_baseline_ids.assert_called_once_with(
            {"test-model"}
        )
        mock_pool._settings_manager.clear_baseline_ids.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_baseline_mode_never_touches_baseline_ids(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            baseline_mode=False,
        )
        run = create_run(req)

        mock_engine = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(return_value=mock_engine)
        mock_pool._unload_engine = AsyncMock()
        mock_pool._settings_manager = MagicMock()
        mock_pool._settings_manager.get_settings = MagicMock(
            return_value=ModelSettings()
        )

        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"
        mock_pool._settings_manager.set_baseline_ids.assert_not_called()
        mock_pool._settings_manager.clear_baseline_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_data_includes_baseline_load_s_run_id(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            baseline_mode=True,
        )
        run = create_run(req)

        mock_engine = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(return_value=mock_engine)
        mock_pool._unload_engine = AsyncMock()
        mock_pool._settings_manager = MagicMock()

        evaluator = _make_fake_evaluator(_make_mock_result())
        mock_bench_cls = MagicMock(return_value=evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        assert len(run.results) == 1
        result_data = run.results[0]
        assert result_data["baseline"] is True
        assert result_data["run_id"] == run.bench_id
        assert "load_s" in result_data
        assert isinstance(result_data["load_s"], float)
