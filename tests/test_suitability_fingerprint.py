# SPDX-License-Identifier: Apache-2.0
"""Weights fingerprinting: does a model's on-disk state still match its scores?

Covers omlx.admin.suitability.weights_fingerprint / model_staleness. All
tmp_path-based — real files, stat only, no models loaded.
"""

import time

import pytest

from omlx.admin import suitability as suit
from omlx.routing.store import SuitabilityStore


class _FakeEntry:
    def __init__(self, model_path):
        self.model_path = str(model_path)


class _FakePool:
    def __init__(self, paths: dict):
        self.paths = paths

    def get_entry(self, model_id):
        path = self.paths.get(model_id)
        return _FakeEntry(path) if path else None


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "gemma-4-12b-oQ8e"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"weights-v1")
    (d / "config.json").write_text('{"model_type": "gemma"}')
    return d


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch, model_dir):
    monkeypatch.setattr(suit, "_engine_pool", _FakePool({"m": model_dir}))
    monkeypatch.setattr(suit, "_fingerprint_cache", {})
    yield


def _requantize(model_dir, payload=b"weights-v2-different-size"):
    """Rewrite the weights the way a re-quantize does: same name, new bytes."""
    (model_dir / "model.safetensors").write_bytes(payload)
    suit._fingerprint_cache.clear()  # bypass the TTL memo, not the logic


class TestWeightsFingerprint:
    def test_stable_across_calls(self, model_dir):
        first = suit.weights_fingerprint("m")
        suit._fingerprint_cache.clear()
        assert first is not None
        assert suit.weights_fingerprint("m") == first

    def test_changes_when_weights_are_rewritten(self, model_dir):
        before = suit.weights_fingerprint("m")
        _requantize(model_dir)
        assert suit.weights_fingerprint("m") != before

    def test_changes_when_only_the_template_changes(self, model_dir):
        # Jason's case: same tensors, new chat template, same model id.
        before = suit.weights_fingerprint("m")
        (model_dir / "chat_template.jinja").write_text("{{ messages }}")
        suit._fingerprint_cache.clear()
        assert suit.weights_fingerprint("m") != before

    def test_unchanged_dir_keeps_fingerprint(self, model_dir):
        before = suit.weights_fingerprint("m")
        suit._fingerprint_cache.clear()
        (model_dir / "config.json").read_text()  # reads must not perturb it
        assert suit.weights_fingerprint("m") == before

    def test_unknown_model_is_none(self):
        assert suit.weights_fingerprint("never-heard-of-it") is None

    def test_no_engine_pool_is_none(self, monkeypatch):
        monkeypatch.setattr(suit, "_engine_pool", None)
        assert suit.weights_fingerprint("m") is None

    def test_missing_directory_is_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            suit, "_engine_pool", _FakePool({"m": tmp_path / "not-there"})
        )
        assert suit.weights_fingerprint("m") is None

    def test_empty_directory_is_none(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(suit, "_engine_pool", _FakePool({"m": empty}))
        assert suit.weights_fingerprint("m") is None

    def test_result_is_memoized_within_ttl(self, model_dir):
        first = suit.weights_fingerprint("m")
        # No cache clear: the rewrite must NOT be observed inside the TTL.
        (model_dir / "model.safetensors").write_bytes(b"weights-v2")
        assert suit.weights_fingerprint("m") == first

    def test_memo_expires(self, model_dir, monkeypatch):
        first = suit.weights_fingerprint("m")
        (model_dir / "model.safetensors").write_bytes(b"weights-v2")
        real_monotonic = time.monotonic
        monkeypatch.setattr(
            suit.time,
            "monotonic",
            lambda: real_monotonic() + suit._FINGERPRINT_TTL_S + 1,
        )
        assert suit.weights_fingerprint("m") != first

    def test_nested_files_are_included(self, model_dir):
        before = suit.weights_fingerprint("m")
        nested = model_dir / "shards"
        nested.mkdir()
        (nested / "part-1.safetensors").write_bytes(b"more")
        suit._fingerprint_cache.clear()
        assert suit.weights_fingerprint("m") != before


class TestModelStaleness:
    def _store_with_eval(self, tmp_path, fp):
        store = SuitabilityStore(tmp_path / "suitability.json")
        store.record_eval(
            "m",
            bench="mmlu",
            accuracy=0.8,
            n=30,
            baseline=True,
            thinking=False,
            time_s=1.0,
            weights_fingerprint=fp,
        )
        return store

    def test_flags_scores_measured_before_a_requantize(self, tmp_path, model_dir):
        store = self._store_with_eval(tmp_path, suit.weights_fingerprint("m"))
        assert suit.model_staleness("m", store.get_model("m"))["stale"] is False

        _requantize(model_dir)
        result = suit.model_staleness("m", store.get_model("m"))
        assert result["stale"] is True
        assert result["records"] == ["mmlu"]
        assert result["current"] == suit.weights_fingerprint("m")

    def test_unstamped_scores_are_not_stale(self, tmp_path, model_dir):
        store = self._store_with_eval(tmp_path, None)
        _requantize(model_dir)
        assert suit.model_staleness("m", store.get_model("m"))["stale"] is False

    def test_unknown_model_never_stale(self, tmp_path):
        store = self._store_with_eval(tmp_path, "whatever")
        result = suit.model_staleness("gone", store.get_model("m"))
        assert result == {"current": None, "stale": False, "records": []}

    def test_rebench_clears_the_flag(self, tmp_path, model_dir):
        store = self._store_with_eval(tmp_path, suit.weights_fingerprint("m"))
        _requantize(model_dir)
        assert suit.model_staleness("m", store.get_model("m"))["stale"] is True

        store.record_eval(
            "m",
            bench="mmlu",
            accuracy=0.7,
            n=30,
            baseline=True,
            thinking=False,
            time_s=1.0,
            weights_fingerprint=suit.weights_fingerprint("m"),
        )
        assert suit.model_staleness("m", store.get_model("m"))["stale"] is False

    def test_clearing_scores_clears_the_flag(self, tmp_path, model_dir):
        store = self._store_with_eval(tmp_path, suit.weights_fingerprint("m"))
        _requantize(model_dir)
        store.clear_scores("m")
        assert suit.model_staleness("m", store.get_model("m"))["stale"] is False
