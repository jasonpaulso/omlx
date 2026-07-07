# SPDX-License-Identifier: Apache-2.0
"""
Suitability store: persists per-model capability records (role, size,
health, category scores, eval provenance) that the routing dispatch table
consumes. Records are harvested by a separate accuracy-bench sweep; this
module only owns storage, role classification, and score derivation.

Category scores are derived only from baseline evals (custom generation
settings corrupt accuracy comparisons across models); non-baseline records
are kept for provenance but never feed a category score.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_VERSION = 1

CATEGORY_AXES: dict[str, str] = {
    "gsm8k": "math",
    "mathqa": "math",
    "humaneval": "code",
    "mbpp": "code",
    "livecodebench": "code",
    "mmlu": "knowledge",
    "mmlu_pro": "knowledge",
    "kmmlu": "knowledge",
    "cmmlu": "knowledge",
    "jmmlu": "knowledge",
    "truthfulqa": "knowledge",
    "arc_challenge": "reasoning",
    "hellaswag": "reasoning",
    "winogrande": "reasoning",
    "bbq": "safety",
    "safetybench": "safety",
}

# Checked in this order, case-insensitive substring match; first match wins.
_DRAFT_NAME_PATTERNS = ("dflash", "mtp", "-assistant", "draft")
_DRAFT_COMPANION_SIZE_GB = 5.0


def classify_role(model_id: str, size_gb: float | None) -> str:
    """Heuristically classify a model's role from its id and size.

    Jason's rule: chat models are >=5GB; smaller things are spec-decode
    companions.
    """
    lowered = model_id.lower()
    if any(pattern in lowered for pattern in _DRAFT_NAME_PATTERNS):
        return "draft_companion"
    if "embed" in lowered:
        return "embedding"
    if "rerank" in lowered:
        return "reranker"
    if "router" in lowered:
        return "router"
    if size_gb is not None and size_gb < _DRAFT_COMPANION_SIZE_GB:
        return "draft_companion"
    return "chat"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - mypy targets py310


def _latest_per_bench(
    evals: list[dict[str, Any]], *, baseline_only: bool
) -> dict[str, dict[str, Any]]:
    """Pick the latest record per bench, optionally restricted to baseline.

    "Latest" is max by date; ties (including missing/equal dates) are
    broken by list order, i.e. the later entry in `evals` wins.
    """
    latest: dict[str, dict[str, Any]] = {}
    latest_date: dict[str, str] = {}
    for record in evals:
        if baseline_only and not record.get("baseline"):
            continue
        bench = record["bench"]
        date = record.get("date") or ""
        if bench not in latest or date >= latest_date[bench]:
            latest[bench] = record
            latest_date[bench] = date
    return latest


def _derive_categories(evals: list[dict[str, Any]]) -> dict[str, float]:
    """Mean accuracy per axis, over the latest baseline record per bench."""
    by_axis: dict[str, list[float]] = {}
    for record in _latest_per_bench(evals, baseline_only=True).values():
        axis = record.get("axis", "other")
        by_axis.setdefault(axis, []).append(record["accuracy"])
    return {axis: sum(scores) / len(scores) for axis, scores in by_axis.items()}


class SuitabilityStore:
    """Persists per-model capability records to a JSON file.

    load()/save() are explicit: construction does not touch disk. Writes
    are atomic (tmp file + os.replace). A file written by a newer, unknown
    version is loaded read-only best-effort; save() then refuses to write
    so it can't clobber data this version doesn't understand.
    """

    def __init__(self, path: str | Path = "~/.omlx/suitability.json") -> None:
        self.path = Path(path).expanduser()
        self._data: dict[str, Any] = self._empty_data()
        self._read_only = False

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {"version": CURRENT_VERSION, "host": {}, "models": {}}

    def load(self) -> None:
        """Load from disk. Tolerant: missing/corrupt file -> empty store."""
        self._read_only = False
        if not self.path.exists():
            self._data = self._empty_data()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("suitability store: failed to read %s: %s", self.path, e)
            self._data = self._empty_data()
            return
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            logger.warning("suitability store: malformed data in %s", self.path)
            self._data = self._empty_data()
            return
        version = data.get("version")
        if not isinstance(version, int) or version > CURRENT_VERSION:
            logger.warning(
                "suitability store: unsupported version %r in %s; " "loading read-only",
                version,
                self.path,
            )
            self._read_only = True
        data.setdefault("host", {})
        self._data = data

    def save(self) -> None:
        """Atomically write the store to disk. No-op if loaded read-only."""
        if self._read_only:
            logger.warning(
                "suitability store: refusing to write %s (unsupported version)",
                self.path,
            )
            return
        self._stamp_host()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def _stamp_host(self) -> None:
        """Best-effort host info, stamped once on first write."""
        if self._data.get("host"):
            return
        try:
            hostname = socket.gethostname()
        except Exception as e:  # noqa: BLE001 - host info is best-effort
            logger.warning("suitability store: hostname lookup failed: %s", e)
            hostname = None
        memory_gb = None
        try:
            from ..utils import psutil_compat

            memory_gb = round(psutil_compat.get_total_memory() / (1024**3), 1)
        except Exception as e:  # noqa: BLE001 - host info is best-effort
            logger.warning("suitability store: memory lookup failed: %s", e)
            memory_gb = None
        self._data["host"] = {"hostname": hostname, "memory_gb": memory_gb}

    def _new_model_entry(self, model_id: str, size_gb: float | None) -> dict[str, Any]:
        return {
            "role": classify_role(model_id, size_gb),
            "role_source": "heuristic",
            "size_gb": size_gb,
            "health": {"status": "ok", "last_error": None},
            "categories": {},
            "evals": [],
            "perf": {"load_s": None, "gen_tps": None, "ttft_s": None},
        }

    def ensure_model(self, model_id: str, *, size_gb: float | None = None) -> None:
        """Create the model entry if absent; refresh size_gb otherwise.

        Role is re-derived from the heuristic unless it was set explicitly
        by a user (role_source == "user"), which is never downgraded.
        """
        models = self._data.setdefault("models", {})
        entry = models.get(model_id)
        if entry is None:
            models[model_id] = self._new_model_entry(model_id, size_gb)
            return
        if size_gb is not None:
            entry["size_gb"] = size_gb
        if entry.get("role_source") != "user":
            entry["role"] = classify_role(model_id, entry.get("size_gb"))
            entry["role_source"] = "heuristic"

    def set_role(self, model_id: str, role: str, *, source: str = "user") -> None:
        """Explicitly set a model's role, e.g. from a user override.

        Persists immediately: role overrides are rare user actions that
        must survive restarts even if no eval ever records afterward.
        """
        models = self._data.setdefault("models", {})
        entry = models.get(model_id)
        if entry is None:
            entry = self._new_model_entry(model_id, None)
            models[model_id] = entry
        entry["role"] = role
        entry["role_source"] = source
        self.save()

    def record_eval(
        self,
        model_id: str,
        *,
        bench: str,
        accuracy: float,
        n: int,
        baseline: bool,
        thinking: bool,
        time_s: float,
        median_q_time_s: float | None = None,
        load_s: float | None = None,
        source: str = "suitability_sweep",
        run_id: str | None = None,
        date: str | None = None,
    ) -> None:
        """Append a provenance record, re-derive categories, mark healthy."""
        self.ensure_model(model_id)
        entry = self._data["models"][model_id]
        entry["evals"].append(
            {
                "bench": bench,
                "axis": CATEGORY_AXES.get(bench, "other"),
                "accuracy": accuracy,
                "n": n,
                "date": date or _now_iso(),
                "baseline": baseline,
                "thinking": thinking,
                "time_s": time_s,
                "median_q_time_s": median_q_time_s,
                "load_s": load_s,
                "source": source,
                "run_id": run_id,
            }
        )
        entry["categories"] = _derive_categories(entry["evals"])
        entry["health"] = {"status": "ok", "last_error": None}
        if load_s is not None:
            entry["perf"]["load_s"] = load_s
        self.save()

    def record_unhealthy(
        self, model_id: str, *, phase: str, message: str, date: str | None = None
    ) -> None:
        """Mark a model unhealthy. Existing evals/categories are untouched."""
        self.ensure_model(model_id)
        entry = self._data["models"][model_id]
        entry["health"] = {
            "status": "unhealthy",
            "last_error": {
                "ts": date or _now_iso(),
                "phase": phase,
                "message": message,
            },
        }
        self.save()

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        models: dict[str, dict[str, Any]] = self._data.get("models", {})
        return models.get(model_id)

    def all_models(self) -> dict[str, dict[str, Any]]:
        models: dict[str, dict[str, Any]] = self._data.get("models", {})
        return models

    def ranked(
        self,
        axis: str,
        *,
        roles: tuple[str, ...] = ("chat",),
        baseline_only: bool = True,
    ) -> list[tuple[str, float]]:
        """Healthy models with a score for `axis`, filtered by role, desc.

        baseline_only=True (default) reads the stored `categories` score,
        which is always derived from baseline evals only. baseline_only=
        False recomputes the axis score on the fly from the latest record
        per bench regardless of baseline flag -- useful for inspecting
        non-baseline data, but never what dispatch should use.
        """
        results: list[tuple[str, float]] = []
        for model_id, entry in self._data.get("models", {}).items():
            if entry.get("role") not in roles:
                continue
            if entry.get("health", {}).get("status") != "ok":
                continue
            if baseline_only:
                score = entry.get("categories", {}).get(axis)
            else:
                per_bench = _latest_per_bench(
                    entry.get("evals", []), baseline_only=False
                )
                scores = [
                    r["accuracy"] for r in per_bench.values() if r.get("axis") == axis
                ]
                score = sum(scores) / len(scores) if scores else None
            if score is None:
                continue
            results.append((model_id, score))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results
