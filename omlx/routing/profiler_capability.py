# SPDX-License-Identifier: Apache-2.0
"""
Classification-family router profiler (M4.5).

A single-forward-pass BERT-style alternative to the generative Supra
profiler. Loads a ModernBERT multi-label sequence classifier (default:
``massaindustries/modernbert-capability-classifier``) via mlx-embeddings and
maps its per-axis capability scores onto the same ``RouterFeatures`` the
policy and N-way table consume.

Why a separate module: this pulls in ``mlx.core`` and ``mlx_embeddings`` at
runtime. ``profiler.py`` is imported by ``policy.py``/``service.py`` on every
routing import, so the heavy MLX imports live here and stay lazy (inside
methods), keeping the routing core importable on non-Apple test envs.

Mapping (capability sigmoid scores in [0,1] -> RouterFeatures):
- ``code``       = coding        >= threshold
- ``math``       = math_reasoning >= threshold
- ``domain``     = argmax capability axis name (telemetry only)
- ``complexity`` = 1..5 proxy from max(coding, math_reasoning,
  planning_agentic); the classifier has no native complexity head, so this
  drives the binary policy's escalate-on-complexity rules. The N-way table
  reads only ``code``/``math`` (axis) + ``complexity`` (escalation tier).
- ``route_token``= None (only the generative profiler emits one).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

from omlx.routing.profiler import RouterFeatures

logger = logging.getLogger(__name__)

# Capability axes whose strength most indicates the request needs a stronger
# model. world_knowledge is deliberately excluded: it steers the table's
# "knowledge" axis (via not being code/math), not the complexity tier.
_COMPLEXITY_AXES = ("coding", "math_reasoning", "planning_agentic")

# Default multi-label ModernBERT capability classifier. HF repo id or a local
# path; passed straight to mlx_embeddings.load (which resolves/downloads it).
DEFAULT_CAPABILITY_MODEL = "massaindustries/modernbert-capability-classifier"


def features_from_scores(scores: dict[str, float], threshold: float) -> RouterFeatures:
    """Map capability axis scores -> RouterFeatures. Pure; never raises.

    ``scores`` maps capability-axis label -> sigmoid score in [0,1].
    """
    if not scores:
        return RouterFeatures(
            domain=None, complexity=None, math=False, code=False, route_token=None
        )

    code = scores.get("coding", 0.0) >= threshold
    math = scores.get("math_reasoning", 0.0) >= threshold
    domain = max(scores, key=lambda k: scores[k])

    hard = max((scores.get(axis, 0.0) for axis in _COMPLEXITY_AXES), default=0.0)
    complexity = max(1, min(5, 1 + round(4 * hard)))

    return RouterFeatures(
        domain=domain,
        complexity=complexity,
        math=bool(math),
        code=bool(code),
        route_token=None,
    )


class CapabilityProfiler:
    """Single-forward-pass ModernBERT capability classifier.

    Implements the same ``classify(engine, text)`` contract as
    ``RouterProfiler`` but owns its own MLX model (loaded lazily, cached) and
    does not use the engine pool — so it ignores the passed ``engine``. The
    model is a multi-label sequence classifier, distinct from a chat engine,
    and must not be routed through engine_pool's LM/reranker loaders.
    """

    needs_engine = False

    def __init__(self, model_id: str, threshold: float = 0.5) -> None:
        self.model_id = model_id or DEFAULT_CAPABILITY_MODEL
        self.threshold = threshold
        self._model: Any = None
        self._processor: Any = None
        self._labels: list[str] | None = None
        self._loaded = False

    def _load_sync(self) -> None:
        """Load the classifier via mlx-embeddings. Runs on the MLX thread."""
        from mlx_embeddings import load as mlx_emb_load

        model, processor = mlx_emb_load(self.model_id)

        # mlx-embeddings applies softmax to multi-label logits (num_labels>1),
        # which is wrong for this model's independent per-axis heads. Force the
        # regression path so _process_outputs returns raw logits; we apply our
        # own sigmoid. Set on both the module and its config defensively.
        try:
            model.is_regression = True
            if hasattr(model, "config"):
                model.config.is_regression = True
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "capability profiler: could not force raw-logits path on %s; "
                "scores may be softmax-normalized",
                self.model_id,
            )

        # Resolve the axis label order from the model config when available,
        # so we map by name rather than assuming a fixed head order.
        labels: list[str] | None = None
        cfg = getattr(model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if isinstance(id2label, dict) and id2label:
            try:
                labels = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))]
            except (ValueError, TypeError):
                labels = list(id2label.values())

        self._model = model
        self._processor = processor
        self._labels = labels
        self._loaded = True

    def _classify_sync(self, text: str) -> dict[str, float]:
        """Tokenize + forward + sigmoid on the MLX thread. Returns axis->score."""
        import mlx.core as mx

        if not self._loaded:
            self._load_sync()

        processor = self._processor
        # mlx-embeddings returns a TokenizerWrapper; unwrap to the underlying
        # tokenizer which supports __call__ batch encoding.
        if type(processor).__name__ == "TokenizerWrapper" and hasattr(
            processor, "_tokenizer"
        ):
            processor = processor._tokenizer

        inputs = processor(
            [text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.pooler_output  # raw logits (is_regression forced)
        probs = mx.sigmoid(logits)
        mx.eval(probs)
        values = cast("list[Any]", probs[0].tolist())
        row = [float(x) for x in values]

        labels = self._labels or [f"label_{i}" for i in range(len(row))]
        return {labels[i]: row[i] for i in range(min(len(labels), len(row)))}

    async def warm_up(self) -> None:
        """Best-effort preload so the first request doesn't pay cold-load."""
        loop = asyncio.get_running_loop()
        from omlx.engine_core import get_mlx_executor

        await loop.run_in_executor(get_mlx_executor(), self._load_sync)

    async def classify(self, engine: Any, text: str) -> tuple[RouterFeatures, str]:
        """Classify one prompt. Ignores ``engine`` (interface compatibility).

        Raises on model/forward failure; RoutingService owns timeout and
        fail-open handling, exactly as for the generative profiler.
        """
        loop = asyncio.get_running_loop()
        from omlx.engine_core import get_mlx_executor

        scores = await loop.run_in_executor(
            get_mlx_executor(), self._classify_sync, text
        )
        raw = json.dumps({k: round(v, 4) for k, v in scores.items()})
        return features_from_scores(scores, self.threshold), raw
