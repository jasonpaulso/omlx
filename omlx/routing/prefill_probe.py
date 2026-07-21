# SPDX-License-Identifier: Apache-2.0
"""Prefill-throughput probe for M8 est_ttft dispatch.

Measures how fast each model prefills a prompt at fixed depths so the router
can penalise slow-prefill-at-depth leaders — a model can be interactive on the
short-prompt suitability bench yet take a minute to prefill a 24k-token agent
prompt (Claude Code incident 2026-07-13: gemma-4-31B ~230 tok/s vs Ornith-35B
~1,240 on the same 14.5k prompt → 62.6s vs 12.1s cold TTFT). ``median_q_time_s``
is prompt-length-blind and cannot see this.

One model load + a handful of single-token generations, timing prefill only.
Per-host (prefill speed is pure hardware; "tables don't travel" applies doubly).
Best-effort: any failure is logged and skipped, never raised — the dispatch
gate fails open on missing data. Prompts are salted-unique so prefix caching
cannot fake the number (same hygiene as bench_qmlx.py).
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

logger = logging.getLogger(__name__)

# Fixed measurement depths (tokens). Chosen to bracket the interactive range:
# a short turn, a medium context, and a deep agent prompt.
DEFAULT_DEPTHS: tuple[int, ...] = (2048, 8192, 24576)


# Short, common, single-token filler words (leading-space form is one BPE
# token in the tokenizers this serves). Rotating a small set keeps each word
# ~1 token so the prompt's token count tracks the requested depth; earlier
# per-word salts (`{salt}{i:x}`) tokenized to ~5 tokens each, inflating every
# depth ~5x (a "24576" probe became ~120k tokens → OOM + multi-minute probes).
_FILLER = ("the", "of", "and", "to", "in", "is", "it", "for", "on", "as")


def build_salted_prompt(approx_tokens: int, salt: str) -> str:
    """A ~approx_tokens single-token-word filler prompt, unique per salt.

    Uniqueness matters: a repeated prompt would be served from the prefix
    cache and time as ~instant, faking the throughput. A unique salt prefix
    makes the whole prompt novel each probe (so prefix caching can't serve
    it), while the rotating single-token filler makes the token count track
    `approx_tokens` (±~10%; the probe reports the engine's actual
    prompt_tokens, so the small drift doesn't matter to the tps math).
    """
    approx_tokens = max(1, approx_tokens)
    words = [_FILLER[i % len(_FILLER)] for i in range(approx_tokens)]
    return f"probe{salt} " + " ".join(words)


async def probe_prefill(
    engine: Any,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    *,
    salt: str | None = None,
) -> dict[int, float]:
    """Time prefill-only throughput at each depth on an already-loaded engine.

    Returns ``{depth: tokens_per_second}``, keyed by the nominal target depth
    (so dispatch lookups by estimated prompt size hit the intended bucket). A
    depth whose generation fails or produces no usable timing is omitted. Uses
    the engine's native ``prompt_tps`` when present, else wall-clocks the
    single-token generation (prefill dominates a 1-token decode at these
    depths, so ``prompt_tokens / elapsed`` is a faithful prefill rate).
    """
    salt = salt or secrets.token_hex(4)
    out: dict[int, float] = {}
    for depth in depths:
        prompt = build_salted_prompt(depth, f"{salt}d{depth}")
        try:
            t0 = time.perf_counter()
            result = await engine.generate(prompt=prompt, max_tokens=1, temperature=0.0)
            elapsed = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - a bad depth must not kill the probe
            logger.warning("prefill probe failed at depth %d: %s", depth, e)
            continue
        n = int(getattr(result, "prompt_tokens", 0) or 0)
        logger.debug(
            "prefill probe depth=%d actual_prompt_tokens=%d elapsed=%.2fs",
            depth,
            n,
            elapsed,
        )
        native = float(getattr(result, "prompt_tps", 0.0) or 0.0)
        if native > 0:
            out[depth] = native
            continue
        if elapsed > 0:
            out[depth] = (n or depth) / elapsed
    return out


async def run_prefill_probe(
    engine_pool: Any,
    store: Any,
    model_id: str,
    *,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    weights_fingerprint: str | None = None,
) -> dict[int, float] | None:
    """Acquire the model, probe prefill throughput, persist it. Best-effort.

    Loads the model as an LM (``force_lm``), runs :func:`probe_prefill`, and
    writes the result via ``store.record_prefill``. Returns the samples, or
    None if the engine could not be acquired or nothing measured. Never
    raises: probing is an optimisation, not a serving path.

    `weights_fingerprint` is passed through to the record so the table can
    flag a probe that predates the weights now on disk. The caller supplies
    it (routing doesn't reach into the model directory).
    """
    try:
        engine = await engine_pool.get_engine(
            model_id, force_lm=True, stamp_activity=False
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("prefill probe could not load %s: %s", model_id, e)
        return None
    samples = await probe_prefill(engine, depths)
    if not samples:
        logger.warning("prefill probe measured nothing for %s", model_id)
        return None
    try:
        store.record_prefill(model_id, samples, weights_fingerprint=weights_fingerprint)
    except Exception as e:  # noqa: BLE001
        logger.warning("prefill probe could not record %s: %s", model_id, e)
        return None
    logger.info(
        "prefill probe %s: %s",
        model_id,
        ", ".join(f"{d}={tps:.0f}tps" for d, tps in sorted(samples.items())),
    )
    return samples
