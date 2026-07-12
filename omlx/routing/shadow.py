# SPDX-License-Identifier: Apache-2.0
"""Apple FM shadow labeler: an async second opinion on routing decisions.

Fires an on-device Apple Foundation Models classification after each
routing decision and attaches the label to the pending telemetry row —
via the apple-fm-sdk Python bindings when installed, else the macOS `fm`
CLI. Never on the hot path: the request is already routed when the
classification starts, and any failure (no backend, timeout, refusal,
malformed output) drops the label silently. The point is the labeled
corpus — an independent judge accumulating alongside every decision for
the M6 outcome loop and for validating the in-process profiler.

Config and mitigations follow the fm-conductor eval (2026-07-12): greedy
decoding, enum-constrained schema, payload elision (head 500 + tail 300
chars). Labels are the eval's 4-class scale; downstream analysis bins
{TRIVIAL, SIMPLE} vs {MODERATE, COMPLEX}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

try:  # macOS 26+ with the SDK installed; CLI fallback otherwise
    import apple_fm_sdk as _fm
except ImportError:  # pragma: no cover - environment-dependent
    _fm = None

logger = logging.getLogger(__name__)

VALID_LABELS = frozenset({"TRIVIAL", "SIMPLE", "MODERATE", "COMPLEX"})

_SCHEMA = {
    "title": "RouteDecision",
    "x-order": ["label", "reason"],
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": sorted(VALID_LABELS),
            "description": "Smallest capability class that can fully answer the request",
        },
        "reason": {
            "type": "string",
            "description": "One-line reason, max 12 words",
        },
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}

_INSTRUCTIONS = """\
You are a request router. Read the USER REQUEST and output the smallest \
capability class that can fully answer it.

Classes:
  TRIVIAL   - lookup, restatement, formatting, single-step arithmetic
  SIMPLE    - short factual synthesis or short code with no design choices
  MODERATE  - multi-step reasoning, code with tradeoffs, ambiguity to resolve
  COMPLEX   - long-horizon planning, novel synthesis, deep domain expertise

Judge only what the ANSWER requires. Do NOT judge by the length of the \
request. A long request may be TRIVIAL. A short request may be COMPLEX. \
Never answer the request itself."""


def elide(text: str, head: int = 500, tail: int = 300) -> str:
    """The labeler sees only the head and tail of long payloads."""
    if len(text) <= head + tail + 50:
        return text
    return (
        text[:head]
        + f"\n[... {len(text) - head - tail} characters of pasted content elided ...]\n"
        + text[-tail:]
    )


class ShadowLabeler:
    """Classifies via the apple-fm-sdk when importable, else the fm CLI.

    Benchmarked equivalent latency (0.64s vs 0.65s median per classify);
    the SDK is preferred for typed errors — a RateLimitedError or
    GuardrailViolationError is diagnosable, a CLI nonzero exit is not.
    Classifies are serialized through a lock: shadow labels are
    best-effort telemetry, so one at a time avoids tripping the system
    model's concurrency/rate limits during agent bursts.
    """

    def __init__(self, *, use_case: str = "general", timeout_s: float = 10.0) -> None:
        self.use_case = use_case
        self.timeout_s = timeout_s
        self._schema_path: Path | None = None
        self._available: bool | None = None
        self._backend: str | None = None
        self._lock = asyncio.Lock()

    def backend(self) -> str | None:
        """ "sdk" | "cli" | None, resolved once and logged once."""
        if self._available is None:
            if _fm is not None:
                self._available = True
                self._backend = "sdk"
            elif shutil.which("fm") is not None:
                self._available = True
                self._backend = "cli"
            else:
                self._available = False
                self._backend = None
                logger.warning(
                    "Routing shadow labeler enabled but neither apple-fm-sdk "
                    "nor the `fm` CLI is available (needs macOS 26+); "
                    "labels disabled"
                )
        return self._backend

    def _schema_file(self) -> Path:
        if self._schema_path is None or not self._schema_path.exists():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="-omlx-fm-route-schema.json", delete=False
            ) as f:
                json.dump(_SCHEMA, f)
            self._schema_path = Path(f.name)
        return self._schema_path

    async def classify(self, text: str) -> dict | None:
        """One greedy fm classification. Returns a shadow record or None.

        Never raises: every failure path logs at debug and returns None.
        """
        if not text:
            return None
        backend = self.backend()
        if backend is None:
            return None
        start = time.perf_counter()
        try:
            async with self._lock:
                if backend == "sdk":
                    parsed = await asyncio.wait_for(
                        self._classify_sdk(text), timeout=self.timeout_s
                    )
                else:
                    parsed = await asyncio.wait_for(
                        self._classify_cli(text), timeout=self.timeout_s
                    )
            if parsed is None:
                return None
            label = parsed.get("label")
            if label not in VALID_LABELS:
                logger.debug("shadow labeler: invalid label %r", label)
                return None
            return {
                "provider": "apple_fm",
                "backend": backend,
                "label": label,
                "reason": parsed.get("reason"),
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }
        except TimeoutError:
            logger.debug("shadow labeler: timeout after %.1fs", self.timeout_s)
            return None
        except Exception as e:  # noqa: BLE001 - shadow path must never raise
            logger.debug("shadow labeler: %s: %s", type(e).__name__, e)
            return None

    async def _classify_sdk(self, text: str) -> dict | None:
        # Fresh session per classify: a reused session accumulates
        # transcript and earlier requests would leak into later judgments.
        session = _fm.LanguageModelSession(instructions=_INSTRUCTIONS)
        out = await session.respond(
            "USER REQUEST:\n" + elide(text),
            json_schema=_SCHEMA,
            options=_fm.GenerationOptions(sampling=_fm.SamplingMode.greedy()),
        )
        # GeneratedContent.value is a method in apple-fm-sdk 0.2.1; keep a
        # callable check so a future property change also works.
        value = getattr(out, "value", out)
        if callable(value):
            value = value()
        if not isinstance(value, dict):
            logger.debug("shadow labeler: unexpected sdk payload %r", type(value))
            return None
        return value

    async def _classify_cli(self, text: str) -> dict | None:
        proc = await asyncio.create_subprocess_exec(
            "fm",
            "respond",
            "--no-stream",
            "--greedy",
            "--schema",
            str(self._schema_file()),
            "--instructions",
            _INSTRUCTIONS,
            "--use-case",
            self.use_case,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate(("USER REQUEST:\n" + elide(text)).encode())
        if proc.returncode != 0:
            logger.debug("shadow labeler: fm exit %s", proc.returncode)
            return None
        return json.loads(stdout.decode().strip())
