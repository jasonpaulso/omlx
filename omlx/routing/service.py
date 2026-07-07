# SPDX-License-Identifier: Apache-2.0
"""
Routing service: glues the profiler and policy together, resolves a
concrete target model for "auto" requests, and records telemetry.

Never raises from route_chat_request: any classify failure (no engine
getter configured, engine error, timeout) fails open to
settings.policy.fail_open_target.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omlx.routing import table as dispatch_table
from omlx.routing.policy import decide
from omlx.routing.profiler import RouterFeatures, RouterProfiler
from omlx.settings import RoutingSettings

logger = logging.getLogger(__name__)

EngineGetter = Callable[[str], Awaitable[Any]]
ModelsGetter = Callable[[], dict[str, dict]]
ResidentGetter = Callable[[], set[str]]


@dataclass
class RouteDecision:
    """Result of routing one request."""

    target: str  # concrete model id
    rule_fired: str
    override: str | None  # None | "tools" | "turns"
    features: RouterFeatures | None
    raw_analysis: str | None
    classify_ms: float
    candidates: list[tuple[str, float]] | None = None  # table dispatch only

    @property
    def header_value(self) -> str:
        """Value for the x-omlx-route response header."""
        return (
            f"{self.target}; rule={self.rule_fired}; "
            f"classify_ms={self.classify_ms:.0f}"
        )


def _msg_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off a message/part that may be a dict or a pydantic model."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _last_user_content(messages: list) -> str:
    """Extract the last user message's text, flattening multimodal parts.

    Messages/parts may be plain dicts or pydantic request models (attribute
    access), depending on the caller.
    """
    for msg in reversed(messages):
        if _msg_field(msg, "role") == "user":
            content = _msg_field(msg, "content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    part_type = _msg_field(part, "type")
                    if part_type not in (None, "text"):
                        continue
                    text = _msg_field(part, "text")
                    if text:
                        parts.append(text)
                return " ".join(parts)
            return content or ""
    return ""


def _has_non_text_parts(messages: list) -> bool:
    """True if any message carries a non-text content part (image_url etc.).

    Scans the whole conversation, not just the last turn: an image three
    turns back still requires a vision-capable target. File parts are
    normally flattened to text by MarkItDown preprocessing before routing
    sees the request, so in practice this triggers on image/audio/video
    part types.
    """
    for msg in messages:
        content = _msg_field(msg, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _msg_field(part, "type") not in (None, "text"):
                return True
    return False


class RoutingService:
    """Classifies "auto" requests and rewrites them to a concrete target."""

    def __init__(self, settings: RoutingSettings) -> None:
        self.settings = settings
        self._profiler = RouterProfiler(settings.router_model)
        self._engine_getter: EngineGetter | None = None
        self._models_getter: ModelsGetter | None = None
        self._resident_getter: ResidentGetter | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._writer_task: asyncio.Task | None = None

    def set_engine_getter(self, fn: EngineGetter) -> None:
        """Set the async callable used to resolve the router engine.

        Wired from the server lifespan: fn(model_id) -> engine.
        """
        self._engine_getter = fn

    def set_table_sources(
        self, models_getter: ModelsGetter, resident_getter: ResidentGetter
    ) -> None:
        """Wire suitability-store and pool-residency snapshots (M3 N-way).

        Cheap sync callables evaluated per routed request. Table dispatch
        stays inert unless settings.table_dispatch.enabled is also true.
        """
        self._models_getter = models_getter
        self._resident_getter = resident_getter

    async def route_chat_request(
        self,
        *,
        messages: list,
        has_tools: bool,
        request_id: str,
        stream: bool,
        endpoint: str = "chat",
    ) -> RouteDecision:
        """Classify and route one request. Never raises."""
        start = time.perf_counter()
        cfg = self.settings.policy

        # Shape rule precedes everything (decision #11: semantic routing
        # layers BEHIND shape-based rules). A request carrying image/audio
        # parts must reach a vision-capable target — text-only targets
        # cannot see the content regardless of tools or complexity.
        if _has_non_text_parts(messages):
            vision_target = self.settings.targets.get("vision")
            if vision_target:
                decision = RouteDecision(
                    target=vision_target,
                    rule_fired="shape:vision",
                    override=None,
                    features=None,
                    raw_analysis=None,
                    classify_ms=(time.perf_counter() - start) * 1000,
                )
                self._record_decision(decision, request_id, endpoint, stream)
                return decision
            logger.warning(
                "Routing: request has non-text parts but no targets.vision "
                "configured; text-only dispatch will not see them"
            )

        override: str | None = None
        if has_tools and cfg.agentic_override.on_tools:
            override = "tools"
        else:
            user_turns = sum(1 for m in messages if _msg_field(m, "role") == "user")
            if user_turns > cfg.agentic_override.max_user_turns:
                override = "turns"

        features: RouterFeatures | None = None
        raw_analysis: str | None = None

        if override is None:
            try:
                if self._engine_getter is None:
                    raise RuntimeError("no engine getter configured")

                async def _classify() -> tuple[RouterFeatures, str]:
                    text = _last_user_content(messages)
                    engine = await self._engine_getter(self.settings.router_model)  # type: ignore[misc]
                    return await self._profiler.classify(engine, text)

                features, raw_analysis = await asyncio.wait_for(
                    _classify(), timeout=self.settings.classify_timeout_s
                )
            except TimeoutError:
                decision = self._fail_open("timeout", override, start)
                self._record_decision(decision, request_id, endpoint, stream)
                return decision
            except Exception as e:  # noqa: BLE001 - must never raise
                logger.warning("routing classify failed: %s", e)
                decision = self._fail_open("error", override, start)
                self._record_decision(decision, request_id, endpoint, stream)
                return decision

        # M3: table dispatch (opt-in) — measured per-axis leaders beat the
        # binary pair when the table has data; binary remains the fallback.
        table_choice = self._try_table(features, override)
        if table_choice is not None and table_choice.target is not None:
            decision = RouteDecision(
                target=table_choice.target,
                rule_fired=table_choice.rule,
                override=override,
                features=features,
                raw_analysis=raw_analysis,
                classify_ms=(time.perf_counter() - start) * 1000,
                candidates=table_choice.candidates or None,
            )
            self._record_decision(decision, request_id, endpoint, stream)
            return decision

        target_key, rule_fired = decide(features, override, cfg)
        classify_ms = (time.perf_counter() - start) * 1000
        decision = RouteDecision(
            target=self.settings.targets.get(target_key, target_key),
            rule_fired=rule_fired,
            override=override,
            features=features,
            raw_analysis=raw_analysis,
            classify_ms=classify_ms,
        )
        self._record_decision(decision, request_id, endpoint, stream)
        return decision

    def _try_table(
        self, features: RouterFeatures | None, override: str | None
    ) -> dispatch_table.TableChoice | None:
        """Attempt N-way table dispatch. Returns None to fall back to binary.

        Never raises: any store/pool snapshot failure logs and falls back.
        """
        cfg = self.settings.table_dispatch
        if (
            not cfg.enabled
            or self._models_getter is None
            or self._resident_getter is None
        ):
            return None
        try:
            models = self._models_getter()
            if not models:
                return None
            generalist = cfg.default_target or self.settings.targets.get("big")
            if override is not None:
                # Agentic overrides route to the generalist spine in N-way
                # mode, mirroring the binary policy's big-target behavior.
                if generalist:
                    return dispatch_table.TableChoice(
                        generalist, f"override:{override}", []
                    )
                return None
            return dispatch_table.choose(
                features,
                models,
                self._resident_getter(),
                escalate_at=self.settings.policy.escalate_complexity_at,
                residency_epsilon=cfg.residency_epsilon,
                max_interactive_median_q_time_s=cfg.max_interactive_median_q_time_s,
                default_target=generalist,
            )
        except Exception as e:  # noqa: BLE001 - dispatch must never 5xx
            logger.warning("table dispatch failed, falling back to binary: %s", e)
            return None

    def _fail_open(
        self, reason: str, override: str | None, start: float
    ) -> RouteDecision:
        target_key = self.settings.policy.fail_open_target
        return RouteDecision(
            target=self.settings.targets.get(target_key, target_key),
            rule_fired=f"fail_open:{reason}",
            override=override,
            features=None,
            raw_analysis=None,
            classify_ms=(time.perf_counter() - start) * 1000,
        )

    def record_outcome(
        self,
        request_id: str,
        *,
        completion_tokens: int | None,
        finish_reason: str | None,
        gen_ms: float | None,
    ) -> None:
        """Attach an outcome to a pending telemetry row and enqueue the write.

        Sync and exception-free: may be called from a streaming response's
        cleanup path, so it must never raise.
        """
        try:
            row = self._pending.pop(request_id, None)
            if row is None:
                return
            row["outcome"] = {
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "gen_ms": gen_ms,
            }
            self._enqueue(row)
        except Exception as e:  # noqa: BLE001 - must never raise
            logger.warning("routing record_outcome failed: %s", e)

    async def close(self) -> None:
        """Flush pending telemetry (rows without outcomes flush with outcome=null)."""
        for row in list(self._pending.values()):
            self._enqueue(row)
        self._pending.clear()
        if self._queue is not None:
            await self._queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None

    def _record_decision(
        self, decision: RouteDecision, request_id: str, endpoint: str, stream: bool
    ) -> None:
        if not self.settings.telemetry.enabled:
            return
        self._ensure_writer()
        self._pending[request_id] = {
            "ts": _now_iso(),
            "request_id": request_id,
            "endpoint": endpoint,
            "stream": stream,
            "override": decision.override,
            "raw_analysis": decision.raw_analysis,
            "features": asdict(decision.features) if decision.features else None,
            "rule_fired": decision.rule_fired,
            "target": decision.target,
            "classify_ms": round(decision.classify_ms, 1),
            "candidates_considered": (
                [[m, round(s, 4)] for m, s in decision.candidates]
                if decision.candidates
                else None
            ),
            "outcome": None,
        }

    def _ensure_writer(self) -> None:
        if self._writer_task is not None:
            return
        self._queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._writer_loop())

    def _enqueue(self, row: dict[str, Any]) -> None:
        if self._queue is None:
            return
        self._queue.put_nowait(row)

    async def _writer_loop(self) -> None:
        path = Path(self.settings.telemetry.path).expanduser()
        assert self._queue is not None
        while True:
            row = await self._queue.get()
            if row is None:
                self._queue.task_done()
                break
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError as e:
                logger.warning("routing telemetry write failed: %s", e)
            self._queue.task_done()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - mypy targets py310
