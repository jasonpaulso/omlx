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
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omlx.routing import table as dispatch_table
from omlx.routing.policy import decide
from omlx.routing.profiler import RouterFeatures, make_profiler
from omlx.routing.shadow import ShadowLabeler, elide
from omlx.settings import RoutingSettings

logger = logging.getLogger(__name__)

EngineGetter = Callable[[str], Awaitable[Any]]
ModelsGetter = Callable[[], dict[str, dict]]
ResidentGetter = Callable[[], set[str]]
FitBudgetGetter = Callable[[], float | None]
EnabledGetter = Callable[[], set[str]]
# Target-health sources (roster validity + the global default-model fallback
# flag). Both optional: absent -> validation is skipped and routing behaves
# exactly as before (fail-open).
ValidGetter = Callable[[str], bool]
FallbackGetter = Callable[[], bool]

# Orphan flush (P1-D): a 507/disconnected request never calls
# record_outcome, so its pending row would otherwise sit in memory until
# shutdown. Flush anything this stale on every new decision.
_ORPHAN_MAX_AGE_S = 600

# Ring buffer of recent decision rows kept for the admin Router tab. Row
# dicts are shared with _pending, so outcomes and shadow labels appear in
# the buffer as they land without extra bookkeeping.
_RECENT_MAX = 256


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
    unfit: list[str] | None = None  # table dispatch only: excluded, can't fit
    disabled: list[str] | None = None  # table dispatch only: routing opt-out
    # Original target id that did not resolve on the live roster. Set whether
    # the request was substituted to a valid target or passed through to the
    # server's default-or-404 contract; surfaced in telemetry and the header.
    invalid_target: str | None = None

    @property
    def header_value(self) -> str:
        """Value for the x-omlx-route response header."""
        base = (
            f"{self.target}; rule={self.rule_fired}; "
            f"classify_ms={self.classify_ms:.0f}"
        )
        if self.invalid_target:
            base += f"; invalid_target={self.invalid_target}"
        return base


def _msg_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off a message/part that may be a dict or a pydantic model."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_text(msg: Any) -> str:
    """Flatten one message's textual content, skipping non-text parts.

    Tool payloads (`tool_use`/`tool_result`), thinking blocks, and media
    parts are excluded — only `text` parts (or a plain string content)
    count as classify-able text. Messages/parts may be plain dicts or
    pydantic request models (attribute access), depending on the caller.
    """
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


def _last_user_content(messages: list) -> str:
    """Extract the last user message's text, flattening multimodal parts."""
    for msg in reversed(messages):
        if _msg_field(msg, "role") == "user":
            return _message_text(msg)
    return ""


def build_classify_window(messages: list, *, max_turns: int, max_chars: int) -> str:
    """Bounded multi-turn transcript for classification (phase C).

    Collects user/assistant text newest-first (tool payloads, thinking
    blocks, and media parts excluded; system prompts and role="tool"
    messages skipped), elides each message like the shadow labeler does,
    and stops at `max_turns` entries or `max_chars` total. Rendered
    chronologically with role prefixes so follow-ups like "make it
    faster" carry the context they refer to. The newest entry is always
    included even if it alone exceeds the budget.
    """
    max_turns = max(1, max_turns)
    max_chars = max(1, max_chars)
    entries: list[str] = []  # newest first
    total = 0
    for msg in reversed(messages):
        role = _msg_field(msg, "role")
        if role not in ("user", "assistant"):
            continue
        text = _message_text(msg).strip()
        if not text:
            continue
        line = f"{'User' if role == 'user' else 'Assistant'}: {elide(text)}"
        if entries and total + len(line) > max_chars:
            break
        entries.append(line)
        total += len(line) + 1
        if len(entries) >= max_turns:
            break
    return "\n".join(reversed(entries))


# Part-type classification for the shape rule. Anthropic-protocol agent
# traffic represents tool calls, tool outputs, and reasoning as content
# blocks; production evidence 2026-07-11 showed that treating every
# non-text part as vision routed 95% of a real Claude Code session to the
# vision target and bypassed the classifier entirely. Text-flow parts are
# textual control-flow, not media. Unknown part types fail open to the
# classifier rather than forcing a modality target.
_TEXT_FLOW_PART_TYPES = frozenset(
    {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}
)
# OpenAI: image_url, file. Anthropic: image, document (PDF needs a VLM).
_VISION_PART_TYPES = frozenset(
    {"image", "image_url", "document", "file", "video", "video_url"}
)
# OpenAI: input_audio. "audio" reserved for symmetric future shapes.
_AUDIO_PART_TYPES = frozenset({"input_audio", "audio"})


def _detect_modality(messages: list) -> str | None:
    """Return "vision"/"audio" if the conversation carries media parts.

    Scans the whole conversation, not just the last turn: an image three
    turns back still requires a vision-capable target. Recurses into
    tool_result nested content — a screenshot returned by a browser tool
    is a real vision need even though the enclosing block is text-flow.
    File/document parts are normally flattened to text by MarkItDown
    preprocessing before routing sees the request; if one survives to
    here it still needs a vision-capable target. Vision wins when both
    modalities appear.
    """
    saw_audio = False
    for msg in messages:
        content = _msg_field(msg, "content")
        if not isinstance(content, list):
            continue
        stack = list(content)
        while stack:
            part = stack.pop()
            part_type = _msg_field(part, "type")
            if part_type in _VISION_PART_TYPES:
                return "vision"
            if part_type in _AUDIO_PART_TYPES:
                saw_audio = True
            elif part_type == "tool_result":
                nested = _msg_field(part, "content")
                if isinstance(nested, list):
                    stack.extend(nested)
    return "audio" if saw_audio else None


@dataclass
class ImplicitSignal:
    """An implicit outcome proxy derived from the multi-turn stream (M6.1)."""

    kind: str  # "tool_error" | "negation" | "rephrase" | "approval"
    score: float  # 0.0 (bad) .. 1.0 (good)
    marker: str  # the cue that matched, for auditing


# High-precision / low-recall cues. A false positive poisons the outcome
# corpus worse than a miss, so a cue must *begin* the newest user message
# (after a light filler strip) — a substring test is not safe here
# ("incorrect" appears in "correct the incorrect assumption"; "No thanks"
# is a decline, not approval). Meta-feedback about the prior answer almost
# always leads the turn. Order below is precedence: dissatisfaction wins
# over approval ("no, thanks anyway" is negative).
_NEGATION_CUES = (
    "no,",
    "no.",
    "nope",
    "that's wrong",
    "thats wrong",
    "that is wrong",
    "that's not",
    "thats not",
    "that is not",
    "wrong answer",
    "incorrect",
    "you didn't",
    "you did not",
    "that doesn't work",
    "that does not work",
    "doesn't work",
    "does not work",
    "still broken",
    "still failing",
    "still doesn't",
    "still not",
    "not what i",
)
_REPHRASE_CUES = (
    "try again",
    "again please",
    "please try again",
    "do it again",
    "redo",
    "that didn't help",
    "that didnt help",
)
_APPROVAL_CUES = (
    "thanks",
    "thank you",
    "perfect",
    "that works",
    "works now",
    "that's correct",
    "thats correct",
    "that's right",
    "thats right",
    "looks good",
    "lgtm",
    "awesome",
)
# Leading span of the newest user message searched for a cue.
_IMPLICIT_LEAD_CHARS = 80
# Conversational preambles stripped before the leading-cue test so
# "ok, that's wrong" and "hmm, try again" still register.
_IMPLICIT_FILLER_RE = re.compile(
    r"^(?:ok(?:ay)?|hmm+|wait|actually|well|so|um+|uh+|hey|yeah|right)[\s,.:;-]+"
)


def _lead_matches(lead: str, cues: tuple[str, ...]) -> str | None:
    """Return the first cue that the leading text begins with, else None."""
    for cue in cues:
        if lead.startswith(cue):
            return cue
    return None


def _last_assistant_index(messages: list) -> int:
    """Index of the last assistant message, or -1 if there is none."""
    for i in range(len(messages) - 1, -1, -1):
        if _msg_field(messages[i], "role") == "assistant":
            return i
    return -1


def _trailing_tool_error(messages: list, after: int) -> bool:
    """True if any message after `after` carries an error tool_result.

    An agent turn returns tool output as ``tool_result`` blocks; an
    ``is_error`` flag (Anthropic) — or an OpenAI ``role: "tool"`` message
    whose text opens with an error marker — means the route's tool call
    failed, the strongest free dissatisfaction signal there is.
    """
    for msg in messages[after + 1 :]:
        content = _msg_field(msg, "content")
        parts = content if isinstance(content, list) else [msg]
        for part in parts:
            if _msg_field(part, "type") == "tool_result" and _msg_field(
                part, "is_error"
            ):
                return True
    return False


def detect_implicit_signal(messages: list) -> ImplicitSignal | None:
    """Derive a free outcome proxy for the previous turn's route (M6.1).

    Pure and cheap (string matching only). Returns None unless the
    conversation has a prior assistant turn to judge and the newest turn
    carries a distinctive cue. High precision by design — see the cue notes.
    """
    a = _last_assistant_index(messages)
    if a < 0:
        return None  # nothing prior to attribute a signal to

    if _trailing_tool_error(messages, a):
        return ImplicitSignal("tool_error", 0.0, "tool_result.is_error")

    # Newest user text after the last assistant turn. A pure detector
    # returns None on malformed content rather than raising.
    try:
        text = ""
        for msg in messages[a + 1 :]:
            if _msg_field(msg, "role") == "user":
                text = _message_text(msg)
        lead = str(text).strip().lower()[:_IMPLICIT_LEAD_CHARS]
    except Exception:  # noqa: BLE001
        return None
    lead = _IMPLICIT_FILLER_RE.sub("", lead)
    if not lead:
        return None

    cue = _lead_matches(lead, _NEGATION_CUES)
    if cue:
        return ImplicitSignal("negation", 0.0, cue)
    cue = _lead_matches(lead, _REPHRASE_CUES)
    if cue:
        return ImplicitSignal("rephrase", 0.2, cue)
    cue = _lead_matches(lead, _APPROVAL_CUES)
    if cue:
        return ImplicitSignal("approval", 1.0, cue)
    return None


def _user_text_hash(text: str) -> str | None:
    """Stable content-addressed key for a user message (M6.1 join key).

    Normalizes whitespace/case so the same message hashes identically when
    it re-appears in the next turn's history. None for empty text.
    """
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if not norm:
        return None
    return hashlib.sha1(norm[:2000].encode("utf-8")).hexdigest()


class RoutingService:
    """Classifies "auto" requests and rewrites them to a concrete target."""

    def __init__(self, settings: RoutingSettings) -> None:
        self.settings = settings
        self._profiler = make_profiler(settings)
        self._engine_getter: EngineGetter | None = None
        self._models_getter: ModelsGetter | None = None
        self._resident_getter: ResidentGetter | None = None
        self._fit_budget_getter: FitBudgetGetter | None = None
        self._enabled_getter: EnabledGetter | None = None
        self._valid_getter: ValidGetter | None = None
        self._model_fallback_getter: FallbackGetter | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=_RECENT_MAX)
        # Implicit feedback (M6.1, off by default): bounded content-hash index
        # mapping each decision's last-user-text hash -> request_id, so the
        # next turn can attribute a free outcome proxy to the prior decision
        # without any persistent conversation identity.
        self._decision_by_userhash: OrderedDict[str, str] = OrderedDict()
        self._queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._writer_task: asyncio.Task | None = None
        # Apple FM shadow labeler (off by default): async second-opinion
        # labels attached to pending telemetry rows, never on the hot path.
        sl = settings.shadow_labeler
        self._shadow: ShadowLabeler | None = (
            ShadowLabeler(use_case=sl.use_case, timeout_s=sl.timeout_s)
            if sl.enabled
            else None
        )
        self._shadow_tasks: set[asyncio.Task] = set()

    def set_engine_getter(self, fn: EngineGetter) -> None:
        """Set the async callable used to resolve the router engine.

        Wired from the server lifespan: fn(model_id) -> engine.
        """
        self._engine_getter = fn

    def set_table_sources(
        self,
        models_getter: ModelsGetter,
        resident_getter: ResidentGetter,
        fit_budget_getter: FitBudgetGetter | None = None,
        enabled_getter: EnabledGetter | None = None,
    ) -> None:
        """Wire suitability-store and pool-residency snapshots (M3 N-way).

        Cheap sync callables evaluated per routed request. Table dispatch
        stays inert unless settings.table_dispatch.enabled is also true.

        `fit_budget_getter` (P0-C, optional) returns the never-fits ceiling
        in GB -- the ceiling minus pinned/resident memory, computed by the
        caller from EnginePool.get_status(). None (absent or returning
        None) disables fit filtering; dispatch behaves exactly as before.

        `enabled_getter` (optional) returns the set of model ids the operator
        has opted in as routing targets (per-model `enable_routing`). An
        empty set means nobody opted in -> the gate is inert and every chat
        model stays a candidate (fail-open). None (absent) is equivalent.
        """
        self._models_getter = models_getter
        self._resident_getter = resident_getter
        self._fit_budget_getter = fit_budget_getter
        self._enabled_getter = enabled_getter

    def set_validity_sources(
        self,
        valid_getter: ValidGetter,
        model_fallback_getter: FallbackGetter,
    ) -> None:
        """Wire roster-validity + default-model-fallback snapshots.

        `valid_getter(model_id)` returns True if the id resolves to a model on
        the live roster (alias-aware, non-loading). `model_fallback_getter()`
        returns the global ``model.model_fallback`` flag — the single switch
        that permits serving a model other than the one requested. Both are
        cheap sync callables. When unset (or when either raises), target
        validation is skipped and routing behaves exactly as before: a stale
        target passes through to the server's model-load contract.
        """
        self._valid_getter = valid_getter
        self._model_fallback_getter = model_fallback_getter

    def _target_resolves(self, model_id: str | None) -> bool:
        """True if `model_id` resolves on the live roster.

        Fail-open: with no validity source wired (or if the check raises),
        treat the target as valid so validation can never itself break routing.
        """
        if not model_id or self._valid_getter is None:
            return True
        try:
            return bool(self._valid_getter(model_id))
        except Exception:  # noqa: BLE001 - validation must never break routing
            return True

    def _fallback_enabled(self) -> bool:
        """Global ``model.model_fallback`` flag (False when unwired/erroring)."""
        if self._model_fallback_getter is None:
            return False
        try:
            return bool(self._model_fallback_getter())
        except Exception:  # noqa: BLE001
            return False

    def _finalize_target(
        self, proposed: str, *, substitute: bool = True
    ) -> tuple[str, str | None]:
        """Validate a proposed target; substitute a valid one or pass through.

        Returns ``(target, invalid_target)``. ``invalid_target`` is None when
        ``proposed`` resolves on the roster. When it does not:

        - if ``substitute`` and ``model_fallback`` is on, swap in the first
          valid configured routing target (default_target, big, small, vision)
          and return ``(substitute, proposed)``;
        - otherwise return ``(proposed, proposed)`` unchanged, so the server's
          model-load contract (default-model-or-404, per model_fallback)
          applies downstream.

        Never raises. ``substitute=False`` (modality path) validates and records
        but never swaps: a text generalist cannot see media, so a stale vision
        target must fall through rather than silently downgrade.
        """
        if self._target_resolves(proposed):
            return proposed, None
        if substitute and self._fallback_enabled():
            t = self.settings.targets
            for cand in (
                self.settings.table_dispatch.default_target,
                t.get("big"),
                t.get("small"),
                t.get("vision"),
            ):
                if cand and cand != proposed and self._target_resolves(cand):
                    return cand, proposed
        return proposed, proposed

    def validate_targets(self) -> dict[str, dict[str, Any]]:
        """Roster health of every configured routing target.

        Returns ``{slot: {"id": model_id|None, "resolves": bool|None}}`` for
        the small/big/vision/default_target/fail_open_target slots. ``resolves``
        is None when the slot is unset. Cheap (set membership) and safe to call
        at startup and on every admin Router-tab fetch.
        """
        t = self.settings.targets
        fo_key = self.settings.policy.fail_open_target
        slots = {
            "small": t.get("small"),
            "big": t.get("big"),
            "vision": t.get("vision"),
            "default_target": self.settings.table_dispatch.default_target,
            "fail_open_target": (t.get(fo_key, fo_key) if fo_key else None),
        }
        return {
            name: {
                "id": mid,
                "resolves": (None if not mid else self._target_resolves(mid)),
            }
            for name, mid in slots.items()
        }

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
        # parts must reach a modality-capable target — text-only targets
        # cannot see the content regardless of tools or complexity.
        modality = _detect_modality(messages)
        if modality is not None:
            modality_target = self.settings.targets.get(modality)
            if modality_target:
                final_target, invalid = self._finalize_target(
                    modality_target, substitute=False
                )
                decision = RouteDecision(
                    target=final_target,
                    rule_fired=f"shape:{modality}",
                    override=None,
                    features=None,
                    raw_analysis=None,
                    classify_ms=(time.perf_counter() - start) * 1000,
                    invalid_target=invalid,
                )
                self._record_decision(decision, request_id, endpoint, stream, messages)
                return decision
            logger.warning(
                "Routing: request has %s parts but no targets.%s "
                "configured; text-only dispatch will not see them",
                modality,
                modality,
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

                async def _classify() -> tuple[RouterFeatures, str]:
                    text = self._classify_text(messages)
                    engine = None
                    # The capability profiler owns its own model and ignores
                    # the engine; only the generative profiler needs one from
                    # the pool.
                    if getattr(self._profiler, "needs_engine", True):
                        if self._engine_getter is None:
                            raise RuntimeError("no engine getter configured")
                        engine = await self._engine_getter(self.settings.router_model)
                    return await self._profiler.classify(engine, text)

                features, raw_analysis = await asyncio.wait_for(
                    _classify(), timeout=self.settings.classify_timeout_s
                )
            except TimeoutError:
                decision = self._fail_open("timeout", override, start)
                self._record_decision(decision, request_id, endpoint, stream, messages)
                return decision
            except Exception as e:  # noqa: BLE001 - must never raise
                logger.warning("routing classify failed: %s", e)
                decision = self._fail_open("error", override, start)
                self._record_decision(decision, request_id, endpoint, stream, messages)
                return decision

        # M3: table dispatch (opt-in) — measured per-axis leaders beat the
        # binary pair when the table has data; binary remains the fallback.
        table_choice = self._try_table(features, override)
        if table_choice is not None and table_choice.target is not None:
            final_target, invalid = self._finalize_target(table_choice.target)
            decision = RouteDecision(
                target=final_target,
                rule_fired=table_choice.rule,
                override=override,
                features=features,
                raw_analysis=raw_analysis,
                classify_ms=(time.perf_counter() - start) * 1000,
                candidates=table_choice.candidates or None,
                unfit=table_choice.unfit or None,
                disabled=table_choice.disabled or None,
                invalid_target=invalid,
            )
            self._record_decision(decision, request_id, endpoint, stream, messages)
            return decision

        target_key, rule_fired = decide(features, override, cfg)
        classify_ms = (time.perf_counter() - start) * 1000
        proposed = self.settings.targets.get(target_key, target_key)
        final_target, invalid = self._finalize_target(proposed)
        decision = RouteDecision(
            target=final_target,
            rule_fired=rule_fired,
            override=override,
            features=features,
            raw_analysis=raw_analysis,
            classify_ms=classify_ms,
            invalid_target=invalid,
        )
        self._record_decision(decision, request_id, endpoint, stream, messages)
        return decision

    def _classify_text(self, messages: list) -> str:
        """Profiler input: multi-turn window when enabled, else last user text.

        Falls back to the last user message if the window comes back empty
        (e.g. a conversation with no user/assistant text at all), so
        enabling the window can only add context, never remove it.
        """
        w = self.settings.classify_window
        if w.enabled:
            text = build_classify_window(
                messages, max_turns=w.max_turns, max_chars=w.max_chars
            )
            if text:
                return text
        return _last_user_content(messages)

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
            fit_budget_gb = (
                self._fit_budget_getter() if self._fit_budget_getter else None
            )
            enabled_ids = self._enabled_getter() if self._enabled_getter else None
            if override is not None:
                # Agentic overrides dispatch on the measured agentic axis
                # (toolcall bench) with the shared residency/load tiebreak;
                # the generalist spine remains the fallback when no agentic
                # scores exist, mirroring the pre-axis behavior.
                choice = dispatch_table.choose_override(
                    models,
                    self._resident_getter(),
                    residency_epsilon=cfg.residency_epsilon,
                    max_interactive_median_q_time_s=(
                        cfg.max_interactive_median_q_time_s
                    ),
                    fit_budget_gb=fit_budget_gb,
                    enabled_ids=enabled_ids,
                )
                if choice.target is not None:
                    return choice
                if generalist:
                    # Historical rule string for the no-agentic-data path so
                    # telemetry stays comparable across deploys.
                    return dispatch_table.TableChoice(
                        generalist,
                        f"override:{override}",
                        [],
                        choice.unfit,
                        choice.disabled,
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
                fit_budget_gb=fit_budget_gb,
                enabled_ids=enabled_ids,
            )
        except Exception as e:  # noqa: BLE001 - dispatch must never 5xx
            logger.warning("table dispatch failed, falling back to binary: %s", e)
            return None

    def _fail_open(
        self, reason: str, override: str | None, start: float
    ) -> RouteDecision:
        target_key = self.settings.policy.fail_open_target
        proposed = self.settings.targets.get(target_key, target_key)
        final_target, invalid = self._finalize_target(proposed)
        return RouteDecision(
            target=final_target,
            rule_fired=f"fail_open:{reason}",
            override=override,
            features=None,
            raw_analysis=None,
            classify_ms=(time.perf_counter() - start) * 1000,
            invalid_target=invalid,
        )

    def record_outcome(
        self,
        request_id: str,
        *,
        completion_tokens: int | None,
        finish_reason: str | None,
        gen_ms: float | None,
        ttft_ms: float | None = None,
        decode_ms: float | None = None,
        prompt_tokens: int | None = None,
        cached_tokens: int | None = None,
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
                # ttft/decode are stream-only (None on non-streaming paths);
                # cached_tokens distinguishes warm vs cold prefill per request.
                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "decode_ms": round(decode_ms, 1) if decode_ms is not None else None,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
            }
            self._enqueue(row)
        except Exception as e:  # noqa: BLE001 - must never raise
            logger.warning("routing record_outcome failed: %s", e)

    def record_feedback(
        self,
        request_id: str,
        *,
        score: float | None = None,
        label: str | None = None,
        tags: list[str] | None = None,
        comment: str | None = None,
        source: str = "client",
    ) -> None:
        """Attach out-of-band feedback to a prior routing decision (M6.0).

        Feedback arrives after the decision row has already flushed, so it is
        written as its own append-only record (``kind: "feedback"``) keyed by
        request_id and joined offline; the decision row is never mutated. When
        the referenced decision is still in the recent ring, the feedback is
        also attached there so the admin Router tab reflects it live.

        Sync and exception-free: an ingest endpoint must never 5xx on a store
        hiccup. No-op when telemetry is disabled (no corpus to append to).
        """
        if not self.settings.telemetry.enabled:
            return
        try:
            self._ensure_writer()
            row: dict[str, Any] = {
                "kind": "feedback",
                "ts": _now_iso(),
                "request_id": request_id,
                "score": score,
                "label": label,
                "tags": tags or None,
                "comment": comment,
                "source": source,
            }
            self._enqueue(row)
            # Best-effort live attach: newest matching decision in the ring.
            for entry in reversed(self._recent):
                if entry.get("request_id") == request_id and "kind" not in entry:
                    entry.setdefault("feedback", []).append(row)
                    break
        except Exception as e:  # noqa: BLE001 - ingest must never raise
            logger.warning("routing record_feedback failed: %s", e)

    def _maybe_implicit_feedback(self, messages: list) -> None:
        """Harvest a free outcome proxy for the prior turn's route (M6.1).

        Detects an implicit signal (tool error / correction / rephrase /
        approval) in the incoming request and, if the prior user turn hashes
        to a recent decision, records it as ``source:"implicit"`` feedback on
        that decision — reusing the M6.0 append-only feedback plumbing. Fully
        best-effort and exception-free; never on the request-serving path.
        """
        if not self.settings.implicit_feedback.enabled:
            return
        try:
            sig = detect_implicit_signal(messages)
            if sig is None:
                return
            if sig.kind == "approval" and not self.settings.implicit_feedback.approval:
                return
            # The prior decision was made on the last user message that
            # preceded the last assistant turn; hash it to find its row.
            a = _last_assistant_index(messages)
            prior_text = ""
            for msg in messages[:a]:
                if _msg_field(msg, "role") == "user":
                    prior_text = _message_text(msg)
            h = _user_text_hash(prior_text)
            if h is None:
                return
            prior_id = self._decision_by_userhash.get(h)
            if prior_id is None:
                return
            self.record_feedback(
                prior_id,
                score=sig.score,
                label=sig.kind,
                tags=["implicit", sig.marker],
                source="implicit",
            )
            # One implicit signal per decision: drop the consumed key so a
            # retried/duplicated identical turn can't re-emit against it.
            self._decision_by_userhash.pop(h, None)
        except Exception as e:  # noqa: BLE001 - must never raise
            logger.warning("routing implicit feedback failed: %s", e)

    def _index_user_hash(self, request_id: str, messages: list) -> None:
        """Record this turn's last-user-text hash for the next turn to join.

        Exception-safe: runs on every routed request when implicit feedback
        is on, so a malformed message must never break the routing contract.
        """
        try:
            h = _user_text_hash(_last_user_content(messages))
        except Exception:  # noqa: BLE001 - must never break routing
            return
        if h is None:
            return
        self._decision_by_userhash[h] = request_id
        self._decision_by_userhash.move_to_end(h)
        while len(self._decision_by_userhash) > _RECENT_MAX:
            self._decision_by_userhash.popitem(last=False)

    def _maybe_shadow_label(self, request_id: str, messages: list) -> None:
        """Fire-and-forget Apple FM second opinion for a pending row."""
        if self._shadow is None:
            return
        text = _last_user_content(messages)
        if not text and self.settings.classify_window.enabled:
            # tool_result-only agent turns have no last-user text; the
            # multi-turn window closes that shadow-label coverage gap.
            # When the last user text exists it is used as-is so labels
            # stay comparable with the pre-window corpus.
            w = self.settings.classify_window
            text = build_classify_window(
                messages, max_turns=w.max_turns, max_chars=w.max_chars
            )
        if not text:
            return
        task = asyncio.create_task(self._shadow_label(request_id, text))
        self._shadow_tasks.add(task)
        task.add_done_callback(self._shadow_tasks.discard)

    async def _shadow_label(self, request_id: str, text: str) -> None:
        rec = await self._shadow.classify(text) if self._shadow else None
        if rec is None:
            return
        row = self._pending.get(request_id)
        if row is not None:
            # If the row already flushed (fast non-streaming response beat
            # the ~0.7s fm call), the label is dropped — shadow data is
            # best-effort by design.
            row["shadow"] = rec

    @property
    def pending_count(self) -> int:
        """Number of in-flight decision rows awaiting an outcome."""
        return len(self._pending)

    def shadow_status(self) -> dict[str, Any]:
        """Shadow labeler state for the admin surface."""
        return {
            "enabled": self._shadow is not None,
            "backend": self._shadow.backend() if self._shadow else None,
        }

    def recent_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest-first recent decision rows for the admin Router tab.

        Serves from the in-memory ring buffer (rows mutate in place as
        outcomes and shadow labels land), topped up from the telemetry
        file tail so the feed survives a restart. Never raises.
        """
        limit = max(1, min(limit, _RECENT_MAX))
        rows = list(self._recent)[-limit:]
        rows.reverse()
        if len(rows) < limit:
            seen = {r.get("request_id") for r in rows}
            path = Path(self.settings.telemetry.path).expanduser()
            for row in read_telemetry_tail(path, limit):
                # Feedback rows are their own records, not decisions to display;
                # they join to decisions offline (join_feedback). Live feedback
                # is already attached to ring entries by record_feedback.
                if row.get("kind") == "feedback":
                    continue
                if row.get("request_id") in seen:
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows

    async def close(self) -> None:
        """Flush pending telemetry (rows without outcomes flush with outcome=null)."""
        for task in list(self._shadow_tasks):
            task.cancel()
        for row in list(self._pending.values()):
            self._enqueue(row)
        self._pending.clear()
        if self._queue is not None:
            await self._queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None

    def _record_decision(
        self,
        decision: RouteDecision,
        request_id: str,
        endpoint: str,
        stream: bool,
        messages: list | None = None,
    ) -> None:
        if not self.settings.telemetry.enabled:
            return
        self._ensure_writer()
        self._flush_orphans()
        if messages is not None:
            if self.settings.implicit_feedback.enabled:
                # Attribute a prior-turn implicit signal before indexing this
                # turn, so this request's own hash can't shadow the lookup.
                self._maybe_implicit_feedback(messages)
                self._index_user_hash(request_id, messages)
            self._maybe_shadow_label(request_id, messages)
        row: dict[str, Any] = {
            "ts": _now_iso(),
            "request_id": request_id,
            "endpoint": endpoint,
            "stream": stream,
            "override": decision.override,
            "raw_analysis": decision.raw_analysis,
            "features": asdict(decision.features) if decision.features else None,
            "rule_fired": decision.rule_fired,
            "target": decision.target,
            "invalid_target": decision.invalid_target,
            "classify_ms": round(decision.classify_ms, 1),
            "candidates_considered": (
                [[m, round(s, 4)] for m, s in decision.candidates]
                if decision.candidates
                else None
            ),
            "unfit": decision.unfit if decision.unfit else None,
            "disabled": decision.disabled if decision.disabled else None,
            "shadow": None,
            "outcome": None,
        }
        self._pending[request_id] = row
        self._recent.append(row)

    def _flush_orphans(self) -> None:
        """Move pending rows older than _ORPHAN_MAX_AGE_S to the write queue.

        A 507'd or disconnected request never calls record_outcome, so its
        row would otherwise sit in `_pending` until shutdown. Cheap: only
        scans the (small) in-memory pending dict, no timers.
        """
        now = datetime.now(timezone.utc)  # noqa: UP017 - mypy targets py310
        stale_ids = []
        for request_id, row in self._pending.items():
            try:
                row_age_s = (now - datetime.fromisoformat(row["ts"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if row_age_s > _ORPHAN_MAX_AGE_S:
                stale_ids.append(request_id)
        for request_id in stale_ids:
            self._enqueue(self._pending.pop(request_id))

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


def join_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach feedback rows to their decision rows by request_id (M6.0/M6.2).

    Given a mixed list of decision and ``kind:"feedback"`` rows (e.g. from
    read_telemetry_tail over the whole file), returns the decision rows with a
    ``feedback`` list populated from any matching feedback records. Feedback
    with no matching decision in the window is dropped (it joins in a wider
    read). Pure; the offline corpus join that M6.2 consumes.
    """
    feedback_by_id: dict[str | None, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") == "feedback":
            feedback_by_id.setdefault(row.get("request_id"), []).append(row)
        else:
            decisions.append(row)
    for d in decisions:
        fb = feedback_by_id.get(d.get("request_id"))
        if fb:
            d["feedback"] = fb
    return decisions


def read_telemetry_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    """Parse the last `limit` rows of a routing telemetry jsonl, newest first.

    Reads the whole file line-by-line (admin-surface only, never on the
    request path). Returns [] on any I/O problem; skips malformed lines.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines: deque[str] = deque(f, maxlen=limit)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - mypy targets py310
