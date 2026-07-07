# SPDX-License-Identifier: Apache-2.0
"""
Router profiler: formats the classify prompt for the pinned router model
(Supra-Router-51M), runs it as a raw completion against an in-process
engine, and parses the analysis line into structured features.

The parser is total: any input string (empty, garbage, truncated) yields a
RouterFeatures with None/False fields rather than raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CLASSIFY_MAX_TOKENS = 96
PROMPT_TEMPLATE = "Task: {text}\nAnalysis: "

FIELD_RE = re.compile(
    r"Domain:\s*(?P<domain>[^|]+)\|\s*Complexity:\s*(?P<complexity>\d)\s*\|"
    r"\s*Math:\s*(?P<math>True|False)\s*\|\s*Code:\s*(?P<code>True|False)",
    re.IGNORECASE,
)
ROUTE_RE = re.compile(r"Route:\s*(small|big)\s*model", re.IGNORECASE)


@dataclass
class RouterFeatures:
    """Structured features parsed from a router model's analysis line."""

    domain: str | None
    complexity: int | None
    math: bool
    code: bool
    route_token: str | None  # raw Route: token if present


def parse_router_output(text: str) -> RouterFeatures:
    """Parse a router analysis line into RouterFeatures. Total: never raises."""
    if not text:
        return RouterFeatures(
            domain=None, complexity=None, math=False, code=False, route_token=None
        )

    domain: str | None = None
    complexity: int | None = None
    math = False
    code = False

    m = FIELD_RE.search(text)
    if m:
        domain = m.group("domain").strip() or None
        try:
            complexity = int(m.group("complexity"))
        except (TypeError, ValueError):
            complexity = None
        math = m.group("math").lower() == "true"
        code = m.group("code").lower() == "true"

    route_match = ROUTE_RE.search(text)
    route_token = route_match.group(1).lower() if route_match else None

    return RouterFeatures(
        domain=domain,
        complexity=complexity,
        math=math,
        code=code,
        route_token=route_token,
    )


def format_classify_prompt(text: str) -> str:
    """Build the raw completion prompt sent to the router model."""
    return PROMPT_TEMPLATE.format(text=text)


class RouterProfiler:
    """Runs one classify call against an already-resolved router engine.

    Does not import engine_pool or resolve engines itself; RoutingService
    owns engine acquisition and passes the resolved engine in.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def classify(self, engine: Any, text: str) -> tuple[RouterFeatures, str]:
        """Run a raw completion against `engine` and parse the result.

        Returns (features, raw_analysis). Raises on engine failure; the
        caller (RoutingService) is responsible for timeout/exception
        handling and fail-open behavior.
        """
        prompt = format_classify_prompt(text)
        output = await engine.generate(
            prompt=prompt,
            max_tokens=CLASSIFY_MAX_TOKENS,
            temperature=0.0,
        )
        raw_analysis = output.text.strip()
        return parse_router_output(raw_analysis), raw_analysis
