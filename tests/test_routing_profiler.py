# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.routing.profiler: prompt format and the total parser.

REAL_FIXTURES below are the raw "text" analysis lines captured from live
Supra-Router-51M probes (see router-gateway/spike-data/probe_results.jsonl),
covering baseline, accuracy (including the 9 label-mismatch outputs), and
multi-turn probes. The expected tuple for each was derived from that same
probe run's FIELD_RE/ROUTE_RE parse (probe_router.py), which profiler.py's
regexes are copied from.
"""

import pytest

from omlx.routing.profiler import (
    RouterFeatures,
    RouterProfiler,
    format_classify_prompt,
    parse_router_output,
)


class _FakeGenerationOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeEngine:
    """Stands in for a BatchedEngine; records the last generate() call."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_kwargs: dict | None = None

    async def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeGenerationOutput(self.response_text)


# (raw_text, (expected_domain, expected_complexity, expected_math,
#             expected_code, expected_route_token))
REAL_FIXTURES = [
    (
        "Domain: Geography | Complexity: 1 | Math: False | Code: False | "
        "Route: small model | Justification: Automated override: Simple "
        "text or factual task with low complexity (1), perfectly "
        "optimized for an edge SLM.",
        ("Geography", 1, False, False, "small"),
    ),
    (
        "Domain: Poetry | Complexity: 2 | Math: False | Code: False | "
        "Route: small model | Justification: Automated override: Simple "
        "text or factual task with low complexity (2), perfectly "
        "optimized for an edge SLM.",
        ("Poetry", 2, False, False, "small"),
    ),
    (
        "Domain: Programming | Complexity: 3 | Math: False | Code: True | "
        "Route: big model | Justification: Automated override: Task "
        "complexity is high (3) or involves technical logic (Math: False, "
        "Code: True), requiring frontier capabilities.",
        ("Programming", 3, False, True, "big"),
    ),
    (
        "Domain: Algebra | Complexity: 3 | Math: True | Code: False | "
        "Route: big model | Justification: Automated override: Task "
        "complexity is high (3) or involves technical logic (Math: True, "
        "Code: False), requiring frontier capabilities.",
        ("Algebra", 3, True, False, "big"),
    ),
    (
        "Domain: Database Design | Complexity: 3 | Math: False | "
        "Code: False | Route: big model | Justification: Automated "
        "override: Task complexity is high (3) or involves technical "
        "logic (Math: False, Code: False), requiring frontier "
        "capabilities.",
        ("Database Design", 3, False, False, "big"),
    ),
    (
        "Domain: Mathematics | Complexity: 2 | Math: True | Code: False | "
        "Route: big model | Justification: Automated override: Task "
        "complexity is high (2) or involves technical logic (Math: True, "
        "Code: False), requiring frontier capabilities.",
        ("Mathematics", 2, True, False, "big"),
    ),
    (
        "Domain: Linear Algebra | Complexity: 3 | Math: True | Code: True "
        "| Route: big model | Justification: Automated override: Task "
        "complexity is high (3) or involves technical logic (Math: True, "
        "Code: True), requiring frontier capabilities.",
        ("Linear Algebra", 3, True, True, "big"),
    ),
    (
        "Domain: Theoretical Computer Science | Complexity: 4 | "
        "Math: True | Code: False | Route: big model | Justification: "
        "Automated override: Task complexity is high (4) or involves "
        "technical logic (Math: True, Code: False), requiring frontier "
        "capabilities.",
        ("Theoretical Computer Science", 4, True, False, "big"),
    ),
    (
        "Domain: strawberry | Complexity: 1 | Math: False | Code: False | "
        "Route: small model | Justification: Automated override: Simple "
        "text or factual task with low complexity (1), perfectly "
        "optimized for an edge SLM.",
        ("strawberry", 1, False, False, "small"),
    ),
    (
        # accuracy-probe label mismatch: SQL fix mislabeled non-code/simple
        "Domain: SQL | Complexity: 2 | Math: False | Code: False | "
        "Route: small model | Justification: Automated override: Simple "
        "text or factual task with low complexity (2), perfectly "
        "optimized for an edge SLM.",
        ("SQL", 2, False, False, "small"),
    ),
    (
        # accuracy-probe label mismatch: Cauchy-Schwarz mislabeled non-math
        "Domain: Product Design | Complexity: 3 | Math: False | "
        "Code: False | Route: big model | Justification: Automated "
        "override: Task complexity is high (3) or involves technical "
        "logic (Math: False, Code: False), requiring frontier "
        "capabilities.",
        ("Product Design", 3, False, False, "big"),
    ),
    (
        # accuracy-probe label mismatch: podcast-intro keyword trap
        "Domain: Programming | Complexity: 2 | Math: False | Code: True | "
        "Route: big model | Justification: Automated override: Task "
        "complexity is high (2) or involves technical logic (Math: False, "
        "Code: True), requiring frontier capabilities.",
        ("Programming", 2, False, True, "big"),
    ),
    (
        # multiturn: last-user-only mid-convo (code context)
        "Domain: logic | Complexity: 2 | Math: False | Code: False | "
        "Route: small model | Justification: Automated override: Simple "
        "text or factual task with low complexity (2), perfectly "
        "optimized for an edge SLM.",
        ("logic", 2, False, False, "small"),
    ),
    (
        # multiturn: last-user-only mid-convo (vague followup)
        "Domain: Human Resources | Complexity: 2 | Math: False | "
        "Code: False | Route: small model | Justification: Automated "
        "override: Simple text or factual task with low complexity (2), "
        "perfectly optimized for an edge SLM.",
        ("Human Resources", 2, False, False, "small"),
    ),
    (
        # multiturn: whole-convo concatenation
        "Domain: Web Development | Complexity: 2 | Math: False | "
        "Code: False | Route: small model | Justification: Automated "
        "override: Simple text or factual task with low complexity (2), "
        "perfectly optimized for an edge SLM.",
        ("Web Development", 2, False, False, "small"),
    ),
    (
        # multiturn: long agentic-style context
        "Domain: Web Development | Complexity: 3 | Math: False | "
        "Code: True | Route: big model | Justification: Automated "
        "override: Task complexity is high (3) or involves technical "
        "logic (Math: False, Code: True), requiring frontier "
        "capabilities.",
        ("Web Development", 3, False, True, "big"),
    ),
]


@pytest.mark.parametrize("text,expected", REAL_FIXTURES)
def test_parse_real_router_output(text, expected):
    domain, complexity, math, code, route_token = expected
    features = parse_router_output(text)
    assert features == RouterFeatures(
        domain=domain,
        complexity=complexity,
        math=math,
        code=code,
        route_token=route_token,
    )


MALFORMED_FIXTURES = [
    "",
    "   ",
    "garbage nonsense output with no structure at all",
    "Domain: Programming | Complexity: ",  # truncated mid-field
    "Domain: Programming | Complexity: N/A | Math: False | Code: False",
    "Route: big model",  # route only, no fields
    "Domain: Foo | Complexity: 3 | Math: Maybe | Code: False",  # bad enum
    "domain:foo|complexity:2|math:true|code:false|route:small model",
    "Domain: Foo | Complexity: 3 | Math: False | Code: False | Route: medium model",
]


@pytest.mark.parametrize("text", MALFORMED_FIXTURES)
def test_parser_never_raises_on_malformed_input(text):
    features = parse_router_output(text)
    assert isinstance(features, RouterFeatures)


def test_parser_malformed_yields_no_usable_features():
    features = parse_router_output("garbage nonsense output with no structure at all")
    assert features.domain is None
    assert features.complexity is None
    assert features.math is False
    assert features.code is False
    assert features.route_token is None


def test_parser_empty_string():
    features = parse_router_output("")
    assert features == RouterFeatures(
        domain=None, complexity=None, math=False, code=False, route_token=None
    )


def test_parser_none_like_route_without_fields():
    features = parse_router_output("Route: big model")
    assert features.domain is None
    assert features.complexity is None
    assert features.route_token == "big"


def test_format_classify_prompt():
    assert format_classify_prompt("hello") == "Task: hello\nAnalysis: "


async def test_router_profiler_classify_uses_raw_completion_engine():
    engine = _FakeEngine(
        "Domain: Geography | Complexity: 1 | Math: False | Code: False | "
        "Route: small model | Justification: trivial."
    )
    profiler = RouterProfiler("Supra-Router-51M")

    features, raw_analysis = await profiler.classify(engine, "capital of Portugal?")

    assert engine.last_kwargs["prompt"] == "Task: capital of Portugal?\nAnalysis: "
    assert engine.last_kwargs["max_tokens"] == 96
    assert engine.last_kwargs["temperature"] == 0.0
    assert raw_analysis.startswith("Domain: Geography")
    assert features == RouterFeatures(
        domain="Geography",
        complexity=1,
        math=False,
        code=False,
        route_token="small",
    )


async def test_router_profiler_classify_strips_whitespace():
    engine = _FakeEngine(
        "  Domain: Foo | Complexity: 1 | Math: False | Code: False  \n"
    )
    profiler = RouterProfiler("Supra-Router-51M")

    _, raw_analysis = await profiler.classify(engine, "text")

    assert raw_analysis == "Domain: Foo | Complexity: 1 | Math: False | Code: False"
