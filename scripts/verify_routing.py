# SPDX-License-Identifier: Apache-2.0
"""
Live verification for semantic routing (M1, plan task 1.8).

Sends labeled prompts through the virtual model id and checks:
  - every response succeeds and carries an x-omlx-route header
  - the routed target distribution is sane (both targets exercised,
    escalation rate printed for comparison against probe expectations)
  - agentic override fires when tools are present
  - concrete model ids do NOT get a route header (bypass intact)

Usage:
    python scripts/verify_routing.py --base-url http://localhost:8888 \
        [--virtual-id auto] [--telemetry ~/.omlx/routing_decisions.jsonl]

The server must be running with routing.enabled=true. Exits non-zero on
any hard failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# (prompt, expect_big) — expect_big mirrors the spike's "hard" label
# (probe_router.py LABELED). Distribution is advisory, not a hard assert:
# policy thresholds are intentionally one notch above Supra's own rule.
LABELED: list[tuple[str, bool]] = [
    ("What year did the Berlin Wall fall?", False),
    ("Write a haiku about the moon.", False),
    ("Fix my SQL query: SELECT * FROM users WHERE (name = 'a'", False),
    ("Implement Dijkstra's algorithm in Python with a binary heap.", False),
    ("Prove the Cauchy-Schwarz inequality for inner product spaces.", True),
    ("What is 15% of 240?", False),
    ("Write a bash script that renames all .txt files to .md.", False),
    ("Explain quantum entanglement to a 10-year-old.", False),
    (
        "Design a distributed rate limiter handling 1M QPS across regions "
        "with clock skew.",
        True,
    ),
    ("Draft a polite email declining a meeting invitation.", False),
    ("Solve the differential equation y'' + 4y' + 4y = e^(-2x).", True),
    ("What's a good name for a coffee shop?", False),
    ("Implement a lock-free MPMC queue in C++ with correct memory ordering.", True),
    ("How many r's are in strawberry?", False),
    (
        "Derive the closed form of the Fibonacci sequence using generating "
        "functions.",
        True,
    ),
    ("What's the difference between TCP and UDP?", False),
    ("Calculate the tip on a $85 dinner bill at 20%.", False),
    ("Architect a CRDT-based collaborative text editor with offline sync.", True),
    ("Movie script about a mathematician who falls in love.", False),
    ("What should I cook with chicken, rice, and broccoli?", False),
]

TOOLS_FIXTURE = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def chat(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    tools: list | None = None,
) -> httpx.Response:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    return client.post(f"{base_url}/v1/chat/completions", json=body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8888")
    ap.add_argument("--virtual-id", default="auto")
    ap.add_argument("--telemetry", default="~/.omlx/routing_decisions.jsonl")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    failures: list[str] = []
    routes: list[tuple[str, str, bool]] = []  # (prompt, target, expect_big)

    with httpx.Client(timeout=args.timeout) as client:
        # 0. Virtual model discoverable
        r = client.get(f"{args.base_url}/v1/models")
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
        if args.virtual_id not in ids:
            failures.append(f"virtual id {args.virtual_id!r} missing from /v1/models")
        print(f"/v1/models contains {args.virtual_id!r}: {args.virtual_id in ids}")

        # 1. Labeled prompts through the router
        t0 = time.perf_counter()
        for prompt, expect_big in LABELED:
            r = chat(client, args.base_url, args.virtual_id, prompt)
            header = r.headers.get("x-omlx-route")
            if r.status_code != 200:
                failures.append(f"HTTP {r.status_code} for {prompt[:40]!r}")
                continue
            if not header:
                failures.append(f"missing x-omlx-route for {prompt[:40]!r}")
                continue
            target = header.split(";")[0].strip()
            routes.append((prompt, target, expect_big))
            print(f"  [{target:<40}] {prompt[:60]}")
        elapsed = time.perf_counter() - t0

        # 2. Tools override
        r = chat(
            client,
            args.base_url,
            args.virtual_id,
            "What's the weather in Paris?",
            tools=TOOLS_FIXTURE,
        )
        hdr = r.headers.get("x-omlx-route", "")
        if "override:tools" not in hdr:
            failures.append(f"tools override did not fire (header: {hdr!r})")
        print(f"tools override header: {hdr!r}")

        # 3. Concrete model id bypasses routing
        if routes:
            concrete = routes[0][1]
            r = chat(client, args.base_url, concrete, "Say hi.")
            if r.headers.get("x-omlx-route"):
                failures.append("concrete model id got an x-omlx-route header")
            print(f"concrete id {concrete!r} bypass ok: "
                  f"{'x-omlx-route' not in r.headers}")

    # 4. Distribution summary (advisory)
    if routes:
        targets = sorted({t for _, t, _ in routes})
        print(f"\n{len(routes)} routed in {elapsed:.1f}s; targets used: {targets}")
        for tgt in targets:
            n = sum(1 for _, t, _ in routes if t == tgt)
            print(f"  {tgt}: {n}/{len(routes)}")
        if len(targets) < 2:
            print("  WARNING: only one target exercised — check policy thresholds")
        agree = sum(
            1
            for _, t, big in routes
            if (t == targets[-1]) == big or len(targets) == 1
        )
        print(f"  label agreement (advisory): {agree}/{len(routes)}")

    # 5. Telemetry rows
    tpath = Path(args.telemetry).expanduser()
    if tpath.exists():
        rows = [json.loads(x) for x in tpath.read_text().splitlines() if x.strip()]
        recent = rows[-len(LABELED) :]
        missing = [k for k in ("features", "rule_fired", "target", "classify_ms")
                   if recent and k not in recent[-1]]
        print(f"telemetry: {len(rows)} rows total; last row missing keys: {missing}")
        if missing:
            failures.append(f"telemetry rows incomplete: missing {missing}")
    else:
        failures.append(f"telemetry file not found at {tpath}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll routing verifications passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
