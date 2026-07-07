# Semantic Routing

_"The server that knows its own models."_

This is the design-and-operations doc for the semantic routing feature (`omlx/routing/`). It carries the context that `CLAUDE.md` deliberately keeps out of the loaded-every-session budget. Read it before changing anything under `omlx/routing/`, the suitability store, the routing admin page, or the `routing` settings block.

> **Scope note.** This doc is public (the fork is a public mirror of `jundot/omlx`). It stays engineering-only: no credentials, no homelab hostnames/IPs, no per-host pin config. Per-machine deployment specifics live in the operator's private notes, not here. Runbook commands below use `localhost`.

## What it is

oMLX gains an opt-in **virtual model** (default id `auto`). A request naming it is classified in-process by a small pinned router model and dispatched to the best local model on the roster. Everything is opt-in and off by default; naming a concrete model id bypasses routing entirely.

Three strictly separated layers:

| Layer | Job | Owner |
|---|---|---|
| **Profiler** | prompt → features (`complexity`, `math`, `code`, `domain`). Supra-Router-51M today, swappable behind an interface. | model (swappable) |
| **Suitability table** | per-model measured strengths (eval scores × speed × footprint), built by the server benchmarking its own roster. | measurement |
| **Policy** | deterministic rules joining features × table × pool state → target model. Shape/agentic overrides sit above the profiler. | config (user-ownable) |

The genuinely novel layer is the **measured** suitability table: capabilities are benchmarked, not authored and not model-inferred.

## Status (as of this writing)

All planned milestones plus the first M4 item are implemented and live-verified:

- **M1 — binary routing.** `auto` → small/big target via classify + policy. Native, in-process.
- **M2 — suitability harness.** Server benchmarks its own roster in a baseline (stock-settings) mode, persists a per-model capability table with provenance, surfaced in an admin page.
- **M3 — N-way dispatch.** Table-driven per-axis routing (code/math/knowledge), thinking-lane exclusion, escalation tier, residency-aware tiebreak. Opt-in; binary is the fallback.
- **M4.1 — shape-based vision pre-route.** Requests carrying image/non-text parts route to a vision target before the text classifier runs.

Measured on an M-series Mac: classify overhead **~70 ms p50** in-process (an HTTP sidecar prototype was ~2.15 s). With routing disabled the server is byte-for-byte unchanged.

## Architecture

### Package layout (`omlx/routing/`)

```
profiler.py   RouterProfiler: prompt format, in-process generate, total parser -> RouterFeatures
policy.py     pure decide(features, override, cfg) -> (target_key, rule_fired)
service.py    RoutingService: classify (timeout+fail-open), shape rule, overrides,
              binary vs N-way selection, jsonl decision telemetry with post-response outcomes
table.py      N-way dispatch: choose(features, store snapshot, resident ids) -> TableChoice
store.py      SuitabilityStore: persistent per-model capability table + role taxonomy
```

One-way dependency: `routing/` imports engine/pool/settings interfaces; `server.py` imports `routing/`. Nothing in `routing/` imports `server.py`. State is wired in from the lifespan via setters (`set_engine_getter`, `set_table_sources`) — the same `set_*_getters` philosophy the rest of the server uses.

The suitability harness lives under admin (`omlx/admin/suitability.py`) because it drives the existing accuracy-benchmark queue; `store.py` itself is pure and under `routing/` because dispatch consumes it.

### Request flow

```
POST /v1/chat/completions {model: "auto", ...}
  └─ hook in create_chat_completion, AFTER the oQ-quantization 503 guard
       1. model != virtual id?  → passthrough (zero cost)
       2. non-text parts present? → targets.vision (shape rule, precedes everything)
       3. tools present / user turns > N? → generalist/big (agentic override)
       4. classify(last user text) via pinned router, greedy, timeout 3s
            any failure → fail_open_target
       5. table_dispatch on + table has data? → N-way choose(); else binary policy.decide()
       6. request.model rewritten in place; downstream resolve/settings/engine re-read it
       7. x-omlx-route header on both stream + non-stream paths
       8. telemetry row appended; outcome attached post-response
```

Rewriting `request.model` in place is sufficient — model resolution, per-model settings, and engine acquisition all re-read it downstream. Streaming needs nothing special: the rewrite happens pre-dispatch and every target is local.

### Suitability store

Persistent JSON (`~/.omlx/suitability.json`), versioned, atomic writes. Per model: role, size, health, per-axis category scores, an append-only list of eval provenance records (bench, n, date, baseline flag, timings), and perf. Key rules:

- **Category scores derive from baseline evals only.** A non-baseline (custom-settings) run is stored as provenance but never feeds a score.
- **Largest sample size is authoritative per bench**; freshness only breaks ties. A quick n=4 spot-check must not displace an n=100 run. (Model snapshots on disk are immutable, so a newer tiny run carries no extra information — learned the hard way when a UI-test n=4 run briefly displaced an n=12 score.)
- **Role taxonomy**: chat / draft_companion / embedding / reranker / router. Name-pattern first (dflash/mtp/-assistant/draft/embed/rerank/router), then a size heuristic (<5 GB ⇒ companion). Chat models are ≥5 GB; smaller things are spec-decode companions and are **excluded from standalone suitability evals** (benching a draft model standalone is a category error). User role overrides win and persist.
- **Tables are per-host.** tps/load times and even fit differ across machines; the store stamps the host. Each machine runs its own sweep.

### Baseline mode

Suitability evals must measure *the model*, not someone's tuning. `ModelSettingsManager.set_baseline_ids({id})` makes `get_settings(id)` return stock defaults for the duration of a baseline run, so both sampling customizations and load-time variants (draft/MTP/KV-quant) are ignored. The bench queue evicts all models before each run, so the target always loads fresh under the bypass.

Note: the accuracy evaluator already forces `temperature=0`, `presence_penalty=0`, `repetition_penalty=1` (`omlx/eval/base.py`), so **sampling** penalties were never the live taint vector — **load-time** variants were, which is exactly what baseline mode neutralizes.

## Decisions (do not relitigate without new evidence)

1. **In-server, not a sidecar.** HTTP round-trip added ~1.2 s over in-process; residency-aware dispatch needs live pool state.
2. **Router is a pinned, permanently-resident engine, invoked in-process.** No new scheduler code — per-model engine threads + Metal streams already give concurrency. Pinning exempts it from eviction. ~100 MB bf16, negligible.
3. **bf16 router weights, never quantized.** (`Supra-Router-51M-oQ8` exists on rosters; do not use it for routing.)
4. **Parse the full analysis line; policy keys on features, not the router's own `Route:` token** — that baked-in rule is calibrated for edge SLMs and over-escalates for a capable roster. `Route:` is only the fallback when policy config is absent.
5. **Complexity is primary; math/code are modifiers; domain is telemetry-only.**
6. **Shape rule → agentic overrides → profiler.** Image/non-text parts route to a vision target before anything else (decision #11's layering); tools/multi-turn route to the generalist; only then does semantic classification run.
7. **Fail open, always.** Any classify failure (timeout, parse miss, engine gone, store/pool snapshot error) routes to a configured fail-open target. A routing bug must never 5xx a request or strand it on a weak model.
8. **Opt-in via virtual id; concrete ids bypass.** MarkItDown's virtual-model pattern is the in-repo precedent.
9. **Decision telemetry from day one, jsonl, first-class.** The labeled dataset is worth more than the router itself.
10. **Decision carried in a header (`x-omlx-route`), not body mutation.** Strict OpenAI clients validate response shape.
11. **Upstream-compatible posture.** Semantic routing layers *behind* shape rules (image→VLM etc. — upstream's planned scope for `auto`). Virtual id stays configurable so a future upstream `auto` doesn't collide; the whole feature stays a small isolated patch (rebase insurance + PR-ability).
12. **Suitability is measured, not authored and not model-inferred.** Evals bootstrap it; telemetry refines it.

## Configuration

Under `routing` in `~/.omlx/settings.json` (defaults shown; whole feature is OFF by default):

```jsonc
"routing": {
  "enabled": false,
  "virtual_model_id": "auto",
  "router_model": "Supra-Router-51M",     // pinned + preloaded when enabled
  "classify_timeout_s": 3.0,
  "targets": {
    "small":  "<a fast, cheap chat model>",
    "big":    "<the local frontier / fail-open target>",
    "vision": "<a VLM; optional — shape rule needs it to route images>"
  },
  "policy": {
    "escalate_complexity_at": 4,          // complexity >= 4 -> big
    "escalate_math_complexity_at": 3,     // math AND complexity >= 3 -> big
    "escalate_code_complexity_at": 3,     // code AND complexity >= 3 -> big
    "agentic_override": { "on_tools": true, "max_user_turns": 3 },
    "fail_open_target": "big"
  },
  "telemetry": { "enabled": true, "path": "~/.omlx/routing_decisions.jsonl" },
  "table_dispatch": {                     // M3 N-way; OFF until a sweep populates the table
    "enabled": false,
    "default_target": null,               // generalist spine; falls back to targets.big
    "residency_epsilon": 0.02,            // prefer a resident model within this score margin
    "max_interactive_median_q_time_s": 30.0  // thinking-lane exclusion threshold
  }
}
```

Enable order in practice: turn on `routing` (binary) first, run a roster sweep to populate the table, verify it, then flip `table_dispatch.enabled`.

### UI-configurable subset

The **Global Settings** admin tab has a *Semantic Routing* panel exposing the
knobs an operator flips at runtime: `enabled`, `virtual_model_id`,
`telemetry.enabled`, `table_dispatch.enabled`, `table_dispatch.default_target`,
and `targets.vision`. The panel is badged **restart-required** — the POST
persists to `settings.json` immediately but the `RoutingService` is built once
at startup, so a restart is needed to pick the changes up (`POST
/admin/api/global-settings` returns them under `restart_required`, not
`runtime_applied`). The other fields (policy thresholds, `router_model`, the
`small`/`big` targets, `residency_epsilon`) remain settings.json-only.

### Per-model routing gate (`enable_routing`)

A per-model **opt-in** flag (Model Settings → *Enable Semantic Routing*, stored
in `model_settings.json`) gates whether a model is eligible as an N-way
table-dispatch target. Key semantics:

- **Gate scope is the ranked pool only.** `table.choose()`'s `eligible()`
  predicate drops any candidate not in the enabled set (recorded in the new
  `TableChoice.disabled` / telemetry `disabled` field). Explicitly-named
  targets — `table_dispatch.default_target`, `policy.fail_open_target`,
  `targets.vision`, and the binary `small`/`big` — **bypass the gate**: naming
  a model by id in config *is* the opt-in, which keeps fail-open (decision #7)
  from ever pointing at a disabled model.
- **Empty set = inert.** If *no* model has `enable_routing=True`, the gate does
  nothing and dispatch behaves exactly as before. This makes shipping the
  feature default-off a no-op until the operator opts models in one by one —
  no silent collapse to `default_target` on the first restart.
- **Read live, baseline-independent.** The enabled set is computed per routing
  decision from `ModelSettingsManager.get_all_settings()`, which reads the
  persisted flag directly and is *not* subject to the `set_baseline_ids`
  bypass. So a model held out by an in-flight suitability sweep keeps its
  operator-set eligibility.
- **Never in a profile.** `enable_routing` is in `EXCLUDED_FROM_PROFILES`: a
  profile is a sampling variant served on the same engine, so routing
  eligibility belongs to the base model, not the profile.

Like the rest of `routing`, the gate is read fresh per request from a getter
wired via `set_table_sources(..., enabled_getter=...)`; no restart needed for
`enable_routing` edits (unlike the global panel).

## Dev runbook (localhost)

```bash
# run the server from a checkout of the deploy branch
uv run omlx serve                     # :8000 by default; port/model-dir persist to settings.json

# M1/M4.1 verification (labeled prompts, header, override, bypass, telemetry)
python scripts/verify_routing.py --base-url http://localhost:8000

# M3 verification (axis rules, leader agreement, candidates_considered telemetry)
python scripts/verify_table_dispatch.py --base-url http://localhost:8000

# watch decisions live
tail -f ~/.omlx/routing_decisions.jsonl
```

### Running a suitability sweep

The admin router is mounted under `/admin`. A sweep enqueues **baseline-mode** runs for the selected chat models through the existing accuracy-benchmark queue (which evicts all models per run — politeness is inherited). Non-chat roles are skipped server-side.

```bash
# queue a sweep (mmlu_pro + livecodebench differentiate; gsm8k/humaneval saturate on modern rosters)
curl -X POST http://localhost:8000/admin/api/suitability/sweep \
  -H 'Content-Type: application/json' \
  -d '{"models":["<id1>","<id2>"],"benchmarks":{"mmlu_pro":30,"livecodebench":10}}'

curl http://localhost:8000/admin/api/bench/accuracy/queue/status   # poll until running=false
curl http://localhost:8000/admin/api/suitability/table             # rankings + per-model records
```

Or use the **Roster Suitability** admin page (sweep launcher, live progress, ranked table, role overrides, unhealthy surfacing).

> **A sweep evicts every loaded model and holds them out for the run.** On a machine serving live traffic from pinned models, that's disruptive — the pins come back only on restart. Don't kick a large sweep on a busy instance without intending it.

### Tests

```bash
pytest tests/test_routing_*.py tests/test_suitability_orchestrator.py \
       tests/test_baseline_mode.py -q          # ~190 unit tests, no model files needed
pytest -q                                      # full fast suite
```

## Gotchas

- **A running server evicting all models mid-sweep = cancelled runs.** Restarting the server (even to pick up config) while the bench queue is active cancels the in-flight run. Check queue status before restarting. Cancellation is *not* recorded as unhealthy (by design); a real load/serve failure is.
- **Escalation 507s** mean the big/frontier target can't fit alongside what's pinned/resident. Either free room (unpin) or point `big` at a smaller resident model. Fail-open still routes there, so the 507 comes from the engine, not routing.
- **A 507'd routed request leaves a telemetry row pending** (decision recorded, no outcome) until shutdown flush. Harmless; an orphan-flush timer is a cheap future polish.
- **Tables don't travel between machines.** Copying `suitability.json` across hosts imports wrong timings and possibly wrong fit. Sweep each host.
- **Config loads once at startup.** Editing `routing` in settings.json requires a server restart to take effect — including everything in the Global Settings *Semantic Routing* panel (it persists immediately but the service is built at boot). The **exception is `enable_routing`**: the per-model gate is read live per request, so toggling it in Model Settings takes effect on the next routed request with no restart.
- **The gate is silent when unused.** If you enable `table_dispatch` but forget to opt any model in with `enable_routing`, nothing changes — the empty enabled set makes the gate inert (fail-open), *not* a collapse to `default_target`. To see which models a decision skipped for being un-opted-in, read the `disabled` field in `routing_decisions.jsonl`.

## Roadmap

Done: M1, M2, M3, M4.1, **M4.2** (Anthropic `/v1/messages` hook, commit
`8e62609`), plus fit-aware dispatch, telemetry orphan-flush, and gap-fill
sweeps (commit `140b69a`). **Routing admin UI** (Global Settings routing panel
+ per-model `enable_routing` gate) landed alongside this doc revision.

Remaining M4 (rough priority):

1. **Settings-delta rescoring** — re-run a slice with one setting changed (draft model, thinking, KV quant) and report the accuracy/speed delta per model. Turns the careless-config problem into a measured tuning assistant, and validates spec-decode losslessness per model.
2. **Passive idle-time sweeps** — the hardest 20% (idle detection + interruption semantics).
3. **Classification-family profiler adapter** — single-forward-pass BERT-style routers behind the existing profiler interface.

## M5 — dashboard polish

UI polish and improvements in the areas the routing feature has touched:
the Roster Suitability page, the Global Settings routing panel, the per-model
`enable_routing` control, and any decision-log / route-header surfacing. Catch
rough edges, empty/loading states, copy, and affordances introduced by M1–M4.
Scope grows as the routing surfaces get real operator use.

## Shelved

- **Upstream PR** — offer the virtual-id plumbing + shape rules to `jundot/omlx` (issues #193/#265 asked for `model:"auto"`), with the semantic layer as the differentiator. Cut the PR branch from `main`, not `deploy` (see CLAUDE.md branch model). The routing admin UI and `enable_routing` gate are fork-only polish, not part of a minimal upstream patch. **Held pending rigorous validation** — a comprehensive report/audit of routing correctness and quality before anything is offered upstream. Check-in scheduled 2026-07-14.

## Provenance

The original spike (probes, sweeps, evidence with file:line refs, the full execution plan) lives outside this repo in the operator's `router-gateway` working notes (`PLAN.md`, `FINDINGS.md`). This doc is the durable in-repo summary; those are the archival record.
