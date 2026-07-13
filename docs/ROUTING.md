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
- **M4.1 — shape-based modality pre-route.** Requests carrying image parts route to a vision target (audio parts to an audio target) before the text classifier runs. Part types are classified explicitly: `tool_use`/`tool_result`/`thinking` blocks are text-flow, not modality signals — the original any-non-text check routed 95% of a real Claude Code session to the vision target (2026-07-11 experiment) before the part-type fix.

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
       2. image/audio parts present? → targets.vision / targets.audio (shape rule,
          precedes everything; tool_use/tool_result/thinking are text-flow, and a
          missing modality target logs a warning and falls through)
       3. tools present / user turns > N? → agentic override: table dispatch on
          the measured "agentic" axis (toolcall bench; same health/fit/
          enable_routing/interactive gates, residency-then-latency tiebreak);
          no agentic scores → default_target/big as before
       4. classify via pinned router, greedy, timeout 3s
            input = last user text, or a bounded multi-turn window when
            classify_window.enabled (recent user/assistant text, tool
            payloads excluded); any failure → fail_open_target
       5. table_dispatch on + table has data? → N-way choose(); else binary policy.decide()
       6. request.model rewritten in place; downstream resolve/settings/engine re-read it
       7. x-omlx-route header on both stream + non-stream paths
       8. telemetry row appended; outcome attached post-response
          (gen_ms always; ttft_ms/decode_ms streaming-only; plus
          prompt_tokens/cached_tokens — cached_tokens is the warm-vs-cold
          prefill signal for route-flip cost analysis)
```

Rewriting `request.model` in place is sufficient — model resolution, per-model settings, and engine acquisition all re-read it downstream. Streaming needs nothing special: the rewrite happens pre-dispatch and every target is local.

### Suitability store

Persistent JSON (`~/.omlx/suitability.json`), versioned, atomic writes. Per model: role, size, health, per-axis category scores, an append-only list of eval provenance records (bench, n, date, baseline flag, timings), and perf. Key rules:

- **Category scores derive from baseline evals only.** A non-baseline (custom-settings) run is stored as provenance but never feeds a score.
- **Largest sample size is authoritative per bench**; freshness only breaks ties. A quick n=4 spot-check must not displace an n=100 run. (Model snapshots on disk are immutable, so a newer tiny run carries no extra information — learned the hard way when a UI-test n=4 run briefly displaced an n=12 score.)
- **Role taxonomy**: chat / draft_companion / embedding / reranker / router. Name-pattern first (dflash/-assistant/draft/embed/rerank/router), then a size heuristic (<5 GB ⇒ companion). "mtp" is deliberately not a name pattern: full chat models ship with preserved MTP heads (e.g. `Qwen3.6-27B-*-mtp`); extracted drafter heads are caught by the size gate. Chat models are ≥5 GB; smaller things are spec-decode companions and are **excluded from standalone suitability evals** (benching a draft model standalone is a category error). User role overrides win and persist.
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
6. **Shape rule → agentic overrides → profiler.** Image/audio parts route to a modality target before anything else (decision #11's layering); tools/multi-turn dispatch on the measured agentic axis (falling back to the generalist spine when no agentic scores exist); only then does semantic classification run. The shape rule keys on explicit part types (image/image_url/document/file → vision, input_audio → audio); agent control-flow blocks (`tool_use`/`tool_result`/`thinking`) and unknown part types fail open to the rest of the chain — treating them as media routed 95% of real agent traffic to the vision model before the 2026-07-12 fix. tool_result *nested* content is scanned too: a screenshot returned by a browser tool still needs a VLM.
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
  "profiler_kind": "generative",           // "generative" (Supra) | "capability" (M4.5 ModernBERT)
  "capability_threshold": 0.5,             // capability-kind only: sigmoid cutoff for math/code
  "classify_timeout_s": 3.0,
  "classify_window": {                     // loop-state phase C; OFF = classify last user text only
    "enabled": false,
    "max_turns": 6,                        // user/assistant text messages considered, newest first
    "max_chars": 4000                      // total window budget (each message elided head-500/tail-300)
  },
  "targets": {
    "small":  "<a fast, cheap chat model>",
    "big":    "<the local frontier / fail-open target>",
    "vision": "<a VLM; optional — shape rule needs it to route images>",
    "audio":  "<an audio-capable model; optional — shape rule for input_audio parts>"
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
    "residency_epsilon": 0.02,            // prefer a resident model within this score margin; if none is resident, the fastest wins the tie (median_q_time_s, then load_s)
    "max_interactive_median_q_time_s": 30.0  // thinking-lane exclusion threshold
  },
  "idle_sweep": {                         // M4.4 passive sweeps; OFF by default
    "enabled": false,
    "idle_after_s": 600,                  // quiet time before a sweep may start
    "poll_interval_s": 30,                // how often the loop re-checks idleness
    "benchmarks": { "mmlu_pro": 30, "livecodebench": 10 }  // gap-fill (only_missing)
  }
}
```

Enable order in practice: turn on `routing` (binary) first, run a roster sweep to populate the table, verify it, then flip `table_dispatch.enabled`.

### Router admin tab (full config + live activity)

The admin UI has a dedicated **Router** tab (loop-state phase D, 2026-07-12)
— the old Global Settings *Semantic Routing* panel is gone and points here.

- **Configuration.** The *entire* `routing` settings block is editable: core
  (`enabled`, `virtual_model_id`, `router_model`, `profiler_kind`,
  `capability_threshold`, `classify_timeout_s`), all four `targets`
  (`small`/`big`/`vision`/`audio`), `policy` incl. `agentic_override`,
  `table_dispatch` (incl. `residency_epsilon`,
  `max_interactive_median_q_time_s`), `telemetry`, `shadow_labeler`, and
  `idle_sweep`. Saved as one nested dict via `POST
  /admin/api/routing/settings`, which replaces the whole block
  (`RoutingSettings.from_dict` → `validate()` → `save()`, with rollback on
  failure; empty-string target entries are dropped). Everything is badged
  **restart-required** — the `RoutingService` is built once at startup.
- **Activity.** `GET /admin/api/routing/activity?limit=N` (clamped to 256)
  returns the config snapshot, newest-first recent decision rows, in-flight
  pending count, and shadow-labeler status. Rows come from a 256-row
  in-memory ring buffer in `RoutingService` (`_recent` shares row dicts
  with `_pending`, so outcomes and shadow labels fill in live), topped up
  from the telemetry jsonl tail after a restart (`read_telemetry_tail`).
  The endpoint also works with routing disabled (file tail only), so past
  activity stays visible. Window stats — rule/target distributions,
  override share, target flips, median classify/TTFT/cache-hit, shadow ×
  target matrix — are computed client-side over the loaded window.
- **Log filter.** The Logs tab has a client-side *Router only* toggle
  keeping lines from `omlx.routing.*` loggers or messages carrying
  `Routing:` / `shadow labeler`.

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
- **A 507'd routed request leaves a telemetry row pending** (decision recorded, no outcome) until the orphan flush reaps it: `_flush_orphans()` runs on every new decision and flushes rows older than 10 minutes with `outcome: null`.
- **Tables don't travel between machines.** Copying `suitability.json` across hosts imports wrong timings and possibly wrong fit. Sweep each host.
- **Config loads once at startup.** Editing `routing` in settings.json requires a server restart to take effect — including everything in the admin Router tab's configuration form (it persists immediately but the service is built at boot). The **exception is `enable_routing`**: the per-model gate is read live per request, so toggling it in Model Settings takes effect on the next routed request with no restart.
- **The gate is silent when unused.** If you enable `table_dispatch` but forget to opt any model in with `enable_routing`, nothing changes — the empty enabled set makes the gate inert (fail-open), *not* a collapse to `default_target`. To see which models a decision skipped for being un-opted-in, read the `disabled` field in `routing_decisions.jsonl`.
- **Idle sweeps preempt on the hot path (M4.4).** When `idle_sweep.enabled`, `engine_pool.get_engine()` awaits a preemptor on every *real* request (bench loads pass `stamp_activity=False` and skip it). The preemptor is a no-op unless a passive sweep is live, in which case it `cancel_queue()`s and awaits the run's teardown before the request loads its model — so an interrupting request eats one bench-abort + evict + load (a few seconds) but is never stranded. A *user-initiated* sweep is never preempted (only the passive-sweep tag is). Because this touches the serving path, **validate on a non-production instance before enabling on a busy one**: turn it on, drive traffic mid-sweep, confirm the request is served and the sweep resumes next idle window. The preemption race can't be unit-tested — only the predicate, the tag lifecycle, and the teardown wait are.
- **A passive sweep re-arms its idle clock.** After a sweep drains (or is preempted), the loop resets `_last_request_monotonic` to now, so the next sweep waits a full `idle_after_s` rather than spinning no-op gap-fills every `poll_interval_s`.

## Roadmap

Done: M1, M2, M3, M4.1, **M4.2** (Anthropic `/v1/messages` hook, commit
`8e62609`), plus fit-aware dispatch, telemetry orphan-flush, and gap-fill
sweeps (commit `140b69a`). **Routing admin UI** (Global Settings routing panel
+ per-model `enable_routing` gate) landed alongside this doc revision.
**Agentic-axis override dispatch** (loop-state phase A): tool/turn overrides
rank the eligible pool on the measured agentic axis instead of collapsing
onto `default_target`, with a latency tiebreak for cold near-ties (all
axes): lowest `median_q_time_s`, then lowest `load_s` — per-turn latency
recurs every request while a load is paid once, and residency makes the
cold pick sticky for the whole conversation.
**Apple FM shadow labeler** (`routing.shadow_labeler`, off by default,
macOS 26+): after each routing decision, an async on-device Foundation
Models call (greedy, enum-constrained schema, head-500/tail-300 payload
elision) attaches a TRIVIAL/SIMPLE/MODERATE/COMPLEX second opinion to the
telemetry row (`shadow` field, with `backend: "sdk"|"cli"`). Prefers the
`apple-fm-sdk` Python bindings when importable (typed errors; latency
benchmarked equal to the CLI at ~0.65s/classify), falls back to the `fm`
CLI; classifies serialize through a lock to stay clear of system-model
concurrency limits. Never on
the request path; fail-silent on missing binary, timeout, or refusal;
labels on fast non-streaming responses may be dropped (row already
flushed). Purpose: an independent labeled corpus for the M6 outcome loop
and continuous validation of the in-process profiler. Turns whose last
user message has no text (tool_result-only agent turns) are skipped
unless the classify window (below) is enabled, which closes that
coverage gap; when last-user text exists it is labeled as-is so labels
stay comparable with the pre-window corpus.
**Multi-turn classify window** (loop-state phase C,
`routing.classify_window`, off by default): when enabled, the profiler
classifies a bounded transcript of recent user/assistant text — newest
`max_turns` messages within `max_chars`, chronological, `User:`/
`Assistant:` prefixes, each message elided head-500/tail-300 — instead
of only the last user message. Tool payloads (`tool_use`/`tool_result`),
thinking blocks, system prompts, and `role:"tool"` messages never enter
the window. Follow-ups like "make it faster" and tool_result-only agent
turns classify on real context; prerequisite for ever relaxing
`agentic_override.on_tools`. Off = profiler input byte-identical to the
pre-window behavior.
Measured 2026-07-12 (probe runs 5–6): with the override still on, the
window is a pure win (routing unchanged, shadow coverage 10/34 → 34/34).
But actually relaxing `on_tools` with the window driving was a decisive
negative: once code enters the transcript, *every* turn — even "are you
sure?" — classifies coding≈0.98 → complexity 5, so agent traffic
collapsed onto the escalate-axis leader (slow interactively; TTFT p90
52s vs 5.6s) with zero small-tier recovery. Lesson: **the window helps
the axis but poisons the tier** — the complexity proxy rates the
conversation, not the turn. Do not relax `on_tools` until (a) tier is
computed from the newest user text while axis/domain keep the window,
and (b) axis choice carries an interactive-latency term (an axis winner
never enters the med-q tiebreak) — (b) is now specced as **M8** below.
**Router admin tab** (loop-state phase D): dedicated dashboard tab with
the full routing config surface, a live decision feed + window stats fed
by a 256-row in-memory ring buffer (+ jsonl tail after restart), and a
*Router only* log-viewer filter. Absorbs the older route-explain idea —
the decision feed's expandable detail view shows `raw_analysis`,
`candidates_considered`, `unfit`, `disabled`, and the shadow reason per
row. See "Router admin tab" under Configuration.

## M6 — outcome loop

**M6.0 — `/v1/feedback` ingest.** POST `/v1/feedback` (`score` ∈ [0,1],
`label`, `tags`, `comment`, `source`) records out-of-band feedback as an
append-only `kind:"feedback"` telemetry row keyed by request_id; routed
responses echo the join key in an `x-omlx-request-id` header (both
protocols, stream + JSON). The decision row is never mutated — feedback
joins to it offline via `join_feedback` (and live in the ring buffer when
the decision is still recent). Off-path and exception-free: ingest never
5xx's on a store hiccup.

**M6.1 — implicit outcome proxies** (`routing.implicit_feedback`, off by
default). Every request already carries the full conversation history, so
when a new turn arrives the previous turn's exchange rides along with it —
a *free* satisfaction signal about the route we just made, with **no
persistent conversation identity required** (which is what blocks phase B).
`detect_implicit_signal` reads the tail of the incoming messages and emits
one of: `tool_error` (a `tool_result` with `is_error` after an assistant
turn → score 0.0, the strongest signal), `negation` ("no, that's wrong" →
0.0), `rephrase` ("try again" → 0.2), or `approval` ("thanks, that works"
→ 1.0; suppressible via `implicit_feedback.approval`). Cues are
high-precision / low-recall by design and matched only against the leading
span of the newest user message — a false positive poisons the corpus
worse than a miss. Attribution uses a bounded content-hash index
(`hash(last_user_text) → request_id`): the new turn's *prior* user message
hashes to the decision that routed it, so the signal lands as a normal
`source:"implicit"` feedback row on the right decision — reusing all of the
M6.0 plumbing. Best-effort and off the request-serving path (cheap string
matching only). Purpose: feed M6.2's read-only misroute-rate measurement
without asking any client to send feedback. M6.2's outcome data also
gates **M7 — conversation stickiness** (specced below): its held-vs-fresh
segmentation is how we'll know whether holding a route ever hurts quality.

**M6.2 — misroute measurement (read-only).** A *misroute* is a decision
whose routed target demonstrably hurt the user: a worse answer than a
roster sibling would have given (under-route), or the same answer at
needless latency (over-route). No single telemetry field proves that, so
M6.2 triangulates three planes, each reported under its own evidentiary
weight — never blended into one opaque score:

1. **Direct — joined feedback.** `join_feedback` over the full
   `routing_decisions.jsonl`; a decision is *negative* if its minimum
   joined score < 0.5. Negative rate per `rule_fired` and per `target`,
   compared against the corpus baseline. This is the only plane that
   measures misrouting itself; it is sparse until implicit feedback
   accrues, and that sparseness is reported (joined-n), not hidden.
2. **Proxy — shadow tier disagreement.** Shadow label → expected
   complexity (TRIVIAL→1, SIMPLE→2, MODERATE→3, COMPLEX→5); signed gap =
   profiler `complexity` − expected; |gap| ≥ 2 counts as disagreement,
   split into *over* (router rated harder — latency waste risk) and
   *under* (router rated easier — quality risk). This is a
   classification-quality proxy, not direct misroute evidence: the
   labeler rates the request, and override paths pick targets without
   consulting complexity at all. Reported with the full label×complexity
   matrix so the directionality is inspectable.
3. **Cost — latency paid on over-routes.** Median `outcome.ttft_ms` and
   `gen_ms` on shadow-TRIVIAL rows vs the corpus medians: what the
   escalation habit costs where a small model would have done.

Segmentation: by `rule_fired`, `target`, `endpoint`, and — once M7
lands — `sticky.held`; absent fields are tolerated so the report runs on
any corpus vintage. Surfaces: pure `misroute_report(rows)` in
`omlx/routing/misroute.py` (+ `python -m omlx.routing.misroute [path]`
for on-box runs), a read-only `GET /admin/api/routing/misroute` that
reads the full jsonl, and a Misroute panel in the Router admin tab.
Nothing in M6.2 changes routing behavior.

**Pre-registered M6.3 gate** (written before the first full-corpus run;
do not tune these after seeing the numbers): M6.3 — closed-loop
suitability adjustment, a one-way door — is justified only if, with
≥ 50 feedback-joined decisions, (a) some target or rule segment shows a
negative rate ≥ 2× the corpus baseline with ≥ 10 negative rows in the
segment, **or** (b) under-route disagreement covers ≥ 15% of shadowed
rows *and* those rows' joined feedback corroborates (negative rate above
baseline). Anything less: keep accruing and re-run. Meeting the gate
buys M6.3 a spec and an off-by-default flag, not an enablement.

Done in M4: **M4.3 settings-delta rescoring** (commit `0e1e5aa`), **M4.4
passive idle-time sweeps**, and **M4.5 classification-family profiler adapter**.

### M4.5 — Classification-family profiler adapter

`profiler_kind: "capability"` swaps the generative Supra profiler for a
single-forward-pass ModernBERT multi-label classifier
(`massaindustries/modernbert-capability-classifier` by default; ModernBERT-large,
apache-2.0, ~396M bf16), loaded via mlx-embeddings — the same seq-classification
path the reranker uses. It stays behind the existing `RouterProfiler` contract
(`classify(engine, text) -> (RouterFeatures, raw)`), selected by
`routing.profiler_kind`. Generative remains the default; nothing changes unless
you flip it.

Key facts:

- **Own model, not the engine pool.** The capability profiler owns its MLX model
  (lazy-loaded, warmed at startup best-effort) and ignores the passed engine —
  it's a classifier, not a chat engine, so `set_pinned` / `get_engine` don't
  apply. For this kind `router_model` is an **HF repo id or local path** (passed
  straight to `mlx_embeddings.load`), not a roster short-id.
- **Score mapping.** The 6 capability axes → `RouterFeatures`: `coding ≥ threshold`
  → `code`; `math_reasoning ≥ threshold` → `math`; argmax axis → `domain`
  (telemetry); `route_token` is always None. The model has **no complexity head**,
  so `complexity` is a 1–5 proxy from `max(coding, math_reasoning,
  planning_agentic)` — enough to drive the binary policy's escalate rules; the
  N-way table reads only `code`/`math` (axis) + `complexity` (tier). Live-verified:
  code/math/agentic prompts → complexity 5, factual → 1.
- **Gotcha (why the adapter forces `is_regression`).** mlx-embeddings applies
  **softmax** to `num_labels>1` logits, which is wrong for this model's
  *independent* per-axis sigmoid heads. The adapter sets `is_regression=True`
  post-load so `_process_outputs` returns raw logits, then applies its own
  `sigmoid`. Without this the six scores would be forced to sum to 1.
- **Model must be downloadable.** First `warm_up` (or first request) fetches the
  checkpoint from HF; pre-download for offline boxes. Failure is non-fatal —
  warmup logs and the profiler lazy-loads, or classify fails open.

## M7 — conversation stickiness (KV-affinity hysteresis) — SPEC, not implemented

Specced 2026-07-13, sequenced **after M6.2** (misroute measurement should
land first so held-vs-fresh decisions can be compared from day one).
Prompted by the qMLX write-up (mrzk.io, 2026-07-09): on the same M3-Ultra
hardware class, a warm 32k-token turn is **0.64s of prefill; cold is 88s**,
and the gap widens with depth. Routing today re-classifies every turn of
an `auto` conversation and can flip a deep conversation to a different
target with no awareness of the warm KV prefix it abandons — the new
target cold-prefills the entire history. The existing `residency_epsilon`
tiebreak only softens *resident-vs-cold-model* ties (avoids a **load**);
two resident models still flip freely, and the KV cache is per-model, so
residency stickiness does not prevent the cold-**prefill** penalty.

**Principle.** A mid-conversation flip must pay for itself. Tier
escalation can (correctness beats latency); a lateral axis flip or a
de-escalation on a deep history almost never does. Hysteresis suppresses
the flips that don't, and logs what it suppressed so M6.2 can check the
call. It is jitter damping, **not** a fix for systematic misclassification
— the classify-window lesson (window poisons the tier) must be fixed at
the classifier, not masked here.

**Incumbent recovery (no conversation identity — same stance as M6.1).**
Every request carries its history, so the previous turn's routed exchange
rides along. New bounded in-memory index in `RoutingService`, maintained
in `_record_decision` next to the M6.1 index:

    _sticky_by_key: OrderedDict[key, (target, override, request_id)]
    key = _user_text_hash(first_user_text) + ":" + _user_text_hash(last_user_text)

The **anchor hash on the first user message** is what makes this safe:
the M6.1 single-hash key would collide on high-frequency follow-ups
("thanks", "yes", "continue") — exactly the turns where stickiness fires
— and could stick conversation A to conversation B's target. Anchored,
a collision needs two conversations with identical first *and*
previous-user messages. Lookup on an incoming turn: anchor +
`hash(second-newest user text)` (turn N's newest user message is turn
N+1's second-newest). Bounded by `_RECENT_MAX`, in-memory only — a
restart loses stickiness for one turn and self-heals on the next
decision. Client-side history edits or branching simply miss the index →
today's behavior. Fail-open everywhere.

**Decision rule.** Stickiness engages only when *all* hold: the feature
is enabled; no modality shape rule fired (shape precedes everything,
decision #11); an incumbent was found; the incumbent still resolves
(`_finalize_target` validation — target ids rot on requantize); and
history size ≥ `min_history_chars` (below it, a cold fill is cheap and
the classifier should win). When engaged, the pipeline runs unchanged
(classify → table/binary → proposed target), then a switch away from the
incumbent is **allowed** only if one of:

- **Escalation** — the proposed flip is a tier-up (binary small→big, or
  an escalate rule fired / complexity ≥ `escalate_complexity_at`).
- **Fresh override** — tools/turns override fired this turn but had not
  fired on the incumbent's decision (stored `override` field makes this a
  one-field compare). A conversation *becoming* agentic justifies moving
  to the agentic-axis leader; one that already was stays put.
- **Rotted incumbent** — incumbent id no longer resolves.

Everything else — lateral axis flips at the same tier, de-escalation —
holds the incumbent. De-escalation stays suppressed by default
(`allow_deescalation: false`): the decode-speed savings of dropping to a
small model mid-deep-conversation rarely beat the cold prefill, and the
quality risk is asymmetric.

**Config** (`routing.sticky`, off by default, same dataclass pattern as
`RoutingImplicitFeedbackSettings` in `omlx/settings.py`):

```json
"sticky": {
  "enabled": false,
  "min_history_chars": 8000,
  "allow_deescalation": false
}
```

`min_history_chars` default rationale: ~8k chars ≈ 2k tokens ≈ ~3s of
cold prefill at this hardware class's short-context rates — roughly where
a flip starts being felt. Tune per host, like everything fit-related.

**Telemetry.** `rule_fired` strings are **not** touched (they are kept
comparable across deploys — same reason `override:*` kept its historical
string). `RouteDecision` gains an optional `sticky` field, present only
when stickiness engaged:

    {"held": true,  "incumbent": id, "proposed": id, "reason": "lateral" | "deescalation"}
    {"held": false, "incumbent": id, "reason": "escalation" | "override" | "invalid_incumbent"}

The held rows are the interesting product: M6.2 segments misroute rate by
`sticky.held` to answer "did holding hurt quality?" with the implicit
(M6.1) and explicit (M6.0) outcome signals. Admin decision feed shows a
chip (held → incumbent id) in the expandable detail view.

**Non-goals (v1).** No prefix-cache introspection (probing actual warmth
per candidate — v2 if the chars proxy proves too blunt); no cross-restart
persistence; no client-supplied conversation ids; no numeric score
margins (rules only — margins need calibrated scores we don't have until
M6.2 produces outcome data).

**Verification.** Unit: key computation (anchor + prev-text), index
maintenance/bounds, the allow/hold matrix (escalation, fresh override,
rotted id, lateral, de-escalation, below-threshold), fail-open on
malformed history. Live (Studio): drive a multi-turn `auto` conversation
past `min_history_chars`, confirm a lateral flip is held (chip in the
decision feed, `sticky.held` in `routing_decisions.jsonl`), then send a
clearly-escalating turn and confirm the switch goes through.

**Estimated size.** ~100 lines in `service.py` (index + `_apply_sticky`),
a settings dataclass, an admin chip, tests. No store or schema changes —
the `sticky` field is additive on the decision row.

## M8 — TTFT-aware agentic dispatch (prefill-throughput term) — SPEC, not implemented

Specced 2026-07-13. This is the "axis choice carries an interactive-latency
term" precondition from the classify-window lesson, promoted to its own
milestone by a live incident: Claude Code on `auto` (local box,
2026-07-13) routed `override:tools` → the agentic-axis leader
(gemma-4-31B), which passed every existing latency check — and then took
~60s to prefill CC's ~23.6k-token system prompt. The user aborted before
first token, twice. Routing chose a model that is interactive on the
bench and not interactive under an agent prompt.

**Why the existing gate can't catch this.** `median_q_time_s` is measured
on suitability-bench questions — short prompts, decode-dominated. It has
no prompt-length dimension, so it cannot rise for a model that decodes
fast but prefills slowly at depth. (The qMLX write-up's "throughput lie"
applied to our own signal: any latency number that stays flat as the
prompt grows cannot predict agent TTFT.) Agent traffic is exactly where
20k+-token prompts live, so the agentic axis is where the blindness costs
the most.

**Part 1 — measure prefill throughput per model (sweep side).** A
dedicated prefill probe, *not* derived from eval questions: time prefill
of **salted-unique prompts** (prefix caching must not fake the number —
same hygiene as `bench_qmlx.py`) at fixed depths, e.g. **2k / 8k / 24k
tokens**, phase-split so decode never pollutes the measurement (generate
1 token; the timing window is prefill only). Store per **model**, not per
bench — new optional field on the suitability-store model record:

    "prefill": {"2048": tps, "8192": tps, "24576": tps, "measured_at": ...}

Runs as part of a suitability sweep (and is eligible for M4.4 passive
idle sweeps); a probe is one load + three short prefills, far cheaper
than a bench. Per-host like everything else in the table — "tables don't
travel" applies doubly here, prefill speed is pure hardware.

**Part 2 — estimate TTFT at dispatch (request side).** Routing already
holds the messages, so prompt size is free: `est_tokens ≈ total_chars/4`
(precision is irrelevant — the gate distinguishes 2k from 20k, not 5%).
For each candidate:

    est_ttft_s = (load_s if not resident else 0)
               + est_tokens / prefill_tps(nearest measured depth ≥ est_tokens,
                                          else largest; linear is fine)

**Part 3 — gate with a non-emptying fallback.** New config
`table_dispatch.max_interactive_ttft_s` (default ~20s, off/None to
disable). Applied in `_Eligibility` beside the med-q gate for axis and
override dispatch. Fail-open shape: no prefill data → the gate passes
(exactly like the med-q gate's `lat is None`). **The gate must never
empty the pool**: if every otherwise-eligible candidate fails it, pick
the lowest-`est_ttft_s` candidate instead of falling to nothing — a slow
answer beats a 507 or a silent collapse to the generalist that may be
slower still. Telemetry: candidates excluded by the gate land in a new
`slow_ttft` list on the decision row (mirrors `unfit`/`disabled`), so
the admin decision feed can show why a leader was skipped.

**Composition.** M8 fixes turn one; M7 stickiness protects turns 2+ (a
held incumbent with a warm prefix pays only the delta, so `est_ttft`
deliberately applies to fresh decisions, not sticky-held ones); the
abort-drops-prefill follow-up (open hypothesis from the same incident)
stops retries from paying full price twice. All three trace to the same
cost model: prefill at depth is the dominant interactive cost on this
hardware.

**Non-goals.** No tokenizer-exact prompt counts at dispatch; no per-turn
prefill re-measurement (the probe is sweep-time); no decode-rate term
(med-q already proxies decode); no cross-host table sharing.

**Verification.** Unit: depth selection/interpolation, gate fail-open on
missing data, non-emptying fallback, `slow_ttft` telemetry. Live: on the
local box, replay the CC scenario — `auto` + tools + a ~23k-token prompt
must dispatch away from a slow-prefill leader (or hold it only if
nothing faster exists), with the skip visible in the decision feed.

## M5 — dashboard polish

UI polish and improvements in the areas the routing feature has touched:
the Roster Suitability page, the Global Settings routing panel, the per-model
`enable_routing` control, and any decision-log / route-header surfacing. Catch
rough edges, empty/loading states, copy, and affordances introduced by M1–M4.
Scope grows as the routing surfaces get real operator use.

**M5.1 — Suitability page (done).** Four changes, all admin-UI only (JS +
templates + i18n; no Python):

- Suitability is no longer a top-level nav tab — it's the third sub-tab under
  **Bench** (`?tab=bench&benchTab=suitability`), alongside Performance and
  Intelligence. Moved in the desktop bench dropdown, the bench sub-tab strip,
  and the mobile menu; `DASHBOARD_MAIN_TABS` drops `suitability`,
  `DASHBOARD_BENCH_TABS` gains it, and the load/refresh lifecycle moved from the
  `mainTab` watcher into `setBenchTab` + the `value === 'bench'` branch.
- The Suitability results table gets a model-settings-style toolbar: text
  search, a role filter, a health filter, a reset, and sortable Model / Role /
  Health / Size / Load headers (on top of the existing per-axis sorts). Sort is
  generalized in `suitModelIds()` via `suitSortValue()`; unranked rows (null
  score) always sort last.
- The Sweep Configuration model picker gets its own compact toolbar: search,
  an LLM/VLM type filter, and a name/size sort with a direction toggle
  (`suitPickerModelsFiltered()`). "Select all" is additive over the current
  filtered view, not a wholesale replace.
- The three persistent cards (Config, Per-Axis Rankings, Table) are collapsible
  with the same chevron pattern as the Performance tab's Metrics card
  (`suitConfigOpen` / `suitRankingsOpen` / `suitTableOpen`).
- Draft/assistant companions (`draft_companion` role) are omitted from both the
  sweep picker and the results table. `suitIsDraftOrAssistant()` prefers the
  server-assigned role and falls back to `classify_role()`'s name/size
  heuristic when a model isn't in the table yet.

## Shelved

- **Upstream PR** — offer the virtual-id plumbing + shape rules to `jundot/omlx` (issues #193/#265 asked for `model:"auto"`), with the semantic layer as the differentiator. Cut the PR branch from `main`, not `deploy` (see CLAUDE.md branch model). The routing admin UI and `enable_routing` gate are fork-only polish, not part of a minimal upstream patch. **Held pending rigorous validation** — a comprehensive report/audit of routing correctness and quality before anything is offered upstream. Check-in scheduled 2026-07-14.

## Provenance

The original spike (probes, sweeps, evidence with file:line refs, the full execution plan) lives outside this repo in the operator's `router-gateway` working notes (`PLAN.md`, `FINDINGS.md`). This doc is the durable in-repo summary; those are the archival record.
