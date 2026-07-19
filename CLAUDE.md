# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

oMLX is an LLM inference server for Apple Silicon, built on Apple's MLX. It serves text LLMs, vision-language models (VLM), OCR, embeddings, and rerankers behind OpenAI- and Anthropic-compatible APIs, with continuous batching and a two-tier (RAM + SSD) KV cache. It ships three ways: `pip install`, Homebrew, and a native Swift menubar `.app` (the Swift shell lives in `apps/`, not this Python package).

Requires macOS 15+ and Apple Silicon. Most of the engine cannot run or be meaningfully tested on non-Apple hardware.

## Commands

```bash
pip install -e ".[dev]"          # dev setup (also: uv sync --dev)
uv tool install -e ".[mcp]"      # install the omlx CLI as a uv-managed editable tool on PATH
omlx serve                       # run server (defaults: ~/.omlx/models, port 8000)
omlx serve --model-dir /path --port 8000   # flags persist to ~/.omlx/settings.json
omlx start | stop | restart      # portable lifecycle (Homebrew delegates to brew services)

pytest                           # default: excludes slow + integration (see pytest.ini addopts)
pytest tests/test_config.py -v   # single file
pytest tests/test_config.py::test_name   # single test
pytest -m slow                   # model-loading tests (need model files on disk)
pytest -m integration            # need a running server
pytest -m turboquant             # TurboQuant KV cache suite

black . && ruff check . && mypy omlx   # format, lint, typecheck (line-length 88)
```

Admin UI is at `/admin`; OpenAI/Anthropic APIs under `/v1/*`. The server writes a structured log to `~/.omlx/logs/server.log`.

## Architecture

**Request lifecycle.** `server.py` builds the FastAPI app (`app = FastAPI(...)` near line 462) with a `lifespan` manager doing startup (alias detection, pinned-model preload, memory enforcer, optional MCP init). `init_server()` wires CORS and other middleware. Routers are included with a consistent pattern: `/v1/*` routers get a global `Depends(verify_api_key)`; the admin router does **not** (auth is per-route via `require_admin`). Each router module exposes a `set_*_getters()` the server calls after `include_router`, passing lambdas that close over `_server_state` — this keeps router modules import-free of `server.py` (one-way dependency). Follow this pattern when adding endpoints; don't reach into `_server_state` directly.

**Engine layer (`omlx/engine/`).** One class per modality, all over `base.py`: `batched.py` (text LLM continuous batching via mlx-lm's BatchGenerator), `vlm.py`, `embedding.py`, `reranker.py`, `dflash.py` (speculative decoding), plus audio (`tts`/`stt`/`sts`). `engine_pool.py` manages the live set of loaded models with LRU eviction, pinning, and per-model TTL; `scheduler.py` (large) owns request scheduling and batching decisions. `process_memory_enforcer.py` caps total RSS (default system RAM − 8 GB) to prevent OOM.

**KV cache (`omlx/cache/`).** Block-based, vLLM-inspired, with prefix sharing and copy-on-write. Two tiers: hot (RAM, `hybrid_cache.py`/`paged_cache.py`) and cold (SSD safetensors, `paged_ssd_cache.py`), with `boundary_snapshot_store.py` persisting cache across restarts and `prefix_cache.py` for prefix reuse. `factory.py` assembles the stack; `interface.py` is the contract.

**API surface (`omlx/api/`).** Pydantic models + utils per protocol: `openai_models.py`, `anthropic_models.py` (+ adapters in `adapters/`), `embedding_models.py`, `rerank_models.py`, `audio_routes.py`. `tool_calling.py` parses function calls; `thinking.py` handles reasoning output; `grammar.py` does JSON-schema-constrained decoding (needs the `[grammar]` extra → xgrammar → torch).

**Two MCP directions — do not conflate.** `omlx/mcp/` is oMLX-as-**client** (its chat models call out to external stdio/SSE MCP servers; config at `~/.omlx/mcp.json` or `--mcp-config`), surfaced over REST at `omlx/api/mcp_routes.py` (`/v1/mcp/*`). Separately, `omlx/admin/static/js/webmcp/` is oMLX-as-**tool-source** (WebMCP): the admin pages register ~47 in-page tools via `navigator.modelContext` for browser-use agents to call. Reversed direction; no shared code.

**Model resolution.** `model_discovery.py` scans model dirs; `model_registry.py` tracks state; `model_settings.py` + `model_profiles.py` hold per-model sampling/config and named profiles. A profile can be exposed as its own API model id (`<model>:<profile>`) served on the same engine with settings overlaid per request — no extra memory.

**Integrations (`omlx/integrations/`).** One module per agent client (Claude, Codex, Copilot, Hermes, OpenClaw, OpenCode, Pi) — generates the config that points each tool at the local server.

**Semantic routing (`omlx/routing/`, this fork's flagship feature).** An opt-in virtual model id (`auto`) classifies each request in-process via a pinned Supra-Router-51M profiler and dispatches to the best local model — binary (small/big) or N-way against a measured suitability table the server builds by benchmarking its own roster. Off by default; concrete model ids bypass it entirely; fail-open everywhere. This is the primary work happening on this fork right now. **Read [`docs/ROUTING.md`](docs/ROUTING.md) before touching anything under `omlx/routing/`, the suitability store, or the routing config** — it carries the design decisions, deployment state, ops runbook, and gotchas that this file deliberately keeps out of the loaded-every-session context.

## Admin UI conventions

See `omlx/admin/CLAUDE.md` (auto-loads when working under `omlx/admin/`): server-rendered Jinja + Alpine.js, no build step; CDN deps are vendored offline via `vendor_deps.py`.

## Dependency pins

`mlx-lm`, `mlx-vlm`, `mlx-embeddings`, `mlx-audio`, and `dflash-mlx` are pinned to **specific git commits** in `pyproject.toml`, with `[tool.uv] override-dependencies` forcing the resolver to accept the git `mlx-lm` over transitive pins. The comments above each pin explain why — read them before bumping. `torch` is intentionally absent from the core install (mlx-vlm uses custom processors); only the `[grammar]` extra pulls it in via xgrammar.

## Packaging

See `packaging/CLAUDE.md` (auto-loads when working under `packaging/`): `build.py` builds the embedded Python layers for the Swift `.app`; the `[bundle]` extra is its source of truth.

## Conventions

- License header on new source files: `# SPDX-License-Identifier: Apache-2.0`
- Test naming: `omlx/<module>.py` → `tests/test_<module>.py`

## Merge-seam rules (keep upstream files clean)

Every fork line inside an upstream-owned file is future merge-conflict surface. Keep the contact points few and small:

- **Never run `black .` / `ruff format` repo-wide.** Upstream is not black-clean; a repo-wide run reformats upstream files and manufactures conflicts with zero behavior change. Format only files this fork owns.
- **i18n:** fork strings go in `omlx/admin/i18n/fork.<lang>.json`, never in the upstream `<lang>.json`. `_load_locale()` overlays them. The 9 upstream locale files must stay byte-identical to `main`.
- **Tests:** fork tests go in fork-named files (`tests/test_routing_*.py`, `tests/test_ssd_janitor.py`, …), never appended to an upstream test file. Reuse upstream fixtures/helpers by import rather than editing their file.
- **Don't change an upstream function's signature or pass it new kwargs** — wrap it in a fork-owned function instead. Kwarg additions break upstream's own test stubs silently, with no conflict marker (see `de089c1f`).

## Branch model (upstream-tracking fork)

`origin` is the fork (`jasonpaulso/omlx`, formerly `omlx-plus`); `upstream` is `jundot/omlx`. The layout keeps `main` clean so features can be PR'd upstream:

- **`main`** is a pristine mirror of `upstream/main`. Never commit here; only fast-forward it from upstream (`git fetch upstream && git push origin upstream/main:main`). It carries no CLAUDE.md and none of this fork's features.
- **`deploy`** is the long-lived integration branch — everything this fork runs: upstream + admin features (model visibility/search) + semantic routing + these docs. Both homelab instances track it. Bring upstream in with `git checkout deploy && git merge main` (merge, not rebase — it's a deployed branch, force-pushing it would force every instance to hard-reset).
- **Feature branches for upstream PRs are cut from `main`**, not `deploy` — e.g. `feat/semantic-routing`. This is what keeps `CLAUDE.md` and `docs/ROUTING.md` out of any upstream PR: they live only on `deploy`, and `main` (the base of every upstream-bound branch) never has them. Don't cut an upstream-bound branch from `deploy`.

So: hack on `deploy` day to day; when a feature is ready to offer upstream, replay its commits onto a fresh branch off `main`.
