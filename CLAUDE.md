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
omlx start | stop | restart      # portable lifecycle (.app socket; Homebrew → brew services; source install → launchd)
omlx start --system              # source install: system LaunchDaemon (boot start, headless/SSH); else a per-user LaunchAgent

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

## Admin UI conventions

Server-rendered Jinja + Alpine.js, **not** a SPA. No build step, no bundler, no npm. `base.html` is extended by every page; `dashboard.js` (~237 KB) is the Alpine root. Every UI action is `@click`→JS method→`fetch('/admin/api/...', {credentials:'same-origin'})`.

CDN dependencies are **vendored offline** into `omlx/admin/static/` via `omlx/admin/vendor_deps.py` — the `JS_DEPS`/`CSS_DEPS` dicts are the source of truth. To add or bump a vendored dep, edit those dicts and run `python omlx/admin/vendor_deps.py`; never hand-edit a vendored file or leave a live CDN URL in a template. The static handler is a route (not a `StaticFiles` mount), so anything under `static/` is served at `/admin/static/...` with no config change.

The WebMCP layer (`static/js/webmcp/`) is a parallel ES-module tree loaded from `base.html`; it does not touch `dashboard.js` or the inline partial scripts. Note `.gitignore` has a broad Python `lib/` rule — `webmcp/lib/` is kept tracked by an explicit negation; new ignored-by-default paths under it need the same treatment.

## Dependency pins

`mlx-lm`, `mlx-vlm`, `mlx-embeddings`, `mlx-audio`, and `dflash-mlx` are pinned to **specific git commits** in `pyproject.toml`, with `[tool.uv] override-dependencies` forcing the resolver to accept the git `mlx-lm` over transitive pins. The comments above each pin explain why — read them before bumping. `torch` is intentionally absent from the core install (mlx-vlm uses custom processors); only the `[grammar]` extra pulls it in via xgrammar.

## Packaging

`packaging/build.py` builds the embedded Python layers (venvstacks) consumed by the Swift `.app` — it strips unused packages (torch/pandas/cv2, ~780 MB) and the `[bundle]` extra in `pyproject.toml` is its single source of truth for the mlx-base layer. Changes to the `omlx/` package need no `build.py` change (the bundle reinstalls the package); only touch `build.py` for the macOS packaging/codesign pipeline itself.

## Conventions

- License header on new source files: `# SPDX-License-Identifier: Apache-2.0`
- Test naming: `omlx/<module>.py` → `tests/test_<module>.py`
- `origin` is the fork (`jasonpaulso/omlx-plus`); `upstream` is `jundot/omlx`. Local `main` tracks `upstream/main`.
