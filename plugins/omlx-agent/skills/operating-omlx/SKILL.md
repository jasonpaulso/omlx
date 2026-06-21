---
name: operating-omlx
description: Use when interacting with a running oMLX inference server — listing/loading/downloading models, running chat/completions/embeddings against it, checking server health/stats/logs, adjusting settings, or managing the KV cache. Triggers on mentions of oMLX, a local MLX server, "load a model", "what models are available", or pointing an agent at an OpenAI-compatible endpoint on this Mac.
version: 0.1.0
---

# Operating a running oMLX server

oMLX is a local MLX inference server for Apple Silicon, exposing OpenAI- and
Anthropic-compatible APIs plus an admin management surface. This skill drives a
*running* instance through the `omlx` MCP server bundled with this plugin.

## Connection

The MCP server reads two environment variables (set them in your shell before
launching the host, or in Claude Code's env):

- `OMLX_BASE_URL` — default `http://127.0.0.1:8000`. Point at a remote host if oMLX runs elsewhere.
- `OMLX_API_KEY` — the server's API key. Used as a Bearer token for `/v1/*` **and** to log in to the management surface. Leave unset only if the server runs with auth disabled.

If a tool returns `CONNECTION_FAILED`, oMLX isn't running or the URL is wrong —
the user starts it with `omlx serve` (or the menubar app / `omlx start`). If it
returns `UNAUTHORIZED` or `LOGIN_FAILED`, fix `OMLX_API_KEY`.

## Tool catalog

All tools are `mcp__plugin_omlx-agent_omlx__<name>`.

**Health & introspection**
- `system_info` — hardware, OS, free disk, version. Check disk before downloads.
- `server_stats` — request/token counts, latencies, memory, queue depth.
- `tail_logs(lines, min_level)` — recent log lines; the first place to look when something fails.

**Models — inspect**
- `list_models` — what's servable right now (`/v1/models`).
- `list_models_detailed` — full inventory with loaded/default/pinned/size. Empty ⇒ nothing downloaded.

**Models — manage**
- `load_model(model_id)` / `unload_model(model_id)` — bring a model in/out of memory.
- `set_default_model(model_id)` — what `/v1/*` uses when `model` is omitted.
- `set_model_pinned(model_id, pinned)` — pin to auto-load and resist eviction.
- `reload_models` — rescan model dirs after adding files on disk.

**Models — download**
- `search_models(query, mlx_only, sort)` — HuggingFace Hub search.
- `download_model(repo_id, token)` — returns a `task_id`; long-running.
- `download_status` — poll until a task is `completed` or `failed`.

**Inference**
- `chat(message, model, system, temperature, max_tokens)` — one-shot chat turn.
- `complete(prompt, model, max_tokens, temperature)` — raw completion.
- `embed(text, model)` — embedding vector (needs an embedding model loaded).

**Settings & cache**
- `get_settings` — full global settings tree.
- `clear_ssd_cache` / `clear_hot_cache` — **destructive**; confirm with the user first.

## Common workflows

**Cold start (no models downloaded).** `list_models_detailed` → empty →
`system_info` to check disk → `search_models("qwen3 4b")` → confirm choice with
the user → `download_model(repo_id)` → poll `download_status` until completed →
`load_model(model_id)` → optionally `set_default_model`.

**Run something.** `list_models` to see what's loaded → `chat("...", model=...)`.
If nothing is loaded, `load_model` first (or let the server auto-load the default).

**Switch models.** `unload_model(old)` then `load_model(new)` — unload first if
RAM is tight (check `system_info` / `server_stats`).

**Diagnose a failure.** `tail_logs(min_level="error")` before guessing. Inference
errors, OOM guards, and load failures all surface there.

## Notes

- The server auto-loads the default model on first `/v1/*` request if none is loaded, so `chat` may work even when `list_models` looks empty — but explicit `load_model` is clearer.
- Downloads and cache clears change disk/RAM state; surface size and confirm destructive actions with the user.
- For pure inference you can also point any OpenAI-compatible client straight at `OMLX_BASE_URL/v1` with the API key — the MCP tools add management on top of that.
