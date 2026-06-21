# omlx-agent

A Claude Code plugin that lets you run a **running** [oMLX](https://github.com/jundot/omlx) server through an on-request DevOps engineer instead of the dashboard. It bundles:

- a **DevOps agent** (`omlx-devops`) that operates the server on your behalf, exercises judgment against a desired state you declare, and reports conclusions;
- an **MCP server** (`omlx`) exposing model management, inference, settings, logs, and cache tools over oMLX's REST API (the agent's hands);
- **skills** for operating the server (`operating-omlx`), onboarding the engineer (`omlx-devops-setup`), and live watching (`omlx-health-watch`).

This is oMLX-as-a-managed-target, distinct from oMLX's own MCP *client* (which dials out to external MCP servers) and from its browser-side WebMCP layer. Same protocol name, opposite direction.

## The DevOps engineer

Run `omlx-devops-setup` once. It interviews you to set the foundation: desired
state (pinned models, default, resource budgets), how much autonomy the engineer
has (a green/yellow/red action policy you tune), and which modes to turn on. It
writes a runbook to `~/.omlx/devops/`. The `omlx-devops` agent reads that runbook
on every invocation and refuses to act until it exists.

### Three modes, each opt-in

- **Reactive** (always available, only acts when invoked). Delegate any oMLX job: "get a fast coding model running", "why is it slow", "reconcile against my runbook". The agent loops out-of-context and returns the outcome.
- **Scheduled health check** (off until you opt in during setup). Runs the engineer on a cadence via the `schedule` mechanism. Start it report-only; widen to acting once trusted.
- **Live log monitor** (off until you invoke `omlx-health-watch`). Tails the server log and surfaces errors/warnings for the current session.

## Install

```bash
claude plugin marketplace add jundot/omlx
claude plugin install omlx-agent@omlx
```

(Or add this repo as a local marketplace during development.)

## Requirements

- A running oMLX server (`omlx serve`, the menubar app, or `omlx start`).
- [`uv`](https://docs.astral.sh/uv/) on `PATH` — the server launches via `uv run --with mcp --with httpx`, which provisions dependencies in an ephemeral, cached environment. No manual `pip install` needed.

## Configuration

Two environment variables, read by the MCP server:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMLX_BASE_URL` | `http://127.0.0.1:8000` | Base URL of the running server. Set for a remote host. |
| `OMLX_API_KEY` | _(empty)_ | API key. Used as a `Bearer` token for `/v1/*` and to log in to the cookie-authenticated `/admin/api/*` surface. Required unless the server runs with auth disabled. |

Export them before launching the host:

```bash
export OMLX_API_KEY="your-key"
export OMLX_BASE_URL="http://127.0.0.1:8000"
```

Verify the server is connected with `/mcp`.

## Tools

`system_info`, `server_stats`, `tail_logs`, `list_models`, `list_models_detailed`, `load_model`, `unload_model`, `reload_models`, `set_default_model`, `set_model_pinned`, `search_models`, `download_model`, `download_status`, `get_settings`, `clear_ssd_cache`, `clear_hot_cache`, `chat`, `complete`, `embed`.

The server wraps oMLX's existing REST endpoints — no new backend is required. To run without `uv`, install `mcp` and `httpx` into a Python env and change the `.mcp.json` `command`/`args` to invoke that interpreter against `mcp/omlx_mcp_server.py`.
