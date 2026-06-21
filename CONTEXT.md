# Session Context — 2026-06-21 (afternoon) — WebMCP + omlx-agent plugin shipped

**Status:** Fork `main` has the WebMCP layer, the omlx-agent plugin, Claude GHA workflows, and a signed/iconned `oMLX.app` staged. Plugin in multi-day testing before any upstream PR.

| Work | ID/Hash | Surface |
| --- | --- | --- |
| WebMCP tool surface (47 tools) | f0d3b2a | omlx/admin/static/js/webmcp/ + base.html |
| packaging codesign fix | 7ca9c6f | packaging/build.py |
| CLAUDE.md | 837327c | repo root |
| omlx-agent plugin (PR #1) | 02270c5 / 4f61c84 | plugins/omlx-agent/, .claude-plugin/marketplace.json |
| Claude GHA workflows (PR #2) | 36a844a | .github/workflows/ |
| actool Xcode-resolve fix | e917abd | apps/omlx-mac/Scripts/build.sh |

**WebMCP:** every admin action registered in-page via `navigator.modelContext`; discoverable by browser agents (the meta-tag + sync-polyfill + relay-embed + allowedOrigins:* combo) and bridgeable to stdio MCP hosts via the local relay.

**omlx-agent plugin:** an on-request `omlx-devops` subagent + a stdio MCP server (uv-launched, ~19 tools over oMLX REST) + skills (operating-omlx, omlx-devops-setup interview, omlx-health-watch). Three opt-in modes: reactive / scheduled health check / live monitor. Autonomy tiered green/yellow/red. Runbook lives at `~/.omlx/devops/` once setup runs.

**Build:** `oMLX.app` v0.4.4 staged at `apps/omlx-mac/build/Stage/` via `build.sh swift` (fast path; deps at Jun-9 export snapshot, omlx package fresh). codesign verifies; real icon compiles after the actool fix.

**Git note:** local main drifted 6 behind origin/main because both PRs were merged on GitHub; resolved by fetch + rebase, not blind push. Pull local main after GitHub-side merges.

**Next-tier teed up:**
- Multi-day testing of the omlx-agent plugin (reactive mode first); hold upstream PR until greenlit.
- Optional: run `omlx-devops-setup` to produce a runbook; opt into scheduled/monitor modes once trusted.
- Eventual clean release build via the full venvstacks rebuild path (`pip install -e ".[dev]"` + `release --rebuild-donor`).

---
