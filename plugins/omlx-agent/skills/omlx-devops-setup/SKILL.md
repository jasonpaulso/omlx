---
name: omlx-devops-setup
description: Onboard the oMLX DevOps engineer. Use to set up, configure, or re-run the oMLX devops runbook — declaring desired state (pinned models, default, resource budgets), setting autonomy boundaries, and opting into health/maintenance modes. Triggers on "set up the oMLX engineer", "onboard oMLX devops", "configure the oMLX runbook", or first use of the omlx-devops agent.
version: 0.1.0
---

# Onboarding the oMLX DevOps engineer

This skill conducts an interview with the user and writes their runbook. The
`omlx-devops` agent refuses to act until this has produced
`~/.omlx/devops/runbook.md`. Run it once at onboarding, or again to change the
foundation.

Conduct this as a real conversation. Ask in small batches, propose concrete
defaults by reading the live server first, and never invent the user's
preferences. Everything here is a decision the user owns.

## Step 0 — Connect and read reality

Confirm the server is reachable (the `omlx` MCP tools rely on `OMLX_BASE_URL`
and `OMLX_API_KEY`). Then read `list_models_detailed`, `system_info`, and
`server_stats` so every question below can be anchored to real numbers and the
actual model inventory. If the server is unreachable, stop and help the user
start it before continuing.

## Step 1 — Interview

Cover these areas. Lead each with what you observed, then ask.

1. **Role and reporting.** Confirm the engineer's remit. How terse should
   reports be? What rises to "tell me now" versus "mention next time"?
2. **Desired model state.** From the inventory: which models should always be
   pinned (auto-load, resist eviction)? Which is the default for `/v1/*`
   requests with no model named? Anything that should never be auto-loaded?
3. **Resource budgets.** Minimum free disk to maintain (GB). Maximum models
   loaded at once. Memory-guard tier (aggressive / balanced / permissive).
   Anchor these to the RAM and free disk you read in `system_info`.
4. **Autonomy boundaries.** Walk the three tiers from the agent and let the user
   move actions between them. The defaults:
   - Green (silent): read-only checks, reconcile pin/default, load the default
     if nothing is loaded, report drift.
   - Yellow (propose first): load/unload non-default, change settings, download,
     clear hot cache.
   - Red (explicit go every time): restart server, clear SSD cache, delete a
     model, anything during active inference.
   Ask specifically: may it download models on its own? may it unload to free
   RAM on its own? Record exactly where the user draws the line.
5. **Modes to enable.** Each is OFF until chosen here:
   - **Reactive** (the `omlx-devops` agent answering on request). Always
     available; only acts when invoked. Confirm it is on.
   - **Scheduled health check.** Opt in? If yes: cadence (e.g. daily 9am), what
     to check (drift, disk, errors-since-last-run, loaded set), and whether it
     may act in-tier or report-only. See Step 3.
   - **Live log monitor.** Opt in? If yes, see Step 4.

## Step 2 — Write the runbook

Read `runbook-template.md` from this skill directory. Fill every `{{...}}`
placeholder from the interview. Write the result to `~/.omlx/devops/runbook.md`
(create the directory). Also write `~/.omlx/devops/desired-state.json` with the
machine-readable desired state, for example:

```json
{
  "pinned": ["<model-id>"],
  "default": "<model-id>",
  "max_loaded": 2,
  "min_free_disk_gb": 50,
  "memory_guard_tier": "balanced",
  "autonomy": { "download": "yellow", "unload": "yellow", "restart": "red" }
}
```

Read both files back to the user and confirm before finishing.

## Step 3 — Scheduled health check (only if opted in)

This mode runs the engineer on a cadence with no human present. A plugin cannot
create a recurring job on its own, so set it up with the user using the
`schedule` skill. The routine prompt should be: "Invoke the omlx-devops agent
for a health check: reconcile against the runbook, and {act in-tier | report
only}." Record the cadence and scope in the runbook's Modes section. Because
unattended runs have no session to type a key into, confirm `OMLX_API_KEY` is
available to the scheduler's environment. Start report-only; let the user widen
to acting after they trust it.

## Step 4 — Live log monitor (only if opted in)

Tell the user to invoke the `omlx-health-watch` skill in any session where they
want live watching. That is the opt-in trigger: it starts a background monitor
that tails the oMLX server log and surfaces error/warning lines as they happen.
It runs only for that session and stops when the session ends. Record in the
runbook that this mode is available and how to start it.

## Step 5 — Confirm

Summarize: desired state, autonomy lines, and which modes are now enabled.
Remind the user they can re-run this skill anytime to change the foundation,
and that the `omlx-devops` agent now has what it needs.
