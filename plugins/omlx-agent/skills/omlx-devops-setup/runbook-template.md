# oMLX DevOps Runbook

The operating contract for the `omlx-devops` engineer. Written by the
`omlx-devops-setup` skill. The agent reads this on every invocation and treats
it as authoritative. Re-run setup to change it; append to the change journal as
work happens.

- **Server:** {{base_url}}
- **Onboarded:** {{date}}
- **Reporting style:** {{terse_or_detailed}}
- **Escalate immediately when:** {{escalation_conditions}}

## Desired state

The engineer keeps reality matching this and reports drift.

- **Pinned (always loaded, resist eviction):** {{pinned_models}}
- **Default model (`/v1/*` with no model named):** {{default_model}}
- **Never auto-load:** {{never_autoload}}
- **Max models loaded at once:** {{max_loaded}}
- **Minimum free disk to maintain:** {{min_free_disk_gb}} GB
- **Memory-guard tier:** {{memory_guard_tier}}

Machine-readable copy lives in `desired-state.json` beside this file.

## Autonomy tiers

What the engineer may do without asking, versus what it must propose or escalate.

- **Green (silent, then report):** {{green_actions}}
- **Yellow (propose, wait for yes):** {{yellow_actions}}
- **Red (explicit human go, every time):** {{red_actions}}

Specific lines the user drew:
- May download models autonomously: {{autonomy_download}}
- May unload models to free RAM autonomously: {{autonomy_unload}}
- May restart the server: {{autonomy_restart}}

## Modes

Each is enabled only if listed as ON.

- **Reactive (on-request engineer):** {{mode_reactive}}
- **Scheduled health check:** {{mode_scheduled}} — cadence {{schedule_cadence}}, scope {{schedule_scope}}
- **Live log monitor:** {{mode_monitor}} — start by invoking the `omlx-health-watch` skill

## Known failure-mode runbook

How to handle recurring situations. Extend as the engineer learns this install.

- **Memory-guard rejection on load:** unload the largest non-pinned loaded model (within tier), then retry; if still failing, report the shortfall in GB.
- **Download stuck / failed:** cancel the task and retry once; if it fails again, report the repo and the error from `tail_logs`.
- **Model will not load:** check the model's compatibility flags in `list_models_detailed` (MTP / dflash / engine type) and report the specific reason.
- **Server slow:** correlate `server_stats` (queue depth, latencies) with `tail_logs(error)` and `system_info` (memory pressure) before concluding.

## Change journal

Most recent first. One line per significant action.

- {{date}} — Runbook created via omlx-devops-setup.
