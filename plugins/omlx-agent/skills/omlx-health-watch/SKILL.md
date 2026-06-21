---
name: omlx-health-watch
description: Start live monitoring of the running oMLX server. Use when the user wants to watch oMLX health in real time this session — "watch the oMLX server", "tail oMLX for errors", "keep an eye on oMLX while I work". Opt-in trigger for the background log monitor; it runs only for the current session.
version: 0.1.0
---

# oMLX live health watch

Invoking this skill is the opt-in switch for live monitoring. It starts the
`omlx-log-watch` background monitor, which tails `~/.omlx/logs/server.log` and
surfaces error and warning lines as notifications while you work. The monitor
runs only in this session and stops when the session ends.

When this skill is invoked:

1. Confirm to the user that live watching is now on for this session, and what
   it surfaces (errors, warnings, OOM/guard events, failed loads).
2. As monitor notifications arrive, treat each as a signal, not a command.
   Triage with the `omlx-devops` agent when a line looks actionable: hand it the
   log line and let it correlate with `server_stats` / `system_info` and
   recommend within its autonomy tier. Do not take disruptive action directly
   from a log line.
3. If the user has not run `omlx-devops-setup`, mention that triage will be
   limited until a runbook exists.

This is one of three modes (reactive agent, scheduled health check, live
monitor). It does not enable the others. To stop watching, end the session.
