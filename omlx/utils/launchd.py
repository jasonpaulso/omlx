# SPDX-License-Identifier: Apache-2.0
"""launchd-managed background lifecycle for source / uv-tool / pip installs.

The macOS `.app` and Homebrew have their own managed lifecycle (the app control
socket and `brew services`). A plain `pip` / `uv tool` / source checkout had no
managed background option, so `omlx start` could only suggest `omlx serve`.

This module fills that gap with a launchd job labelled ``com.omlx.server``:

- **LaunchAgent** (default) at ``~/Library/LaunchAgents/com.omlx.server.plist`` —
  per-user, starts at login. Right for a workstation.
- **LaunchDaemon** (``--system``) at ``/Library/LaunchDaemons/com.omlx.server.plist`` —
  system-wide, starts at boot before any GUI login, runs as the invoking user.
  Right for a headless box reached over SSH. Privileged steps shell out to
  ``sudo``; the CLI itself is expected to run as the normal user (not via
  ``sudo omlx ...``), so the job's ``UserName``/``HOME`` resolve correctly.

Both invoke the same ``omlx`` console script that is running this command, so the
generated job points at the correct per-machine interpreter automatically. The
job runs ``omlx serve`` with no flags, so it reads ``~/.omlx/settings.json`` and
nothing is mutated.
"""

from __future__ import annotations

import getpass
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LABEL = "com.omlx.server"
_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
_DAEMON_DIR = Path("/Library/LaunchDaemons")


def _plist_path(system: bool) -> Path:
    return (_DAEMON_DIR if system else _AGENT_DIR) / f"{LABEL}.plist"


def _domain(system: bool) -> str:
    return "system" if system else f"gui/{os.getuid()}"


def _domain_target(system: bool) -> str:
    return f"{_domain(system)}/{LABEL}"


def _omlx_executable() -> str:
    """Absolute path to the ``omlx`` console script running this command."""
    cand = shutil.which(sys.argv[0]) or shutil.which("omlx") or sys.argv[0]
    return os.path.realpath(cand)


def _log_dir() -> Path:
    return Path.home() / ".omlx" / "logs"


def _server_port() -> int:
    """Best-effort read of the configured port, for health waiting."""
    import json

    settings = Path.home() / ".omlx" / "settings.json"
    try:
        data = json.loads(settings.read_text())
        return int(data["server"]["port"])
    except (OSError, KeyError, ValueError, TypeError):
        return 8000


def build_plist(system: bool, omlx_executable: str | None = None) -> dict:
    """Return the launchd job definition as a plist-ready dict.

    Pure and side-effect free so it can be unit-tested without touching launchd.
    """
    omlx = omlx_executable or _omlx_executable()
    log_dir = _log_dir()
    env = {
        "HOME": str(Path.home()),
        "PATH": os.pathsep.join(
            [
                str(Path(omlx).parent),
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
                "/usr/local/bin",
            ]
        ),
    }
    plist: dict = {
        "Label": LABEL,
        "ProgramArguments": [omlx, "serve"],
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_dir / "launchd.out.log"),
        "StandardErrorPath": str(log_dir / "launchd.err.log"),
    }
    if system:
        # System daemon runs as the invoking user (not root), so it uses that
        # user's ~/.omlx and can reach user-owned model volumes.
        plist["UserName"] = os.environ.get("SUDO_USER") or getpass.getuser()
    return plist


def detect_installed() -> bool | None:
    """Return True if a daemon is installed, False for an agent, else None."""
    if _plist_path(True).exists():
        return True
    if _plist_path(False).exists():
        return False
    return None


def _run(cmd: list[str], *, sudo: bool, quiet: bool = False) -> int:
    full = (["sudo"] + cmd) if sudo else cmd
    out = subprocess.DEVNULL if quiet else None
    return subprocess.run(full, stdout=out, stderr=out).returncode


def _is_loaded(system: bool) -> bool:
    return (
        _run(["launchctl", "print", _domain_target(system)], sudo=system, quiet=True)
        == 0
    )


def _wait_gone(system: bool, timeout: float = 10.0) -> None:
    """Block until the job is fully unloaded.

    launchctl bootout is asynchronous; bootstrapping the same label before the
    old job finishes tearing down fails with EIO (error 5). Polling print until
    it reports "no such service" closes that race.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_loaded(system):
            return
        time.sleep(0.3)


def _write_plist(plist: dict, system: bool) -> Path:
    target = _plist_path(system)
    if system:
        # Write to a temp file we own, then install with root ownership.
        with tempfile.NamedTemporaryFile("wb", suffix=".plist", delete=False) as tmp:
            plistlib.dump(plist, tmp)
            tmp_path = Path(tmp.name)
        if _run(["cp", str(tmp_path), str(target)], sudo=True) != 0:
            raise RuntimeError(f"sudo cp to {target} failed")
        _run(["chown", "root:wheel", str(target)], sudo=True)
        _run(["chmod", "644", str(target)], sudo=True)
        tmp_path.unlink(missing_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            plistlib.dump(plist, fh)
    return target


def _wait_healthy(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _start(system: bool, timeout: float, no_wait: bool) -> int:
    _log_dir().mkdir(parents=True, exist_ok=True)
    target = _write_plist(build_plist(system), system)
    domain = _domain(system)
    # Reload cleanly so an updated plist takes effect. bootout is async, so wait
    # for the old job to fully unload before bootstrapping (else EIO error 5).
    _run(["launchctl", "bootout", _domain_target(system)], sudo=system, quiet=True)
    _wait_gone(system)
    if _run(["launchctl", "bootstrap", domain, str(target)], sudo=system) != 0:
        print(f"Failed to bootstrap {_domain_target(system)} from {target}")
        return 1
    _run(["launchctl", "enable", _domain_target(system)], sudo=system)

    kind = "LaunchDaemon" if system else "LaunchAgent"
    if no_wait:
        print(f"oMLX {kind} loaded ({LABEL}).")
        return 0
    port = _server_port()
    if _wait_healthy(port, timeout):
        print(f"oMLX server running on port {port} via {kind} ({LABEL}).")
        return 0
    print(
        f"oMLX {kind} loaded but port {port} did not open within {int(timeout)}s. "
        f"Check {_log_dir() / 'launchd.err.log'}."
    )
    return 1


def _stop(system: bool) -> int:
    if (
        _run(["launchctl", "bootout", _domain_target(system)], sudo=system, quiet=True)
        != 0
    ):
        print(f"oMLX service {_domain_target(system)} was not loaded.")
        return 0
    print(f"oMLX stopped ({_domain_target(system)}).")
    return 0


def _restart(system: bool, timeout: float, no_wait: bool) -> int:
    rc = _run(["launchctl", "kickstart", "-k", _domain_target(system)], sudo=system)
    if rc != 0:
        # Not loaded yet — install and start it.
        return _start(system, timeout, no_wait)
    if no_wait:
        print(f"oMLX restart requested ({_domain_target(system)}).")
        return 0
    port = _server_port()
    if _wait_healthy(port, timeout):
        print(f"oMLX server restarted on port {port} ({_domain_target(system)}).")
        return 0
    print(f"oMLX restarted but port {port} did not open within {int(timeout)}s.")
    return 1


def lifecycle(command: str, *, system: bool, timeout: float, no_wait: bool) -> int:
    """Dispatch start/stop/restart for the launchd-managed service.

    For stop/restart, ``system`` may be left False to auto-detect which job is
    installed; an explicit ``--system`` always forces the daemon domain.
    """
    if command in {"stop", "restart"} and not system:
        detected = detect_installed()
        if detected is None:
            print(
                "No launchd-managed oMLX service is installed. "
                "Start one with: omlx start  (or: omlx start --system)"
            )
            return 1
        system = detected

    if command == "start":
        return _start(system, timeout, no_wait)
    if command == "stop":
        return _stop(system)
    if command == "restart":
        return _restart(system, timeout, no_wait)
    print(f"Unknown lifecycle command: {command}")
    return 1
