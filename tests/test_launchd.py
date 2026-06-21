# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the launchd lifecycle helper (pure logic only)."""

import os
from pathlib import Path

from omlx.utils import launchd


def test_build_plist_agent_has_no_username():
    plist = launchd.build_plist(system=False, omlx_executable="/opt/venv/bin/omlx")
    assert plist["Label"] == "com.omlx.server"
    assert plist["ProgramArguments"] == ["/opt/venv/bin/omlx", "serve"]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert "UserName" not in plist
    # PATH puts the omlx bin dir first so child resolution is correct.
    assert plist["EnvironmentVariables"]["PATH"].startswith("/opt/venv/bin")


def test_build_plist_daemon_sets_username():
    plist = launchd.build_plist(system=True, omlx_executable="/opt/venv/bin/omlx")
    assert plist["UserName"]  # non-empty; the invoking (non-root) user


def test_domain_and_target():
    uid = os.getuid()
    assert launchd._domain(False) == f"gui/{uid}"
    assert launchd._domain(True) == "system"
    assert launchd._domain_target(False) == f"gui/{uid}/com.omlx.server"
    assert launchd._domain_target(True) == "system/com.omlx.server"


def test_plist_paths():
    assert launchd._plist_path(False) == (
        Path.home() / "Library" / "LaunchAgents" / "com.omlx.server.plist"
    )
    assert launchd._plist_path(True) == Path(
        "/Library/LaunchDaemons/com.omlx.server.plist"
    )
