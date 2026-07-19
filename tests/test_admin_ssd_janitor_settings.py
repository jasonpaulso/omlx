# SPDX-License-Identifier: Apache-2.0
"""Fork: SSD janitor knobs in update_global_settings.

Lives in a fork-owned file rather than appended to test_admin_server_aliases.py
so upstream can edit that file freely. Helpers are reused from it by import.
"""

import asyncio

from omlx.admin import routes as admin_routes
from omlx.admin.routes import GlobalSettingsRequest

from test_admin_server_aliases import _make_global_settings, _patched_global_settings


class TestUpdateGlobalSettingsSsdJanitor:
    """update_global_settings: save the SSD janitor settings (restart required)."""

    def test_saves_ssd_janitor_settings(self):
        gs = _make_global_settings()
        request = GlobalSettingsRequest(
            ssd_janitor_enabled=True,
            ssd_janitor_interval_s=60,
            ssd_janitor_max_unlinks_per_sweep=1024,
        )

        with _patched_global_settings(gs):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert result["success"] is True
        assert gs.cache.ssd_janitor_enabled is True
        assert gs.cache.ssd_janitor_interval_s == 60
        assert gs.cache.ssd_janitor_max_unlinks_per_sweep == 1024
        # Restart-required, like initial_cache_blocks: no runtime cache re-apply.
        assert "cache" not in result["runtime_applied"]
        gs.save.assert_called_once()
