# SPDX-License-Identifier: Apache-2.0
"""Fork: enable_routing on ModelSettings, and the M4.3 transient override layer.

Lives in a fork-owned file rather than appended to tests/test_model_settings.py
so upstream can edit that file freely without conflicting.
"""

import tempfile
from pathlib import Path

from omlx.model_settings import ModelSettings, ModelSettingsManager


class TestEnableRouting:
    def test_default_off(self):
        """Semantic-routing opt-in. Default off."""
        assert ModelSettings().enable_routing is False

    def test_roundtrip(self):
        """enable_routing survives to_dict -> from_dict roundtrip."""
        d = ModelSettings(enable_routing=True).to_dict()
        assert d["enable_routing"] is True
        assert ModelSettings.from_dict(d).enable_routing is True

    def test_excluded_from_profiles(self):
        """Routing eligibility is a base-model property, never in a profile."""
        from omlx.model_profiles import EXCLUDED_FROM_PROFILES

        assert "enable_routing" in EXCLUDED_FROM_PROFILES


class TestOverrideSettings:
    def test_override_settings_win_over_persisted_and_baseline(self):
        """M4.3: transient variant override beats baseline and stored settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.9, mtp_enabled=False))
            # Baseline mode would normally force stock defaults...
            manager.set_baseline_ids({"m"})
            # ...but an override takes precedence and yields stock + one knob.
            manager.set_override_settings({"m": ModelSettings(mtp_enabled=True)})
            got = manager.get_settings("m")
            assert got.mtp_enabled is True
            assert got.temperature is None  # stock sampling, not the persisted 0.9
            # Clearing the override falls back to baseline (stock).
            manager.clear_override_settings()
            assert manager.get_settings("m").mtp_enabled is False
            # Clearing baseline too restores the persisted settings.
            manager.clear_baseline_ids()
            assert manager.get_settings("m").temperature == 0.9

    def test_override_is_isolated_copy(self):
        """Mutating the source ModelSettings must not leak into the override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            src = ModelSettings(mtp_enabled=True)
            manager.set_override_settings({"m": src})
            src.mtp_enabled = False
            assert manager.get_settings("m").mtp_enabled is True
