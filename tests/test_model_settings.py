# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.model_settings module."""

import json
import tempfile
from pathlib import Path

import pytest

from omlx.model_settings import ModelSettings, ModelSettingsManager


class TestModelSettings:
    """Tests for ModelSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ModelSettings()
        assert settings.max_context_window is None
        assert settings.max_tokens is None
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.top_k is None
        assert settings.repetition_penalty is None
        assert settings.force_sampling is False
        assert settings.is_pinned is False
        assert settings.is_default is False
        # Issue #926: opt-in per model. Default off.
        assert settings.trust_remote_code is False

    def test_trust_remote_code_roundtrip(self):
        """Test trust_remote_code field survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(trust_remote_code=True)
        d = original.to_dict()
        assert d["trust_remote_code"] is True
        restored = ModelSettings.from_dict(d)
        assert restored.trust_remote_code is True

    def test_trust_remote_code_excluded_from_profiles(self):
        """Security flag must never propagate via profiles or templates."""
        from omlx.model_profiles import EXCLUDED_FROM_PROFILES
        assert "trust_remote_code" in EXCLUDED_FROM_PROFILES

    def test_max_context_window(self):
        """Test max_context_window field."""
        settings = ModelSettings(max_context_window=4096)
        assert settings.max_context_window == 4096
        d = settings.to_dict()
        assert d["max_context_window"] == 4096

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        settings = ModelSettings(temperature=0.7, is_pinned=True)
        d = settings.to_dict()
        assert "temperature" in d
        assert "is_pinned" in d
        assert "max_tokens" not in d  # None should be excluded
        assert "max_context_window" not in d  # None should be excluded
        assert "repetition_penalty" not in d  # None should be excluded

    def test_to_dict_preserves_zero_values(self):
        """Test to_dict preserves zero values (not treated as None)."""
        settings = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
        d = settings.to_dict()
        assert "temperature" in d
        assert d["temperature"] == 0.0
        assert "top_p" in d
        assert d["top_p"] == 0.0
        assert "top_k" in d
        assert d["top_k"] == 0

    def test_zero_values_roundtrip(self):
        """Test zero values survive to_dict -> from_dict roundtrip."""
        original = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
        restored = ModelSettings.from_dict(original.to_dict())
        assert restored.temperature == 0.0
        assert restored.top_p == 0.0
        assert restored.top_k == 0

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "temperature": 0.8,
            "repetition_penalty": 1.3,
            "is_pinned": True,
            "invalid_key": "should be ignored"
        }
        settings = ModelSettings.from_dict(data)
        assert settings.temperature == 0.8
        assert settings.repetition_penalty == 1.3
        assert settings.is_pinned is True
        assert not hasattr(settings, "invalid_key")

    def test_repetition_penalty_roundtrip(self):
        """Test repetition_penalty survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(repetition_penalty=1.5)
        d = original.to_dict()
        assert d["repetition_penalty"] == 1.5
        restored = ModelSettings.from_dict(d)
        assert restored.repetition_penalty == 1.5

    def test_chat_template_kwargs_default(self):
        """Test chat_template_kwargs defaults to None."""
        settings = ModelSettings()
        assert settings.chat_template_kwargs is None

    def test_chat_template_kwargs_to_dict(self):
        """Test chat_template_kwargs included in to_dict when set."""
        settings = ModelSettings(
            chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}
        )
        d = settings.to_dict()
        assert "chat_template_kwargs" in d
        assert d["chat_template_kwargs"]["enable_thinking"] is False
        assert d["chat_template_kwargs"]["reasoning_effort"] == "low"

    def test_chat_template_kwargs_excluded_when_none(self):
        """Test chat_template_kwargs excluded from to_dict when None."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert "chat_template_kwargs" not in d

    def test_chat_template_kwargs_roundtrip(self):
        """Test chat_template_kwargs survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(
            chat_template_kwargs={"enable_thinking": True, "custom_key": 42}
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.chat_template_kwargs == {"enable_thinking": True, "custom_key": 42}

    def test_chat_template_kwargs_from_dict(self):
        """Test chat_template_kwargs created from dict."""
        data = {
            "temperature": 0.8,
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }
        settings = ModelSettings.from_dict(data)
        assert settings.temperature == 0.8
        assert settings.chat_template_kwargs == {"reasoning_effort": "high"}


    def test_ttl_seconds_default(self):
        """Test ttl_seconds defaults to None."""
        settings = ModelSettings()
        assert settings.ttl_seconds is None

    def test_ttl_seconds_roundtrip(self):
        """Test ttl_seconds survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(ttl_seconds=300)
        d = original.to_dict()
        assert d["ttl_seconds"] == 300
        restored = ModelSettings.from_dict(d)
        assert restored.ttl_seconds == 300

    def test_ttl_seconds_excluded_when_none(self):
        """Test ttl_seconds excluded from to_dict when None."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert "ttl_seconds" not in d

    def test_model_alias_default(self):
        """Test model_alias defaults to None."""
        settings = ModelSettings()
        assert settings.model_alias is None

    def test_model_alias_roundtrip(self):
        """Test model_alias survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(model_alias="gpt-4")
        d = original.to_dict()
        assert d["model_alias"] == "gpt-4"
        restored = ModelSettings.from_dict(d)
        assert restored.model_alias == "gpt-4"

    def test_model_alias_excluded_when_none(self):
        """Test model_alias excluded from to_dict when None."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert "model_alias" not in d

    def test_model_type_override_default(self):
        """Test model_type_override defaults to None."""
        settings = ModelSettings()
        assert settings.model_type_override is None

    def test_model_type_override_roundtrip(self):
        """Test model_type_override survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(model_type_override="vlm")
        d = original.to_dict()
        assert d["model_type_override"] == "vlm"
        restored = ModelSettings.from_dict(d)
        assert restored.model_type_override == "vlm"

    def test_model_type_override_excluded_when_none(self):
        """Test model_type_override excluded from to_dict when None."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert "model_type_override" not in d


class TestModelSettingsManager:
    """Tests for ModelSettingsManager class."""

    def test_empty_settings(self):
        """Test with no settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = manager.get_settings("nonexistent")
            assert settings.is_pinned is False
            assert settings.is_default is False

    def test_load_existing_file(self):
        """Test loading from existing settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "model_settings.json"
            settings_file.write_text(json.dumps({
                "version": 1,
                "models": {
                    "llama-3b": {
                        "temperature": 0.7,
                        "is_pinned": True,
                        "is_default": True
                    }
                }
            }))

            manager = ModelSettingsManager(Path(tmpdir))
            settings = manager.get_settings("llama-3b")
            assert settings.temperature == 0.7
            assert settings.is_pinned is True
            assert settings.is_default is True

    def test_set_settings(self):
        """Test setting and saving settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(temperature=0.9, is_pinned=True)
            manager.set_settings("test-model", settings)

            # Verify saved
            loaded = manager.get_settings("test-model")
            assert loaded.temperature == 0.9
            assert loaded.is_pinned is True

            # Verify file was created
            settings_file = Path(tmpdir) / "model_settings.json"
            assert settings_file.exists()

    def test_zero_values_persist(self):
        """Test zero sampling values survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.temperature == 0.0
            assert loaded.top_p == 0.0
            assert loaded.top_k == 0

    def test_repetition_penalty_persist(self):
        """Test repetition_penalty survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(repetition_penalty=1.3)
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.repetition_penalty == 1.3

    def test_exclusive_default(self):
        """Test only one model can be default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            # Set first model as default
            settings1 = ModelSettings(is_default=True)
            manager.set_settings("model-1", settings1)
            assert manager.get_default_model_id() == "model-1"

            # Set second model as default
            settings2 = ModelSettings(is_default=True)
            manager.set_settings("model-2", settings2)

            # model-2 should be default, model-1 should not
            assert manager.get_default_model_id() == "model-2"
            assert manager.get_settings("model-1").is_default is False
            assert manager.get_settings("model-2").is_default is True

    def test_multiple_pinned(self):
        """Test multiple models can be pinned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            manager.set_settings("model-1", ModelSettings(is_pinned=True))
            manager.set_settings("model-2", ModelSettings(is_pinned=True))
            manager.set_settings("model-3", ModelSettings(is_pinned=False))

            pinned = manager.get_pinned_model_ids()
            assert "model-1" in pinned
            assert "model-2" in pinned
            assert "model-3" not in pinned

    def test_get_all_settings(self):
        """Test getting all settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            manager.set_settings("model-1", ModelSettings(temperature=0.5))
            manager.set_settings("model-2", ModelSettings(temperature=0.9))

            all_settings = manager.get_all_settings()
            assert len(all_settings) == 2
            assert "model-1" in all_settings
            assert "model-2" in all_settings

    def test_chat_template_kwargs_persist(self):
        """Test chat_template_kwargs survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(
                chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "medium"}
            )
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.chat_template_kwargs == {
                "enable_thinking": False,
                "reasoning_effort": "medium",
            }

    def test_chat_template_kwargs_clear(self):
        """Test clearing chat_template_kwargs by setting to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            # Set kwargs
            settings = ModelSettings(
                chat_template_kwargs={"enable_thinking": True}
            )
            manager.set_settings("test-model", settings)
            assert manager.get_settings("test-model").chat_template_kwargs is not None

            # Clear kwargs
            settings = ModelSettings(chat_template_kwargs=None)
            manager.set_settings("test-model", settings)
            loaded = manager.get_settings("test-model")
            assert loaded.chat_template_kwargs is None

    def test_forced_ct_kwargs_persist(self):
        """Test forced_ct_kwargs survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(
                chat_template_kwargs={"enable_thinking": False},
                forced_ct_kwargs=["enable_thinking"],
            )
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.forced_ct_kwargs == ["enable_thinking"]
            assert loaded.chat_template_kwargs == {"enable_thinking": False}

    def test_model_alias_persist(self):
        """Test model_alias survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(model_alias="my-model")
            manager.set_settings("test-model", settings)

            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.model_alias == "my-model"

    def test_model_alias_clear(self):
        """Test clearing model_alias by setting to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(model_alias="my-model")
            manager.set_settings("test-model", settings)
            assert manager.get_settings("test-model").model_alias == "my-model"

            settings = ModelSettings(model_alias=None)
            manager.set_settings("test-model", settings)
            loaded = manager.get_settings("test-model")
            assert loaded.model_alias is None

    def test_model_type_override_persist(self):
        """Test model_type_override survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(model_type_override="embedding")
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.model_type_override == "embedding"

    def test_model_type_override_clear(self):
        """Test clearing model_type_override by setting to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(model_type_override="vlm")
            manager.set_settings("test-model", settings)
            assert manager.get_settings("test-model").model_type_override == "vlm"

            # Clear override
            settings = ModelSettings(model_type_override=None)
            manager.set_settings("test-model", settings)
            loaded = manager.get_settings("test-model")
            assert loaded.model_type_override is None

    def test_forced_ct_kwargs_default_none(self):
        """Test forced_ct_kwargs defaults to None."""
        settings = ModelSettings()
        assert settings.forced_ct_kwargs is None
        d = settings.to_dict()
        assert "forced_ct_kwargs" not in d

    def test_forced_ct_kwargs_roundtrip(self):
        """Test forced_ct_kwargs survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
            forced_ct_kwargs=["enable_thinking", "reasoning_effort"],
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.forced_ct_kwargs == ["enable_thinking", "reasoning_effort"]

    def test_thread_safety(self):
        """Test thread-safe access."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            errors = []

            def worker(model_id):
                try:
                    for i in range(10):
                        manager.set_settings(model_id, ModelSettings(temperature=i/10))
                        _ = manager.get_settings(model_id)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(f"model-{i}",)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


class TestEphemeralOverrides:
    """Tests for ModelSettingsManager.ephemeral_overrides context manager.

    The override layer is the foundation of the bench-tab inline settings
    panel: it lets a benchmark run apply per-run overrides without writing
    to model_settings.json.
    """

    def test_noop_when_overrides_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            with manager.ephemeral_overrides("m", None):
                assert manager.get_settings("m").temperature == 0.5
            with manager.ephemeral_overrides("m", {}):
                assert manager.get_settings("m").temperature == 0.5

    def test_overrides_apply_inside_and_revert_after(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5, top_p=0.9))

            with manager.ephemeral_overrides("m", {"temperature": 0.1}):
                s = manager.get_settings("m")
                assert s.temperature == 0.1
                # Untouched fields keep persisted values.
                assert s.top_p == 0.9

            after = manager.get_settings("m")
            assert after.temperature == 0.5
            assert after.top_p == 0.9

    def test_persisted_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))
            settings_path = Path(tmpdir) / "model_settings.json"
            before = settings_path.read_text()

            with manager.ephemeral_overrides(
                "m", {"temperature": 0.1, "top_p": 0.7}
            ):
                pass

            assert settings_path.read_text() == before

    def test_overrides_apply_to_model_with_no_persisted_settings(self):
        """Engine-init flag overrides should work even for fresh models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            with manager.ephemeral_overrides(
                "fresh", {"turboquant_kv_enabled": True, "turboquant_kv_bits": 4}
            ):
                s = manager.get_settings("fresh")
                assert s.turboquant_kv_enabled is True
                assert s.turboquant_kv_bits == 4

            # And the manager has no persisted state for this model.
            assert manager.get_settings("fresh").turboquant_kv_enabled is False

    def test_unknown_keys_dropped(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            with caplog.at_level("WARNING"):
                with manager.ephemeral_overrides(
                    "m", {"temperature": 0.1, "totally_made_up_key": 42}
                ):
                    s = manager.get_settings("m")
                    assert s.temperature == 0.1
                    assert not hasattr(s, "totally_made_up_key")

            assert any("totally_made_up_key" in r.message for r in caplog.records)

    def test_none_value_defers_to_lower_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            with manager.ephemeral_overrides("m", {"temperature": None}):
                # None means "don't override this field" — persisted wins.
                assert manager.get_settings("m").temperature == 0.5

    def test_nested_overrides_inner_wins_then_outer_restored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            with manager.ephemeral_overrides("m", {"temperature": 0.2}):
                assert manager.get_settings("m").temperature == 0.2
                with manager.ephemeral_overrides("m", {"temperature": 0.1}):
                    assert manager.get_settings("m").temperature == 0.1
                # After inner exits, outer override is back in effect.
                assert manager.get_settings("m").temperature == 0.2
            # After outer exits, persisted wins.
            assert manager.get_settings("m").temperature == 0.5

    def test_out_of_order_exit_uses_token(self):
        """Two overlapping overrides exited out of LIFO order shouldn't corrupt state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            outer = manager.ephemeral_overrides("m", {"temperature": 0.2})
            inner = manager.ephemeral_overrides("m", {"temperature": 0.1})
            outer.__enter__()
            inner.__enter__()
            assert manager.get_settings("m").temperature == 0.1

            # Exit outer first (out of LIFO order).
            outer.__exit__(None, None, None)
            # Inner is still active, so its temperature wins.
            assert manager.get_settings("m").temperature == 0.1

            inner.__exit__(None, None, None)
            assert manager.get_settings("m").temperature == 0.5

    def test_overrides_released_on_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(temperature=0.5))

            with pytest.raises(RuntimeError):
                with manager.ephemeral_overrides("m", {"temperature": 0.1}):
                    assert manager.get_settings("m").temperature == 0.1
                    raise RuntimeError("boom")

            assert manager.get_settings("m").temperature == 0.5
            # Internal stack is empty.
            assert "m" not in manager._overrides

    def test_overrides_isolated_per_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("a", ModelSettings(temperature=0.5))
            manager.set_settings("b", ModelSettings(temperature=0.7))

            with manager.ephemeral_overrides("a", {"temperature": 0.1}):
                assert manager.get_settings("a").temperature == 0.1
                # Overrides for "a" don't leak into "b".
                assert manager.get_settings("b").temperature == 0.7

    def test_engine_init_flag_overrides(self):
        """TurboQuant/DFlash/MTP overrides should compose just like sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(turboquant_kv_enabled=False))

            with manager.ephemeral_overrides(
                "m",
                {
                    "turboquant_kv_enabled": True,
                    "turboquant_kv_bits": 3,
                    "dflash_enabled": True,
                },
            ):
                s = manager.get_settings("m")
                assert s.turboquant_kv_enabled is True
                assert s.turboquant_kv_bits == 3
                assert s.dflash_enabled is True

            after = manager.get_settings("m")
            assert after.turboquant_kv_enabled is False
            assert after.dflash_enabled is False

    def test_override_respects_mutual_exclusion_constraints(self):
        """ModelSettings rejects mtp_enabled=True with dflash_enabled=True.

        get_settings runs the merged dict through ModelSettings.from_dict,
        which triggers __post_init__ validation. An override that creates
        an invalid combination should surface as an exception from
        get_settings — not silently succeed with a corrupted state.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            manager.set_settings("m", ModelSettings(mtp_enabled=True))

            # Override turns on dflash_enabled while persisted has
            # mtp_enabled=True — invalid combo per ModelSettings.__post_init__.
            with manager.ephemeral_overrides("m", {"dflash_enabled": True}):
                with pytest.raises(Exception):
                    manager.get_settings("m")

            # Override is still released after the exception path.
            assert "m" not in manager._overrides

    def test_thread_safe_concurrent_overrides(self):
        """Concurrent overrides on different models don't corrupt each other."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            errors: list[Exception] = []

            def worker(model_id: str, target_temp: float) -> None:
                try:
                    for _ in range(20):
                        with manager.ephemeral_overrides(
                            model_id, {"temperature": target_temp}
                        ):
                            assert (
                                manager.get_settings(model_id).temperature
                                == target_temp
                            )
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=worker, args=(f"m{i}", i / 10))
                for i in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == []
            # All override stacks released.
            assert manager._overrides == {}
