# SPDX-License-Identifier: Apache-2.0
"""Fork: off-by-default SSD cache janitor.

Lives in a fork-owned file rather than appended to
tests/test_paged_ssd_cache.py so upstream can edit that file freely.
"""

import time
from pathlib import Path

from omlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheManager,
)
from omlx.settings import GlobalSettings


class TestSSDJanitor:
    """Tests for the off-by-default SSD janitor (reclaims stale
    ``_incompatible_index`` blocks that reactive size-based eviction never
    touches when the compatible cache alone stays under budget)."""

    @staticmethod
    def _add_incompatible_block(
        manager: PagedSSDCacheManager, tmp_path: Path, tag: bytes
    ) -> PagedSSDBlockMetadata:
        """Add a block to ``_index`` then move it to ``_incompatible_index``
        via ``forget_block`` — mirrors ``test_forget_block_tracks_file_as_incompatible``.
        """
        block_hash = tag * 32
        file_path = tmp_path / "ssd_cache" / "0" / f"{tag.hex()}.safetensors"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x" * 100)
        metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=file_path,
            file_size=100,
            token_count=1,
            created_at=1.0,
            last_access=1.0,
            num_layers=1,
            model_name="test-model",
        )
        manager._index.add(metadata)
        assert manager.forget_block(block_hash) is True
        return metadata

    def test_janitor_disabled_by_default_no_thread(self, tmp_path: Path):
        """ssd_janitor_enabled defaults to False: no thread starts, stats key
        exists but stays at 0."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )
        try:
            assert manager._janitor_thread is None
            time.sleep(0.05)
            assert manager.get_stats_dict()["ssd_janitor_unlinks"] == 0
        finally:
            manager.close()

    def test_sweep_evicts_incompatible_blocks_only(self, tmp_path: Path):
        """Direct sweep call reclaims incompatible blocks, leaves the
        compatible index untouched, unlinks files, and updates stats."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            ssd_janitor_enabled=True,
            ssd_janitor_max_unlinks_per_sweep=256,
        )
        try:
            incompatible_meta = self._add_incompatible_block(manager, tmp_path, b"\x01")

            # A live compatible block that must survive the sweep.
            compatible_hash = b"\x02" * 32
            compatible_path = tmp_path / "ssd_cache" / "0" / "compat.safetensors"
            compatible_path.write_bytes(b"y" * 50)
            compatible_meta = PagedSSDBlockMetadata(
                block_hash=compatible_hash,
                file_path=compatible_path,
                file_size=50,
                token_count=1,
                created_at=1.0,
                last_access=1.0,
                num_layers=1,
                model_name="test-model",
            )
            manager._index.add(compatible_meta)

            assert incompatible_meta.file_path.exists()

            count = manager._janitor_sweep_once()

            assert count == 1
            assert not manager._incompatible_index.contains(b"\x01" * 32)
            assert not incompatible_meta.file_path.exists()

            # Compatible index untouched.
            assert manager._index.contains(compatible_hash)
            assert compatible_path.exists()

            assert manager.get_stats_dict()["ssd_janitor_unlinks"] == 1
        finally:
            manager.close()

    def test_sweep_bounded_by_max_unlinks_per_sweep(self, tmp_path: Path):
        """One sweep reclaims at most ``ssd_janitor_max_unlinks_per_sweep``
        blocks even when more incompatible blocks exist."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            ssd_janitor_enabled=True,
            ssd_janitor_max_unlinks_per_sweep=2,
        )
        try:
            for i in range(5):
                self._add_incompatible_block(manager, tmp_path, bytes([i + 1]))

            assert manager._incompatible_index.count == 5

            count = manager._janitor_sweep_once()

            assert count == 2
            assert manager._incompatible_index.count == 3
            assert manager.get_stats_dict()["ssd_janitor_unlinks"] == 2
        finally:
            manager.close()

    def test_janitor_thread_lifecycle_stops_on_close(self, tmp_path: Path):
        """Enabling the janitor starts a background thread that stops
        promptly on close(), with no hang."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            ssd_janitor_enabled=True,
            ssd_janitor_interval_s=1,
        )
        try:
            assert manager._janitor_thread is not None
            assert manager._janitor_thread.is_alive()
        finally:
            manager.close()

        assert not manager._janitor_thread.is_alive()


def test_to_scheduler_config_ssd_janitor():
    """ssd_janitor_* settings pass through to SchedulerConfig."""
    settings = GlobalSettings()
    settings.cache.ssd_janitor_enabled = True
    settings.cache.ssd_janitor_interval_s = 60
    settings.cache.ssd_janitor_max_unlinks_per_sweep = 10

    scheduler_config = settings.to_scheduler_config()
    assert scheduler_config.ssd_janitor_enabled is True
    assert scheduler_config.ssd_janitor_interval_s == 60
    assert scheduler_config.ssd_janitor_max_unlinks_per_sweep == 10
