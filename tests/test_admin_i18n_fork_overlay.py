# SPDX-License-Identifier: Apache-2.0
"""Fork i18n strings live in an overlay, never in the upstream locale files.

This keeps ``omlx/admin/i18n/<lang>.json`` byte-identical to upstream so those
files never conflict when merging upstream in.
"""

import json
from pathlib import Path

import pytest

from omlx.admin.routes import _load_locale

I18N_DIR = Path(__file__).parent.parent / "omlx" / "admin" / "i18n"
LANGUAGES = ["en", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh-TW", "zh"]
# Key prefixes owned by this fork. Anything here belongs in i18n/fork/<lang>.json.
FORK_PREFIXES = ("suitability.", "settings.advanced.ssd_janitor_", "navbar.dropdown.suitability")


def _read(name: str) -> dict:
    return json.loads((I18N_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang", LANGUAGES)
def test_upstream_locale_has_no_fork_keys(lang):
    """Upstream locale files must stay free of fork strings."""
    offenders = [k for k in _read(lang) if k.startswith(FORK_PREFIXES)]
    assert offenders == [], f"{lang}.json carries fork keys: {offenders[:5]}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_overlay_covers_every_language(lang):
    """Every language ships the same fork key set as English."""
    assert set(_read(f"fork/{lang}")) == set(_read("fork/en"))


@pytest.mark.parametrize("lang", LANGUAGES)
def test_loaded_locale_merges_overlay(lang):
    """_load_locale returns upstream strings plus the fork overlay."""
    locale = _load_locale(lang)
    upstream, fork = _read(lang), _read(f"fork/{lang}")
    assert fork, "overlay should not be empty"
    for key, value in fork.items():
        assert locale[key] == value
    for key, value in upstream.items():
        assert locale[key] == value


def test_unknown_language_falls_back_to_english_with_overlay():
    locale = _load_locale("kl")
    assert locale["suitability.heading"] == _read("fork/en")["suitability.heading"]
    assert locale["settings.advanced.cache"] == _read("en")["settings.advanced.cache"]
