# SPDX-License-Identifier: Apache-2.0
"""Same-family self-heal for stale routing target slots.

When an operator requantizes a model (e.g. ``Ornith-1.0-35B-oQ6e`` ->
``Ornith-1.0-35B-oQ8e``), the old id drops off the roster and any
``routing.targets``/``table_dispatch.default_target``/``policy.fail_open_target``
slot naming it rots. This module answers two narrow questions: given a stale
id and the live roster, is there exactly one roster id that is obviously
"the same model, different quant" (`heal_target`); and, separately, which
roster ids are even eligible to be picked (`visible_roster_ids`) -- a heal
must never land on a model the operator has hidden or a spec-decode helper,
since (unlike a named slot) a heal isn't an explicit per-model opt-in. Both
say so by returning None/excluding rather than guessing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Quant-*level* suffix tokens: discarding these is the whole point of a heal
# (a requantize is exactly "same model, different token here"). Stripping one
# of these sets `is_quantized` -- see `_decompose`. Kept in rough sync with
# omlx.oq.resolve_output_name's suffix vocabulary by hand -- review both when
# either's vocabulary changes; nothing enforces the sync.
_QUANT_LEVEL_RE = re.compile(r"-(oq[\d.]+e?|q\d+(?:_\d+)?|[0-9]+[_-]?bit)$")

# Compute-dtype tokens: NOT a quant level. `oq.resolve_output_name` emits
# these *alongside* a quant-level token (e.g. `Ornith-...-oQ8e-fp16` is an
# oQ8-quantized model with an fp16 compute dtype) -- so a dtype token alone
# does not make an id "quantized", but it also isn't load-bearing on its own
# and is discarded like a quant token (not compared).
_DTYPE_RE = re.compile(r"-(fp\d+|bf\d+)$")

# Load-bearing *artifact* flags: NOT quant levels, and differing on one of
# these is not what a heal is for. `mtp` marks a preserved next-token-predict
# head (a different checkpoint shape, not a precision bump); `unquantized`
# marks a dequantized-to-full-precision checkpoint kept alongside its
# genuinely quantized siblings (roughly 2x the RSS of the same stem's
# quantized forms). Stripped from the stem like quant tokens (so
# `family_stem` still recognizes the shared base name), but collected and
# compared separately: `heal_target` requires the full flag set to match, not
# just the stem, or a heal could swap a 4-bit checkpoint for its ~2x-RSS
# dequantized twin.
_FLAG_TOKEN_RE = re.compile(r"-(mtp|unquantized)$")

# (stem, flags, is_quantized). `is_quantized` is True iff at least one
# quant-*level* token (not a bare dtype token) was stripped -- see
# `_decompose`. Two ids are healable only if all three match; in particular
# a bare/full-precision checkpoint (`is_quantized=False`) never heals to or
# from a quantized sibling, no matter how the stem/flags line up.
_Identity = tuple[str, frozenset[str], bool]


def _decompose(model_id: str) -> _Identity:
    """(stem, flags, is_quantized) identity for `family_stem`/`heal_target`.

    Strips an ``owner--repo`` prefix (the ``models--org--repo`` HF-cache
    convention decoded by ``model_discovery._decode_hf_cache_model_id``),
    lowercases, then repeatedly strips one trailing quant-level, dtype, or
    flag token. Quant-level and dtype tokens are discarded (dtype also
    doesn't affect `is_quantized` -- only a quant-level token does); flag
    tokens are collected into the returned set.
    """
    name = model_id
    if "--" in name:
        _, _, name = name.partition("--")
    name = name.lower()
    flags: set[str] = set()
    is_quantized = False
    while True:
        m = _QUANT_LEVEL_RE.search(name)
        if m:
            name = name[: m.start()]
            is_quantized = True
            continue
        m = _DTYPE_RE.search(name)
        if m:
            name = name[: m.start()]
            continue
        m = _FLAG_TOKEN_RE.search(name)
        if m:
            flags.add(m.group(1))
            name = name[: m.start()]
            continue
        break
    return name, frozenset(flags), is_quantized


def family_stem(model_id: str) -> str:
    """Normalize a model id to its family stem for cross-quant comparison.

    Strips an owner prefix and all trailing quant/dtype/flag tokens (see
    `_decompose`), including `mtp`/`unquantized`. Two ids with the same stem
    are the same base model -- but that alone does not mean a heal between
    them is safe: `mtp`/`unquantized` differences, and whether either side
    is even a quantized artifact at all, are stripped from the stem yet
    still block a heal (`heal_target` checks them separately). Use
    `heal_target` for a healing decision; use this only to recognize "same
    base model" for reporting/comparison.
    """
    stem, _, _ = _decompose(model_id)
    return stem


def heal_target(stale_id: str, roster_ids: Iterable[str]) -> str | None:
    """Find the unique roster id that is `stale_id`'s same-family sibling.

    A roster id qualifies only if its family stem, its set of load-bearing
    flags (`mtp`, `unquantized`), AND its quantized-or-not status all match
    `stale_id`'s -- a heal is a requantize, not a swap to a different
    artifact (a preserved-MTP head, a dequantized-to-full-precision
    checkpoint at ~2x the RSS, or a bare full-precision base model). In
    particular, a bare/full-precision id (no quant-level token at all, e.g.
    a plain `bf16`/`fp16` checkpoint or a base model with no suffix at all)
    never heals to or from a quantized sibling: refused before the roster is
    even scanned. Returns None (never guesses) when zero or more than one
    roster id qualifies -- an ambiguous match is worse than no heal.
    """
    identity = _decompose(stale_id)
    if not identity[2]:  # stale isn't a quantized artifact -- refuse outright
        return None
    matches = {
        roster_id
        for roster_id in roster_ids
        if roster_id != stale_id and _decompose(roster_id) == identity
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


def visible_roster_ids(
    models: list[dict[str, Any]],
    settings_manager: Any | None,
    hide_helpers: bool,
) -> list[str]:
    """Chat-visible roster ids, for use as `heal_target`'s candidate source.

    Mirrors the visibility predicate `server.py`'s `/v1/models` endpoint
    applies (per-model `is_hidden`, plus the global `hide_helpers` toggle
    for models flagged `is_helper` or referenced by another model's
    speculative-decode settings). A heal must never reach a model the
    operator has hidden or a spec-decode companion: naming a model in a
    routing slot is itself the opt-in (docs/ROUTING.md), but a heal *picks*
    the model, so it can only pick from what's already visible.

    Known, accepted limitation: this does not filter by `model_type` (e.g.
    embedding/reranker/ASR models are technically heal candidates if
    visible). Left as-is because it faithfully mirrors `/v1/models`, and in
    practice a chat target's stem-and-flag sibling is essentially always
    another chat model -- an embedding/reranker/ASR match would require an
    id collision that doesn't occur naturally. Revisit if that assumption
    ever breaks.

    `models` is `EnginePool.get_status()["models"]` (or anything shaped like
    it: dicts with at least "id", and optionally "is_helper", "model_path",
    "source_repo_id"). `settings_manager` is a `ModelSettingsManager` (or
    None, meaning no per-model settings are known -- nothing is hidden).
    Pure otherwise: takes no pool/settings dependency beyond what's passed.
    """
    referenced_drafts: set[str] = set()
    if hide_helpers and settings_manager is not None:
        for ms in settings_manager.get_all_settings().values():
            for ref in (
                getattr(ms, "specprefill_draft_model", None),
                getattr(ms, "dflash_draft_model", None),
                getattr(ms, "vlm_mtp_draft_model", None),
            ):
                if ref:
                    referenced_drafts.add(ref)

    ids: list[str] = []
    for m in models:
        model_id = m["id"]
        ms = settings_manager.get_settings(model_id) if settings_manager else None
        is_hidden = ms is not None and ms.is_hidden
        is_hidden_helper = hide_helpers and (
            m.get("is_helper")
            or model_id in referenced_drafts
            or m.get("model_path") in referenced_drafts
            or m.get("source_repo_id") in referenced_drafts
        )
        if is_hidden or is_hidden_helper:
            continue
        ids.append(model_id)
    return ids
