# SPDX-License-Identifier: Apache-2.0
"""Tests for same-family routing-target self-heal (target_heal + service wiring).

Covers omlx.routing.target_heal.family_stem/heal_target/visible_roster_ids
directly, plus the RoutingService integration points: _finalize_target
healing ahead of substitution, validate_targets()'s healed_to reporting, and
the RouteDecision/telemetry/header surfacing of healed_from. Classifier/
profiler stubbed throughout -- nothing loads a real model. Table-driven cases
use real ids from ~/.omlx/suitability.json (this machine + the Studio,
2026-07-20) rather than invented names, where a real id demonstrates the case.
"""

from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.routing.service import RoutingService
from omlx.routing.target_heal import family_stem, heal_target, visible_roster_ids
from omlx.settings import RoutingSettings


def make_settings(tmp_path) -> RoutingSettings:
    settings = RoutingSettings()
    settings.telemetry.enabled = True
    settings.telemetry.path = str(tmp_path / "routing_decisions.jsonl")
    return settings


def valid_getter_for(valid_ids: set[str]):
    def _getter(model_id: str) -> bool:
        return model_id in valid_ids

    return _getter


# --- family_stem / heal_target: unit -------------------------------------


def test_family_stem_strips_oq_suffix():
    assert family_stem("Ornith-1.0-35B-oQ6e") == family_stem("Ornith-1.0-35B-oQ8e")


def test_family_stem_strips_oq_mtp_combo():
    assert family_stem("Ornith-1.0-35B-oQ4e-mtp") == family_stem("Ornith-1.0-35B-oQ8e")


def test_family_stem_strips_unquantized_chain():
    assert family_stem("Ornith-1.0-35B-q4_0-unquantized-oQ8e") == family_stem(
        "Ornith-1.0-35B-oQ4e"
    )


def test_family_stem_strips_nbit_suffix():
    assert family_stem("Qwen2.5-7B-Instruct-4bit") == family_stem(
        "Qwen2.5-7B-Instruct-oQ6e"
    )


def test_family_stem_strips_owner_prefix():
    assert family_stem("mlx-community--Qwen2.5-7B-Instruct-4bit") == family_stem(
        "Qwen2.5-7B-Instruct-oQ6e"
    )


def test_family_stem_owner_prefix_independent_of_owner():
    # Same repo name, different owner prefixes -> same stem (owner isn't
    # part of the family; only the model name + stripped suffixes are).
    assert family_stem("mlx-community--Qwen2.5-7B-Instruct-4bit") == family_stem(
        "other-org--Qwen2.5-7B-Instruct-oQ6e"
    )


def test_family_stem_false_family_negative_different_size():
    # Same base name, different parameter count -> must NOT collapse to the
    # same stem. This is the case a wrong heal would misroute a small model
    # to a much bigger (or smaller) one.
    assert family_stem("gemma-4-12B-it-qat-oQ6e") != family_stem(
        "gemma-4-31B-it-qat-oQ8e"
    )


def test_heal_target_unique_match():
    roster = ["Ornith-1.0-35B-oQ8e", "some-other-model-oQ6e"]
    assert heal_target("Ornith-1.0-35B-oQ6e", roster) == "Ornith-1.0-35B-oQ8e"


def test_heal_target_mtp_suffix_form():
    # Real Studio case (2026-07-20 verifier repro): both sides carry the
    # -mtp flag (a preserved next-token-predict head), so the quant token
    # behind it is still reachable and the heal fires.
    roster = ["Qwen3.6-35B-A3B-oQ8e-mtp"]
    assert heal_target("Qwen3.6-35B-A3B-oQ6e-mtp", roster) == "Qwen3.6-35B-A3B-oQ8e-mtp"


def test_heal_target_mtp_suffix_form_second_real_pair():
    # Same shape, different real model (~/.omlx/suitability.json).
    roster = ["ThinkingCap-Qwen3.6-27B-oQ8e-mtp"]
    assert (
        heal_target("ThinkingCap-Qwen3.6-27B-oQ6e-mtp", roster)
        == "ThinkingCap-Qwen3.6-27B-oQ8e-mtp"
    )


def test_heal_target_mtp_flag_blocks_cross_artifact_heal():
    # mtp is a load-bearing artifact flag, not a quant level: a heal must
    # not trade a preserved-MTP-head checkpoint for a plain one even though
    # the quant tokens alone would otherwise line up.
    roster = ["Ornith-1.0-35B-oQ8e"]
    assert heal_target("Ornith-1.0-35B-oQ4e-mtp", roster) is None


def test_heal_target_unquantized_flag_blocks_heal():
    # Real corpus pair (~/.omlx/suitability.json). The trailing "-assistant"
    # token isn't in our vocabulary, so it blocks the strip chain on both
    # sides before "unquantized"/"q4_0" are even reached -- the stems come
    # out different ("...-qat-assistant" vs "...-qat-q4_0-unquantized-
    # assistant"), so this refuses on stem grounds alone. See
    # `test_heal_target_unquantized_flag_blocks_heal_same_stem` below for the
    # case that isolates the flag axis specifically.
    roster = ["gemma-4-31B-it-qat-q4_0-unquantized-assistant"]
    assert heal_target("gemma-4-31B-it-qat-assistant-bf16", roster) is None


def test_heal_target_unquantized_flag_blocks_heal_same_stem():
    # Same real stem ("gemma-4-31b-it-qat"), both genuinely quantized
    # (is_quantized=True on both sides) -- the only difference is the
    # `unquantized` flag (a full-precision checkpoint kept alongside its
    # quantized siblings, ~2x the RSS). Isolates the flag guard: must refuse
    # even though stem and is_quantized both match.
    roster = ["gemma-4-31B-it-qat-q4_0-unquantized-oQ6e"]
    assert heal_target("gemma-4-31B-it-qat-oQ4e", roster) is None


def test_heal_target_unquantized_siblings_still_heal_each_other():
    # Both sides tagged `unquantized` (same flag set), both quantized --
    # only the quant token differs, which is exactly what a heal is for.
    roster = ["gemma-4-31B-it-qat-q4_0-unquantized-oQ8e"]
    assert (
        heal_target("gemma-4-31B-it-qat-q4_0-unquantized-oQ4e", roster)
        == "gemma-4-31B-it-qat-q4_0-unquantized-oQ8e"
    )


def test_heal_target_dtype_alone_does_not_count_as_quantized():
    # Round-2 regression (verifier repro, 2026-07-20): a bare fp16/bf16
    # checkpoint is a full-precision artifact, not a quant level -- adding
    # fp/bf to the discardable-token vocabulary must not make it heal-
    # eligible against a genuinely quantized sibling. Real corpus pair:
    # "Agents-A1-bf16" / "Agents-A1-oQ8e" co-exist in ~/.omlx/suitability.json.
    roster = ["Agents-A1-bf16"]
    assert heal_target("Agents-A1-oQ8e", roster) is None
    roster = ["Agents-A1-oQ8e"]
    assert heal_target("Agents-A1-bf16", roster) is None


def test_heal_target_dflash_fp16_bare_does_not_count_as_quantized():
    # Same bug class, second real corpus pair: "Qwen3.5-9B-DFlash" (bare)
    # vs "Qwen3.5-9B-DFlash-oQ8-fp16" (oQ8-quantized with an fp16 compute
    # dtype). The bare id carries no quant-level token at all.
    roster = ["Qwen3.5-9B-DFlash"]
    assert heal_target("Qwen3.5-9B-DFlash-oQ8-fp16", roster) is None


def test_heal_target_quant_plus_dtype_still_heals():
    # The fix is a precision-class guard, not a revert of fp/bf: a model
    # that carries BOTH a quant-level token and a dtype token (the shape
    # omlx.oq.resolve_output_name actually emits, e.g. "...-oQ8e-fp16") is
    # still genuinely quantized and must still heal against another
    # quantized sibling, dtype token discarded either way.
    roster = ["Ornith-1.0-35B-oQ8e-fp16"]
    assert heal_target("Ornith-1.0-35B-oQ6e-fp16", roster) == "Ornith-1.0-35B-oQ8e-fp16"


def test_heal_target_bare_base_never_heals_to_quantized_sibling():
    # Pre-existing hole (present since round 1, flagged separately): a bare
    # full-precision base model must never be treated as a quant sibling of
    # its own quantized form. Real corpus pairs
    # (~/.omlx/suitability.json, both machines):
    assert (
        heal_target("ThinkingCap-Qwen3.6-27B-oQ8", ["ThinkingCap-Qwen3.6-27B"]) is None
    )
    assert (
        heal_target("ThinkingCap-Qwen3.6-27B", ["ThinkingCap-Qwen3.6-27B-oQ8"]) is None
    )
    assert (
        heal_target(
            "MiniCPM5-1B-Claude-Opus-Fable5-Thinking-oQ8e",
            ["MiniCPM5-1B-Claude-Opus-Fable5-Thinking"],
        )
        is None
    )
    assert (
        heal_target(
            "Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic-oQ8",
            ["Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic"],
        )
        is None
    )


def test_heal_target_nbit_form():
    roster = ["Qwen2.5-7B-Instruct-oQ6e"]
    assert heal_target("Qwen2.5-7B-Instruct-4bit", roster) == "Qwen2.5-7B-Instruct-oQ6e"


def test_heal_target_owner_prefix_form():
    roster = ["mlx-community--Qwen2.5-7B-Instruct-oQ6e"]
    assert (
        heal_target("mlx-community--Qwen2.5-7B-Instruct-4bit", roster)
        == "mlx-community--Qwen2.5-7B-Instruct-oQ6e"
    )


def test_heal_target_false_family_negative():
    roster = ["gemma-4-31B-it-qat-oQ8e"]
    assert heal_target("gemma-4-12B-it-qat-oQ6e", roster) is None


def test_heal_target_ambiguous_match_returns_none():
    roster = ["Ornith-1.0-35B-oQ6e", "Ornith-1.0-35B-oQ8e"]
    assert heal_target("Ornith-1.0-35B-oQ4e", roster) is None


def test_heal_target_no_match_returns_none():
    roster = ["completely-unrelated-model-oQ6e"]
    assert heal_target("Ornith-1.0-35B-oQ4e", roster) is None


def test_heal_target_ignores_stale_id_if_present_in_roster():
    # Defensive: even if the stale id itself is (somehow) in the roster
    # list, it's excluded from its own match set.
    roster = ["Ornith-1.0-35B-oQ6e", "Ornith-1.0-35B-oQ6e"]
    assert heal_target("Ornith-1.0-35B-oQ6e", roster) is None


# --- visible_roster_ids: heal candidate source -----------------------------


def _model(model_id: str, **extra) -> dict:
    return {"id": model_id, "model_path": f"/models/{model_id}", **extra}


def test_visible_roster_ids_excludes_operator_hidden(tmp_path):
    # Verifier repro: an embedding model's quant sibling must not be a heal
    # candidate when the operator has hidden it (the common setup for
    # utility models that never belong in a chat picker).
    models = [_model("Qwen3-Embedding-8B-4bit"), _model("Qwen3-Embedding-8B-8bit")]
    mgr = ModelSettingsManager(base_path=tmp_path)
    mgr.set_settings("Qwen3-Embedding-8B-8bit", ModelSettings(is_hidden=True))

    ids = visible_roster_ids(models, mgr, hide_helpers=False)

    assert ids == ["Qwen3-Embedding-8B-4bit"]


def test_visible_roster_ids_excludes_intrinsic_helper(tmp_path):
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    mgr = ModelSettingsManager(base_path=tmp_path)

    assert visible_roster_ids(models, mgr, hide_helpers=True) == ["chat-a"]


def test_visible_roster_ids_keeps_helper_when_toggle_off(tmp_path):
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    mgr = ModelSettingsManager(base_path=tmp_path)

    ids = visible_roster_ids(models, mgr, hide_helpers=False)

    assert "drafter" in ids


def test_visible_roster_ids_excludes_referenced_draft(tmp_path):
    models = [_model("chat-a"), _model("draft-llm")]
    mgr = ModelSettingsManager(base_path=tmp_path)
    mgr.set_settings("chat-a", ModelSettings(dflash_draft_model="/models/draft-llm"))

    ids = visible_roster_ids(models, mgr, hide_helpers=True)

    assert ids == ["chat-a"]


def test_visible_roster_ids_no_settings_manager_hides_nothing():
    models = [_model("chat-a"), _model("chat-b")]

    assert visible_roster_ids(models, None, hide_helpers=True) == [
        "chat-a",
        "chat-b",
    ]


def test_visible_roster_ids_feeds_heal_target_excludes_hidden_sibling(tmp_path):
    # End-to-end of the fix: a heal candidate that resolves stem+flags but
    # is hidden never reaches heal_target's roster argument in the first
    # place.
    models = [_model("Qwen3-Embedding-8B-4bit"), _model("Qwen3-Embedding-8B-8bit")]
    mgr = ModelSettingsManager(base_path=tmp_path)
    mgr.set_settings("Qwen3-Embedding-8B-8bit", ModelSettings(is_hidden=True))

    roster = visible_roster_ids(models, mgr, hide_helpers=False)

    assert heal_target("Qwen3-Embedding-8B-4bit", roster) is None


# --- RoutingService._finalize_target: heal wiring -------------------------


def test_finalize_target_heals_before_substitution(tmp_path):
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = "gen-model"
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for(
            {"gen-model", "big-model", "small-model", "Ornith-1.0-35B-oQ8e"}
        ),
        lambda: True,
        lambda: ["Ornith-1.0-35B-oQ8e", "gen-model", "big-model", "small-model"],
    )

    target, invalid, healed_from = service._finalize_target("Ornith-1.0-35B-oQ6e")

    # Healed to the same-family sibling, not the cross-model default_target.
    assert target == "Ornith-1.0-35B-oQ8e"
    assert invalid is None
    assert healed_from == "Ornith-1.0-35B-oQ6e"


def test_finalize_target_heal_runs_even_with_fallback_off(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model", "big-model", "Ornith-1.0-35B-oQ8e"}),
        lambda: False,  # model_fallback off -- heal must still fire
        lambda: ["Ornith-1.0-35B-oQ8e"],
    )

    target, invalid, healed_from = service._finalize_target("Ornith-1.0-35B-oQ6e")

    assert target == "Ornith-1.0-35B-oQ8e"
    assert invalid is None
    assert healed_from == "Ornith-1.0-35B-oQ6e"


def test_finalize_target_substitute_false_still_heals(tmp_path):
    """Vision path (substitute=False): can never cross-substitute, but a
    same-family heal is not a cross-substitution and must still apply."""
    settings = make_settings(tmp_path)
    settings.targets = {"vision": "VisionModel-oQ6e"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"VisionModel-oQ8e"}),
        lambda: True,
        lambda: ["VisionModel-oQ8e"],
    )

    target, invalid, healed_from = service._finalize_target(
        "VisionModel-oQ6e", substitute=False
    )

    assert target == "VisionModel-oQ8e"
    assert invalid is None
    assert healed_from == "VisionModel-oQ6e"


def test_finalize_target_falls_back_to_substitution_when_heal_ambiguous(tmp_path):
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = "gen-model"
    settings.targets = {"small": "small-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"gen-model", "small-model", "Model-oQ6e", "Model-oQ8e"}),
        lambda: True,
        lambda: ["Model-oQ6e", "Model-oQ8e", "gen-model", "small-model"],
    )

    # Two same-family siblings -> heal_target refuses to guess -> falls
    # through to the ordinary substitution ladder.
    target, invalid, healed_from = service._finalize_target("Model-oQ4e")

    assert target == "gen-model"
    assert invalid == "Model-oQ4e"
    assert healed_from is None


def test_finalize_target_roster_getter_none_is_byte_identical(tmp_path):
    """No roster_getter wired (default None) -> heal disabled -> identical
    to pre-heal behavior: substitute-or-passthrough, healed_from always
    None."""
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = "gen-model"
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"gen-model", "big-model", "small-model"}), lambda: True
    )

    target, invalid, healed_from = service._finalize_target("Ornith-1.0-35B-oQ6e")

    assert target == "gen-model"  # ordinary substitution ladder, unaffected
    assert invalid == "Ornith-1.0-35B-oQ6e"
    assert healed_from is None


def test_finalize_target_roster_getter_raising_fails_open_no_heal(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model"}
    service = RoutingService(settings)

    def _raising_roster():
        raise RuntimeError("pool blew up")

    service.set_validity_sources(
        valid_getter_for({"small-model"}), lambda: True, _raising_roster
    )

    target, invalid, healed_from = service._finalize_target("stale-model")

    assert target == "small-model"
    assert invalid == "stale-model"
    assert healed_from is None


# --- validate_targets(): healed_to reporting -------------------------------


def test_validate_targets_reports_healed_to_for_stale_slot(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "Ornith-1.0-35B-oQ6e"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model", "Ornith-1.0-35B-oQ8e"}),
        lambda: True,
        lambda: ["small-model", "Ornith-1.0-35B-oQ8e"],
    )

    report = service.validate_targets()

    # resolves stays False -- the stale id itself still doesn't resolve --
    # but healed_to surfaces the sibling that a heal would resolve it to.
    assert report["big"]["resolves"] is False
    assert report["big"]["healed_to"] == "Ornith-1.0-35B-oQ8e"
    assert report["small"]["resolves"] is True
    assert report["small"]["healed_to"] is None


def test_validate_targets_healed_to_none_when_unhealable(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "totally-unrelated-stale"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model"}),
        lambda: True,
        lambda: ["small-model"],
    )

    report = service.validate_targets()

    assert report["big"]["resolves"] is False
    assert report["big"]["healed_to"] is None


def test_validate_targets_no_roster_getter_healed_to_always_none(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"big": "Ornith-1.0-35B-oQ6e"}
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for(set()), lambda: True)

    report = service.validate_targets()

    assert report["big"]["resolves"] is False
    assert report["big"]["healed_to"] is None


# --- end-to-end: healed_from on RouteDecision + telemetry -----------------


async def test_fail_open_heals_and_records_healed_from(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "valid-small", "big": "Ornith-1.0-35B-oQ6e"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    # No engine getter configured -> classify raises -> _fail_open("error").
    service.set_validity_sources(
        valid_getter_for({"valid-small", "Ornith-1.0-35B-oQ8e"}),
        lambda: True,
        lambda: ["valid-small", "Ornith-1.0-35B-oQ8e"],
    )

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="req-heal-1",
        stream=False,
    )

    assert decision.rule_fired == "fail_open:error"
    assert decision.target == "Ornith-1.0-35B-oQ8e"
    assert decision.invalid_target is None
    assert decision.healed_from == "Ornith-1.0-35B-oQ6e"
    assert "healed_from=Ornith-1.0-35B-oQ6e" in decision.header_value

    recent = service.recent_decisions(limit=10)
    row = next(r for r in recent if r["request_id"] == "req-heal-1")
    assert row["target"] == "Ornith-1.0-35B-oQ8e"
    assert row["invalid_target"] is None
    assert row["healed_from"] == "Ornith-1.0-35B-oQ6e"

    await service.close()


async def test_modality_path_heals_stale_vision_target(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {
        "small": "small-model",
        "big": "big-model",
        "vision": "VisionModel-oQ6e",
    }
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model", "big-model", "VisionModel-oQ8e"}),
        lambda: True,
        lambda: ["VisionModel-oQ8e"],
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "text": None}],
        }
    ]
    decision = await service.route_chat_request(
        messages=messages,
        has_tools=False,
        request_id="req-vision-heal-1",
        stream=False,
    )

    assert decision.target == "VisionModel-oQ8e"
    assert decision.invalid_target is None
    assert decision.healed_from == "VisionModel-oQ6e"
    assert decision.rule_fired == "shape:vision"
    await service.close()
