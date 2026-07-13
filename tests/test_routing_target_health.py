# SPDX-License-Identifier: Apache-2.0
"""Tests for routing-target roster-validity + graceful substitution.

Covers RoutingService._finalize_target (the ladder/gating matrix),
validate_targets(), and the end-to-end route_chat_request/_fail_open
integration (modality no-downgrade guarantee, fail-open substitution,
telemetry/header surfacing). Classifier/profiler stubbed throughout --
nothing loads a real model.
"""

from omlx.routing.service import RoutingService
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


# --- _finalize_target: ladder/gating matrix -------------------------------


def test_finalize_target_valid_passthrough(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for({"good-model"}), lambda: True)

    target, invalid = service._finalize_target("good-model")

    assert target == "good-model"
    assert invalid is None


def test_finalize_target_substitutes_default_target_first(tmp_path):
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = "gen-model"
    settings.targets = {
        "small": "small-model",
        "big": "big-model",
        "vision": "vision-model",
    }
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"gen-model", "big-model", "small-model", "vision-model"}),
        lambda: True,
    )

    target, invalid = service._finalize_target("stale-model")

    assert target == "gen-model"  # default_target wins the ladder
    assert invalid == "stale-model"


def test_finalize_target_ladder_falls_to_big_over_small(tmp_path):
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = None
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"big-model", "small-model"}), lambda: True
    )

    target, invalid = service._finalize_target("stale-model")

    assert target == "big-model"
    assert invalid == "stale-model"


def test_finalize_target_passthrough_when_model_fallback_off(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"big-model", "small-model"}), lambda: False
    )

    target, invalid = service._finalize_target("stale-model")

    assert target == "stale-model"
    assert invalid == "stale-model"


def test_finalize_target_on_but_no_valid_candidate(tmp_path):
    settings = make_settings(tmp_path)
    settings.table_dispatch.default_target = None
    settings.targets = {"small": "stale-small", "big": "stale-big"}
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for(set()), lambda: True)

    target, invalid = service._finalize_target("stale-model")

    assert target == "stale-model"
    assert invalid == "stale-model"


def test_finalize_target_substitute_false_never_swaps(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"big-model", "small-model"}), lambda: True
    )

    target, invalid = service._finalize_target("stale-vision-model", substitute=False)

    assert target == "stale-vision-model"
    assert invalid == "stale-vision-model"


def test_finalize_target_unwired_resolver_fails_open(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)
    # set_validity_sources never called.

    target, invalid = service._finalize_target("looks-invalid-but-unchecked")

    assert target == "looks-invalid-but-unchecked"
    assert invalid is None


def test_finalize_target_raising_valid_getter_fails_open(tmp_path):
    settings = make_settings(tmp_path)
    service = RoutingService(settings)

    def _raising_getter(model_id: str) -> bool:
        raise RuntimeError("roster lookup blew up")

    service.set_validity_sources(_raising_getter, lambda: True)

    target, invalid = service._finalize_target("some-model")

    assert target == "some-model"
    assert invalid is None


# --- validate_targets() ---------------------------------------------------


def test_validate_targets_unset_slot_is_none(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model"}  # big/vision unset
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for({"small-model"}), lambda: True)

    report = service.validate_targets()

    assert report["big"] == {"id": None, "resolves": None}
    assert report["vision"] == {"id": None, "resolves": None}


def test_validate_targets_valid_and_stale_slots(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "stale-big-model"}
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for({"small-model"}), lambda: True)

    report = service.validate_targets()

    assert report["small"] == {"id": "small-model", "resolves": True}
    assert report["big"] == {"id": "stale-big-model", "resolves": False}


def test_validate_targets_fail_open_target_resolves_through_targets_map(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "big-model"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model", "big-model"}), lambda: True
    )

    report = service.validate_targets()

    assert report["fail_open_target"] == {"id": "big-model", "resolves": True}


# --- tokenizer_target() ----------------------------------------------------


def test_tokenizer_target_resolves_through_targets_map(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "big-model"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    service.set_validity_sources(
        valid_getter_for({"small-model", "big-model"}), lambda: True
    )

    assert service.tokenizer_target() == "big-model"


def test_tokenizer_target_substitutes_stale_fail_open_slot(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "stale-big-model"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for({"small-model"}), lambda: True)

    assert service.tokenizer_target() == "small-model"


def test_tokenizer_target_passthrough_when_fallback_off(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "small-model", "big": "stale-big-model"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    service.set_validity_sources(valid_getter_for({"small-model"}), lambda: False)

    assert service.tokenizer_target() == "stale-big-model"


# --- end-to-end: modality no-downgrade guarantee --------------------------


async def test_modality_path_invalid_vision_target_never_substitutes(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {
        "small": "small-model",
        "big": "big-model",
        "vision": "stale-vision-model",
    }
    service = RoutingService(settings)
    # Only text targets are valid; vision is stale. Fallback ON, but the
    # modality path must never let a text generalist swallow an image.
    service.set_validity_sources(
        valid_getter_for({"small-model", "big-model"}), lambda: True
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
        request_id="req-vision-1",
        stream=False,
    )

    assert decision.target == "stale-vision-model"
    assert decision.invalid_target == "stale-vision-model"
    assert decision.rule_fired == "shape:vision"
    await service.close()


# --- end-to-end: fail-open substitution + telemetry/header surfacing -----


async def test_fail_open_substitutes_invalid_target_and_records_telemetry(tmp_path):
    settings = make_settings(tmp_path)
    settings.targets = {"small": "valid-small", "big": "stale-big-model"}
    settings.policy.fail_open_target = "big"
    service = RoutingService(settings)
    # No engine getter configured -> classify raises -> _fail_open("error").
    service.set_validity_sources(valid_getter_for({"valid-small"}), lambda: True)

    decision = await service.route_chat_request(
        messages=[{"role": "user", "content": "hello"}],
        has_tools=False,
        request_id="req-failopen-1",
        stream=False,
    )

    assert decision.rule_fired == "fail_open:error"
    assert decision.target == "valid-small"
    assert decision.invalid_target == "stale-big-model"
    assert "invalid_target=stale-big-model" in decision.header_value

    recent = service.recent_decisions(limit=10)
    row = next(r for r in recent if r["request_id"] == "req-failopen-1")
    assert row["invalid_target"] == "stale-big-model"
    assert row["target"] == "valid-small"

    await service.close()
