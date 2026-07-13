# SPDX-License-Identifier: Apache-2.0
"""Tests for M6.1 implicit outcome proxies:

detect_implicit_signal (pure) and RoutingService's content-hash join that
attributes a prior turn's implicit signal to its decision as source:"implicit"
feedback (omlx/routing/service.py).
"""

import json

from omlx.routing.service import (
    RouteDecision,
    RoutingService,
    detect_implicit_signal,
)
from omlx.settings import RoutingSettings


def make_service(tmp_path, implicit=True, approval=True, **overrides) -> RoutingService:
    data = {
        "enabled": True,
        "telemetry": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")},
        "implicit_feedback": {"enabled": implicit, "approval": approval},
    }
    data.update(overrides)
    return RoutingService(RoutingSettings.from_dict(data))


def decision(target="model-a", rule="small") -> RouteDecision:
    return RouteDecision(
        target=target,
        rule_fired=rule,
        override=None,
        features=None,
        raw_analysis=None,
        classify_ms=1.0,
    )


def user(text):
    return {"role": "user", "content": text}


def assistant(text="here you go"):
    return {"role": "assistant", "content": text}


def tool_error():
    return {
        "role": "user",
        "content": [{"type": "tool_result", "is_error": True, "content": "boom"}],
    }


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDetect:
    def test_no_assistant_turn_returns_none(self):
        assert detect_implicit_signal([user("write a function")]) is None

    def test_negation(self):
        msgs = [user("write it"), assistant(), user("No, that's wrong")]
        sig = detect_implicit_signal(msgs)
        assert sig is not None and sig.kind == "negation" and sig.score == 0.0

    def test_rephrase(self):
        msgs = [user("write it"), assistant(), user("try again please")]
        sig = detect_implicit_signal(msgs)
        assert sig is not None and sig.kind == "rephrase" and sig.score == 0.2

    def test_approval(self):
        msgs = [user("write it"), assistant(), user("thanks, that works")]
        sig = detect_implicit_signal(msgs)
        assert sig is not None and sig.kind == "approval" and sig.score == 1.0

    def test_tool_error_beats_text(self):
        msgs = [user("run it"), assistant("calling tool"), tool_error()]
        sig = detect_implicit_signal(msgs)
        assert sig is not None and sig.kind == "tool_error" and sig.score == 0.0

    def test_negation_beats_approval_in_same_turn(self):
        # "no," is a negation cue; dissatisfaction is checked first.
        msgs = [user("write it"), assistant(), user("no, thanks anyway")]
        sig = detect_implicit_signal(msgs)
        assert sig is not None and sig.kind == "negation"

    def test_neutral_followup_returns_none(self):
        msgs = [user("write it"), assistant(), user("what about the edge cases?")]
        assert detect_implicit_signal(msgs) is None

    def test_no_false_positive_on_embedded_no(self):
        # A plain "no" inside a sentence must not trip the punctuation-guarded
        # negation cues.
        msgs = [user("write it"), assistant(), user("I have no other questions")]
        assert detect_implicit_signal(msgs) is None

    def test_empty_newest_user_returns_none(self):
        msgs = [user("write it"), assistant(), {"role": "user", "content": ""}]
        assert detect_implicit_signal(msgs) is None

    def test_filler_prefix_still_matches(self):
        for lead in ("ok, that's wrong", "hmm, try again", "actually, that works"):
            msgs = [user("write it"), assistant(), user(lead)]
            assert detect_implicit_signal(msgs) is not None, lead

    def test_no_substring_false_positives(self):
        # Regression: cues must anchor to the start of the (filler-stripped)
        # message, not match mid-sentence. These are all benign task requests.
        benign = [
            "Please correct the incorrect assumption in my draft.",
            "Please redocument the public API surface for me.",
            "The perfect number 6 equals the sum of its divisors.",
            "That's an awesome idea, expand on it.",
            "Is it incorrect to assume the earth is flat?",
            "No thanks needed, just continue with part two.",
        ]
        for text in benign:
            msgs = [user("write it"), assistant(), user(text)]
            assert detect_implicit_signal(msgs) is None, text

    def test_malformed_content_returns_none(self):
        # detect must be exception-safe against non-string/odd content.
        for bad in (123, [{"type": "text", "text": 123}], {"k": "v"}, None):
            msgs = [user("write it"), assistant(), {"role": "user", "content": bad}]
            assert detect_implicit_signal(msgs) is None


class TestServiceAttribution:
    async def test_prior_decision_gets_implicit_feedback(self, tmp_path):
        svc = make_service(tmp_path)
        turn1 = [user("write a function")]
        svc._record_decision(decision(), "r1", "chat", False, turn1)
        turn2 = [user("write a function"), assistant(), user("no, that's wrong")]
        svc._record_decision(decision(), "r2", "chat", False, turn2)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert len(fb) == 1
        assert fb[0]["request_id"] == "r1"
        assert fb[0]["source"] == "implicit"
        assert fb[0]["score"] == 0.0
        assert fb[0]["label"] == "negation"
        assert "implicit" in fb[0]["tags"]

    async def test_tool_error_attributes_to_the_call_that_failed(self, tmp_path):
        # Realistic Anthropic agent flow: the tool_result rides in a role:"user"
        # message, so it must attribute to the decision that emitted the
        # tool_use (rA), not be lost.
        svc = make_service(tmp_path)
        svc._record_decision(decision(), "rA", "chat", False, [user("run the tests")])
        turn_b = [
            user("run the tests"),
            {"role": "assistant", "content": [{"type": "tool_use", "name": "bash"}]},
            tool_error(),
        ]
        svc._record_decision(decision(), "rB", "chat", False, turn_b)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert len(fb) == 1
        assert fb[0]["request_id"] == "rA"
        assert fb[0]["label"] == "tool_error"

    async def test_live_ring_attach_to_prior(self, tmp_path):
        svc = make_service(tmp_path)
        svc._record_decision(decision(), "r1", "chat", False, [user("do X")])
        turn2 = [user("do X"), assistant(), user("try again")]
        svc._record_decision(decision(), "r2", "chat", False, turn2)

        rows = {r["request_id"]: r for r in svc.recent_decisions(limit=10)}
        assert rows["r1"].get("feedback")
        assert rows["r1"]["feedback"][0]["source"] == "implicit"
        await svc.close()

    async def test_disabled_by_default(self, tmp_path):
        svc = make_service(tmp_path, implicit=False)
        svc._record_decision(decision(), "r1", "chat", False, [user("do X")])
        turn2 = [user("do X"), assistant(), user("no, that's wrong")]
        svc._record_decision(decision(), "r2", "chat", False, turn2)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert fb == []

    async def test_approval_toggle_suppresses_positive(self, tmp_path):
        svc = make_service(tmp_path, approval=False)
        svc._record_decision(decision(), "r1", "chat", False, [user("do X")])
        turn2 = [user("do X"), assistant(), user("thanks, that works")]
        svc._record_decision(decision(), "r2", "chat", False, turn2)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert fb == []

    async def test_off_by_default_is_fully_inert(self, tmp_path):
        # Disabled must not even populate the content-hash index (no SHA-1 /
        # OrderedDict churn on the request path).
        svc = make_service(tmp_path, implicit=False)
        svc._record_decision(decision(), "r1", "chat", False, [user("do X")])
        assert len(svc._decision_by_userhash) == 0
        await svc.close()

    async def test_record_decision_never_raises_on_malformed_content(self, tmp_path):
        # _index_user_hash runs on every request when enabled; a malformed
        # message must not break the routing contract.
        svc = make_service(tmp_path)
        for bad in (123, [{"type": "text", "text": 123}], {"k": "v"}):
            svc._record_decision(
                decision(), "r", "chat", False, [{"role": "user", "content": bad}]
            )
        await svc.close()  # no exception == pass

    async def test_no_double_emit_on_duplicate_turn(self, tmp_path):
        svc = make_service(tmp_path)
        svc._record_decision(decision(), "r1", "chat", False, [user("do X")])
        turn2 = [user("do X"), assistant(), user("no, that's wrong")]
        svc._record_decision(decision(), "r2", "chat", False, turn2)
        # A retried identical follow-up (new id, same messages) must not
        # emit a second implicit row against r1.
        svc._record_decision(decision(), "r2-retry", "chat", False, turn2)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert len(fb) == 1
        assert fb[0]["request_id"] == "r1"

    async def test_unknown_prior_is_noop(self, tmp_path):
        # First turn a correction arrives but the prior decision was never
        # seen by this process (e.g. restart / eviction): no crash, no row.
        svc = make_service(tmp_path)
        turn = [user("unseen prompt"), assistant(), user("no, that's wrong")]
        svc._record_decision(decision(), "r1", "chat", False, turn)
        await svc.close()

        fb = [
            r
            for r in read_jsonl(svc.settings.telemetry.path)
            if r.get("kind") == "feedback"
        ]
        assert fb == []
