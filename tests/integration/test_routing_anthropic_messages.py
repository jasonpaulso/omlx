# SPDX-License-Identifier: Apache-2.0
"""Fork: semantic-routing pre-dispatch hook on /v1/messages.

Lives in a fork-owned file rather than appended to test_server_endpoints.py so
upstream can edit that file freely. Fixtures are reused from it by import.
"""

from .test_server_endpoints import (  # noqa: F401
    client,
    mock_embedding_engine,
    mock_engine_pool,
    mock_llm_engine,
    mock_reranker_engine,
)


class TestAnthropicMessagesRouting:
    """Tests for the semantic-routing pre-dispatch hook on /v1/messages.

    Mirrors the /v1/chat/completions routing hook (see docs/ROUTING.md).
    Telemetry is disabled on the fake service to avoid spinning up the
    background jsonl writer task inside a sync TestClient call; the hook's
    behavior (rewrite, header, outcome callback) is verified directly.
    """

    @staticmethod
    def _make_routing_service(targets=None):
        from omlx.routing.service import RoutingService
        from omlx.settings import RoutingSettings

        settings = RoutingSettings()
        settings.virtual_model_id = "auto"
        settings.targets = targets or {"small": "test-model", "big": "test-model"}
        settings.telemetry.enabled = False
        return RoutingService(settings)

    def test_auto_model_routes_and_sets_header(self, client, mock_engine_pool):
        """model="auto" with tools present hits the agentic override (no
        classification needed), gets rewritten, and the response carries
        the x-omlx-route header with the outcome recorded post-response."""
        from omlx.server import _server_state

        service = self._make_routing_service()
        recorded_outcomes = []
        service.record_outcome = lambda request_id, **kw: recorded_outcomes.append(
            (request_id, kw)
        )

        original = _server_state.routing_service
        _server_state.routing_service = service
        try:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "auto",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                },
            )
        finally:
            _server_state.routing_service = original

        assert response.status_code == 200
        assert response.headers["x-omlx-route"].startswith("test-model;")
        assert "rule=override:tools" in response.headers["x-omlx-route"]
        assert mock_engine_pool.get_engine_calls[-1]["model_id"] == "test-model"
        assert len(recorded_outcomes) == 1
        assert recorded_outcomes[0][1]["completion_tokens"] == 5
        assert recorded_outcomes[0][1]["finish_reason"] == "stop"

    def test_auto_model_routes_and_sets_header_streaming(
        self, client, mock_engine_pool
    ):
        """Same hook, streaming path: header on the SSE response and the
        outcome callback fires once the stream completes."""
        from omlx.server import _server_state

        service = self._make_routing_service()
        recorded_outcomes = []
        service.record_outcome = lambda request_id, **kw: recorded_outcomes.append(
            (request_id, kw)
        )

        original = _server_state.routing_service
        _server_state.routing_service = service
        try:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "auto",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                },
            )
        finally:
            _server_state.routing_service = original

        assert response.status_code == 200
        assert response.headers["x-omlx-route"].startswith("test-model;")
        assert len(recorded_outcomes) == 1
        assert recorded_outcomes[0][1]["finish_reason"] == "stop"

    def test_concrete_model_bypasses_routing(self, client, mock_engine_pool):
        """A concrete model id must not be touched even when routing is
        wired up -- only the virtual model id triggers the hook."""
        from omlx.server import _server_state

        service = self._make_routing_service()
        original = _server_state.routing_service
        _server_state.routing_service = service
        try:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        finally:
            _server_state.routing_service = original

        assert response.status_code == 200
        assert "x-omlx-route" not in response.headers
        assert mock_engine_pool.get_engine_calls[-1]["model_id"] == "test-model"

    def test_routing_disabled_leaves_messages_unchanged(
        self, client, mock_engine_pool
    ):
        """No routing_service wired (the client fixture's default) leaves
        the Anthropic path byte-for-byte the pre-hook behavior."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert "x-omlx-route" not in response.headers

    def test_image_content_routes_to_vision_target_without_classification(
        self, client, mock_engine_pool
    ):
        """Anthropic image blocks (type="image") are non-text parts, so the
        shape rule routes to targets.vision before classification runs --
        confirms _detect_modality handles Anthropic's shape."""
        from omlx.server import _server_state

        service = self._make_routing_service(
            targets={
                "small": "test-model",
                "big": "test-model",
                "vision": "test-model",
            }
        )

        async def _getter(model_id):
            raise AssertionError("classification must not run for image content")

        service.set_engine_getter(_getter)

        original = _server_state.routing_service
        _server_state.routing_service = service
        try:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "auto",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aGVsbG8=",
                                    },
                                },
                                {"type": "text", "text": "What is this?"},
                            ],
                        }
                    ],
                },
            )
        finally:
            _server_state.routing_service = original

        assert response.status_code == 200
        assert "rule=shape:vision" in response.headers["x-omlx-route"]
