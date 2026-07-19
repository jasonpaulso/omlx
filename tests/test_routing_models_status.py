# SPDX-License-Identifier: Apache-2.0
"""Fork: routing virtual model ("auto") in /v1/models/status.

Lives in a fork-owned file rather than appended to tests/test_server.py so
upstream can edit that file freely without conflicting.
"""

import pytest


class TestRoutingVirtualModelStatus:
    """/v1/models/status advertises the routing virtual model with a
    model_type, so clients can gate image attachments correctly."""

    class _FakePool:
        def get_status(self):
            return {
                "models": [
                    {
                        "id": "qwen-base",
                        "loaded": True,
                        "pinned": False,
                        "engine_type": "llm",
                        "model_type": "llm",
                        "config_model_type": "qwen3",
                    }
                ]
            }

        def resolve_model_id(self, model_id, settings_manager=None):
            return model_id

        def get_entry(self, model_id):
            return None

    @pytest.fixture
    def routing_state(self):
        import omlx.server as server_module
        from omlx.routing.service import RoutingService
        from omlx.settings import RoutingSettings

        original_pool = server_module._server_state.engine_pool
        original_service = server_module._server_state.routing_service
        original_manager = server_module._server_state.settings_manager
        server_module._server_state.engine_pool = self._FakePool()
        server_module._server_state.settings_manager = None

        def install(settings: RoutingSettings) -> None:
            server_module._server_state.routing_service = RoutingService(settings)

        try:
            yield install
        finally:
            server_module._server_state.engine_pool = original_pool
            server_module._server_state.routing_service = original_service
            server_module._server_state.settings_manager = original_manager

    @pytest.mark.asyncio
    async def test_virtual_model_listed_as_llm_without_vision_target(
        self, routing_state
    ):
        import omlx.server as server_module
        from omlx.settings import RoutingSettings

        routing_state(RoutingSettings(enabled=True))

        status = await server_module.list_models_status(True)

        entry = next(m for m in status["models"] if m["id"] == "auto")
        assert entry["model_type"] == "llm"
        assert entry["loaded"] is True

    @pytest.mark.asyncio
    async def test_virtual_model_listed_as_vlm_with_vision_target(self, routing_state):
        import omlx.server as server_module
        from omlx.settings import RoutingSettings

        routing_state(
            RoutingSettings(
                enabled=True,
                targets={"small": "s", "big": "b", "vision": "gemma-vlm"},
            )
        )

        status = await server_module.list_models_status(True)

        entry = next(m for m in status["models"] if m["id"] == "auto")
        assert entry["model_type"] == "vlm"

    @pytest.mark.asyncio
    async def test_no_routing_service_adds_no_virtual_entry(self, routing_state):
        import omlx.server as server_module

        server_module._server_state.routing_service = None

        status = await server_module.list_models_status(True)

        assert not any(m["id"] == "auto" for m in status["models"])
