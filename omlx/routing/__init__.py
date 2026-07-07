# SPDX-License-Identifier: Apache-2.0
"""Semantic routing: classify "auto" requests to a concrete target model."""

from omlx.routing.profiler import RouterFeatures
from omlx.routing.service import RouteDecision, RoutingService

__all__ = ["RoutingService", "RouteDecision", "RouterFeatures"]
