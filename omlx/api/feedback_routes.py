# SPDX-License-Identifier: Apache-2.0
"""
Routing feedback API (M6.0 outcome loop — ingest).

`POST /v1/feedback` attaches an out-of-band outcome signal to a prior
routing decision, keyed by the `request_id` echoed on the original
response as the `x-omlx-request-id` header (or the client-supplied
`x-request-id`). The signal is appended to the routing telemetry corpus as
its own record and joined to the decision offline; the decision row is
never mutated. Ingest only — no routing behavior changes in this milestone.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/v1", tags=["feedback"])

# Callback to get the live RoutingService (set by server.py); None when
# routing is disabled.
_get_routing_service = None


def set_routing_service_getter(getter):
    """Set the callback returning the RoutingService instance or None."""
    global _get_routing_service
    _get_routing_service = getter


def _get_service():
    if _get_routing_service is None:
        return None
    return _get_routing_service()


class FeedbackRequest(BaseModel):
    """One outcome signal for a prior routed request.

    `score` is a scalar reward in [0, 1] (0 = bad, 1 = good). At least one of
    `score`, `label`, or `tags` must be present.
    """

    request_id: str = Field(
        ..., description="Join key: x-omlx-request-id from the response"
    )
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = Field(
        default=None,
        description="e.g. good/bad/accepted/rejected/regenerated/resolved/failed",
    )
    tags: list[str] | None = None
    comment: str | None = None
    source: str = "client"

    @model_validator(mode="after")
    def _require_signal(self) -> "FeedbackRequest":
        if self.score is None and self.label is None and not self.tags:
            raise ValueError("feedback requires at least one of: score, label, tags")
        return self


class FeedbackResponse(BaseModel):
    recorded: bool
    reason: str | None = None


@router.post("/feedback", status_code=202)
async def submit_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Record an outcome signal for a prior routing decision.

    Returns 202 (accepted, out-of-band); 422 on an empty signal. Never
    surfaces a store error — feedback is best-effort by design.
    """
    service = _get_service()
    if service is None:
        # Routing disabled -> no decision corpus to attach to; accept + no-op.
        return FeedbackResponse(recorded=False, reason="routing_disabled")
    service.record_feedback(
        feedback.request_id,
        score=feedback.score,
        label=feedback.label,
        tags=feedback.tags,
        comment=feedback.comment,
        source=feedback.source,
    )
    return FeedbackResponse(recorded=True)
