"""Meeting intelligence endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.dependencies import get_registry
from app.schemas.meeting import MeetingIntelligenceResponse, MeetingRequest
from app.services.model_registry import ModelRegistry
from app.services.pipeline import run_meeting_intelligence

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post(
    "/analyze",
    response_model=MeetingIntelligenceResponse,
    summary="Analyze a meeting transcript",
    description=(
        "Runs the full pipeline (importance, decision, and action classifiers, evidence selection, "
        "BART narrative summary, T5 bullet fusion) on a meeting transcript and returns a structured "
        "intelligence report."
    ),
)
def analyze_meeting(
    payload: MeetingRequest, registry: ModelRegistry = Depends(get_registry)
) -> MeetingIntelligenceResponse:
    try:
        return run_meeting_intelligence(payload, registry, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
