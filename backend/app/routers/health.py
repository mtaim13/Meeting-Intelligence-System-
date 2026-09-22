"""Health and readiness checks."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    registry = getattr(request.app.state, "registry", None)
    return {
        "status": "ok" if registry is not None else "loading",
        "device": registry.device if registry else None,
        "bullet_fusion_available": registry.has_fusion_model if registry else False,
    }
