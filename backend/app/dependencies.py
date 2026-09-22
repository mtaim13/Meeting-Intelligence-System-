"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import HTTPException, Request

from app.services.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Check server startup logs.")
    return registry
