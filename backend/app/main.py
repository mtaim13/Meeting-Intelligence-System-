"""FastAPI application entrypoint.

Run with: `python run.py` (from the backend/ folder) or `uvicorn app.main:app --reload`.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, meetings
from app.services.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meeting_intelligence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading trained models from %s ...", settings.models_dir)
    app.state.registry = ModelRegistry.load(settings)
    logger.info("Models loaded on device: %s", app.state.registry.device)
    logger.info(
        "Bullet fusion model: %s",
        "available" if app.state.registry.has_fusion_model else "not found, falling back to raw sentences",
    )
    yield
    app.state.registry = None


app = FastAPI(
    title="Meeting Intelligence API",
    description="Turns a meeting transcript into a narrative summary, key decisions, and action items.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(meetings.router)
