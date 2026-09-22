"""Loads every trained model artifact once at startup and holds them for reuse across requests.

Mirrors the "Load trained models" section of notebooks/06_meeting_demo.ipynb.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from app.config import Settings


@dataclass
class ModelRegistry:
    device: str

    importance_model: Any
    importance_vectorizer: Any

    decision_tokenizer: Any
    decision_model: Any

    action_tokenizer: Any
    action_model: Any
    action_id2label: dict[int, str]

    summary_tokenizer: Any
    summary_model: Any

    fusion_tokenizer: Any | None
    fusion_model: Any | None

    @property
    def has_fusion_model(self) -> bool:
        return self.fusion_model is not None

    @classmethod
    def load(cls, settings: Settings) -> "ModelRegistry":
        required = {
            "importance classifier": settings.importance_path,
            "decision classifier": settings.decision_path,
            "action extractor": settings.action_path,
            "narrative summarizer": settings.summarizer_path,
        }
        missing = [f"{name} -> {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing trained artifact(s):\n" + "\n".join(f"  - {item}" for item in missing) +
                "\n\nTrain them with notebooks/02-05, then copy the model folders under "
                f"{settings.models_dir}, or set the MODELS_DIR environment variable to wherever they live."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(settings.importance_path, "rb") as f:
            importance_artifact = pickle.load(f)

        decision_tokenizer = AutoTokenizer.from_pretrained(settings.decision_path)
        decision_model = AutoModelForSequenceClassification.from_pretrained(settings.decision_path).to(device).eval()

        action_tokenizer = AutoTokenizer.from_pretrained(settings.action_path)
        action_model = AutoModelForTokenClassification.from_pretrained(settings.action_path).to(device).eval()
        action_id2label = {int(key): value for key, value in action_model.config.id2label.items()}

        summary_tokenizer = AutoTokenizer.from_pretrained(settings.summarizer_path)
        summary_model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarizer_path).to(device).eval()

        # Optional: the demo degrades gracefully to raw joined sentences if this is missing.
        fusion_tokenizer = fusion_model = None
        if settings.fusion_path.exists():
            fusion_tokenizer = AutoTokenizer.from_pretrained(settings.fusion_path)
            fusion_model = AutoModelForSeq2SeqLM.from_pretrained(settings.fusion_path).to(device).eval()

        return cls(
            device=device,
            importance_model=importance_artifact["model"],
            importance_vectorizer=importance_artifact["tfidf_vectorizer"],
            decision_tokenizer=decision_tokenizer,
            decision_model=decision_model,
            action_tokenizer=action_tokenizer,
            action_model=action_model,
            action_id2label=action_id2label,
            summary_tokenizer=summary_tokenizer,
            summary_model=summary_model,
            fusion_tokenizer=fusion_tokenizer,
            fusion_model=fusion_model,
        )
