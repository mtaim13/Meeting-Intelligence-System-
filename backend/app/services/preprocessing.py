"""Turns API input (utterances or raw transcript) into scoring-ready text chunks."""
from __future__ import annotations

import re

from app.config import Settings
from app.schemas.meeting import MeetingRequest

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_transcript(text: str, max_words: int) -> list[str]:
    """Split raw transcript text into sentence-like chunks.

    Mirrors chunk_external_text() in notebooks/06_meeting_demo.ipynb, used there for
    MeetingBank transcripts that don't come pre-segmented by speaker.
    """
    sentences = SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        while words:
            available = max_words - len(current)
            current.extend(words[:available])
            words = words[available:]
            if len(current) >= max_words or not words:
                if len(current) >= 3:
                    chunks.append(" ".join(current))
                current = []
    return chunks


def build_chunks(request: MeetingRequest, settings: Settings) -> list[dict]:
    """Return [{"index": int, "speaker": str, "text": str}, ...] in chronological order."""
    if request.utterances:
        rows = [
            {"speaker": (utterance.speaker or "Speaker").strip(), "text": utterance.text.strip()}
            for utterance in request.utterances
            if utterance.text and utterance.text.strip()
        ]
    else:
        rows = [
            {"speaker": "Speaker", "text": text}
            for text in chunk_transcript(request.transcript or "", settings.chunk_max_words)
        ]

    if not rows:
        raise ValueError("No usable text found in the request after chunking.")

    for index, row in enumerate(rows):
        row["index"] = index
    return rows
