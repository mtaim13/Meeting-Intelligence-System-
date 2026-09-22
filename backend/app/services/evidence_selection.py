"""Selects transcript evidence and generates the narrative summary.

Mirrors "5. Select evidence and generate the narrative summary" in notebooks/06_meeting_demo.ipynb.
"""
from __future__ import annotations

import re

import torch

from app.config import Settings
from app.services.model_registry import ModelRegistry

# An utterance that is purely a question or a negated modal cannot be a confirmed decision.
# These patterns are checked *before* applying the [DECISION] marker.
_QUESTION_RE = re.compile(r"\?\s*$")
_NEGATED_MODAL_RE = re.compile(
    r"\b(don't|doesn't|didn't|won't|wouldn't|can't|cannot|not)\b.{0,40}"
    r"\b(want|like|think|need|agree|decide|go with)\b",
    re.I,
)

# First-person contractions that can be safely rewritten to third-person without
# breaking verb agreement.  Only applied to [ACTION]-flagged chunks.
# Bare "I" is intentionally excluded: replacing it requires conjugating the
# following verb (e.g. "I need" → "Lucas needs"), which is too error-prone.
_I_WILL = re.compile(r"\bI'll\b")
_I_HAVE = re.compile(r"\bI've\b")
_I_AM   = re.compile(r"\bI'm\b")
_MY     = re.compile(r"\bmy\b", re.I)


def _third_person_action(text: str, speaker: str) -> str:
    """Rewrite common first-person contractions to the speaker's name so BART
    receives 'Lucas will update...' instead of 'Lucas: I'll update...'

    This makes the speaker's name the grammatical subject of the action, which
    helps BART (trained on AMI single-letter speaker labels) attribute the action
    correctly in the narrative summary.
    """
    t = _I_WILL.sub(f"{speaker} will", text)
    t = _I_HAVE.sub(f"{speaker} has", t)
    t = _I_AM.sub(f"{speaker} is", t)
    t = _MY.sub(f"{speaker}'s", t)
    return t


def _is_genuine_decision(text: str) -> bool:
    """Return False for utterances that are clearly questions or negated suggestions.

    The decision classifier was trained on AMI data and assigns high scores to
    modal-verb questions ("Could we...?") and suggestions that are subsequently
    rejected.  This guard prevents [DECISION] from being applied to those cases.
    """
    if _QUESTION_RE.search(text.strip()):
        return False
    if _NEGATED_MODAL_RE.search(text):
        return False
    return True


def build_guided_input(chunks: list[dict], registry: ModelRegistry, settings: Settings) -> tuple[str, list[int]]:
    """Greedily fill a token budget, preferring decision/action-flagged chunks, then importance.

    Improvements over the original notebook implementation:
    - [DECISION] marker is suppressed for questions and negated suggestions.
    - Bridging utterances immediately surrounding any flagged chunk are added
      to the selected set (at lower priority) so BART sees the rejection /
      follow-up context that explains whether a suggestion was accepted.

    Returns the guided source text (chronological order) and the indices of chunks it used.
    """
    n = len(chunks)

    # Build the set of indices that are neighbours of a flagged chunk.  These
    # bridging utterances supply context (e.g. the reply that rejected a proposal)
    # and are included before purely importance-ranked chunks but after flagged ones.
    flagged = {i for i in range(n) if chunks[i]["is_decision"] or chunks[i]["is_action"]}
    bridging = {
        neighbour
        for i in flagged
        for neighbour in (i - 1, i + 1)
        if 0 <= neighbour < n and neighbour not in flagged
    }

    def rank_key(i: int) -> tuple:
        is_flagged = chunks[i]["is_decision"] or chunks[i]["is_action"]
        is_bridge = i in bridging
        return (is_flagged, is_bridge, chunks[i]["importance"])

    ranked = sorted(range(n), key=rank_key, reverse=True)

    selected: list[tuple[int, str]] = []
    used_tokens = 0

    for index in ranked:
        chunk = chunks[index]
        genuine_decision = chunk["is_decision"] and _is_genuine_decision(chunk["text"])
        marker = ("[DECISION] " if genuine_decision else "") + ("[ACTION] " if chunk["is_action"] else "")

        # For action-flagged utterances, rewrite first-person pronouns into the
        # speaker's name so BART sees the name as the grammatical subject of the
        # action rather than as an opaque speaker-prefix label.
        if chunk["is_action"]:
            display_text = _third_person_action(chunk["text"], chunk["speaker"])
            line = f"{marker}{display_text}"
        else:
            line = f"{marker}{chunk['speaker']}: {chunk['text']}"
        line_tokens = len(registry.summary_tokenizer(line, add_special_tokens=False).input_ids)
        if used_tokens + line_tokens <= settings.summary_max_source_tokens:
            selected.append((index, line))
            used_tokens += line_tokens

    selected.sort(key=lambda pair: pair[0])  # restore chronological reading order
    return "\n".join(line for _, line in selected), [index for index, _ in selected]


@torch.no_grad()
def generate_narrative_summary(guided_input: str, registry: ModelRegistry, settings: Settings) -> str:
    encoded = registry.summary_tokenizer(
        guided_input, return_tensors="pt", truncation=True, max_length=settings.summary_max_source_tokens
    ).to(registry.device)
    generated = registry.summary_model.generate(
        **encoded,
        num_beams=settings.summary_num_beams,
        length_penalty=settings.summary_length_penalty,
        min_new_tokens=settings.summary_min_new_tokens,
        max_new_tokens=settings.summary_max_new_tokens,
    )
    return registry.summary_tokenizer.decode(generated[0], skip_special_tokens=True)
