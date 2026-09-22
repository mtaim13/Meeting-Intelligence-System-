"""Runs the importance, decision, and action models over transcript chunks.

Feature definitions and inference logic are copied verbatim from notebooks/02_importance_classifier.ipynb,
notebooks/03_decision_classifier.ipynb, and notebooks/04_action_extractor.ipynb (as reused in
notebooks/06_meeting_demo.ipynb) so scores match the notebooks exactly.
"""
from __future__ import annotations

import re

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

from app.config import Settings
from app.services.model_registry import ModelRegistry

MODAL_VERBS = re.compile(r"\b(will|would|shall|should|must|need to|ought to|going to|have to|need|wants to)\b", re.I)
DECISION_KWS = re.compile(
    r"\b(decided|agreed|confirmed|resolved|concluded|chosen|approved|rejected|accepted|finalised|finalized|"
    r"going with|stick with|settle on)\b",
    re.I,
)
ACTION_VERBS = re.compile(
    r"\b(make|create|build|prepare|write|send|review|check|update|implement|design|discuss|look into|"
    r"find out|contact|set up|organise|organize|schedule|arrange|follow up)\b",
    re.I,
)
NEGATION_KWS = re.compile(r"\b(not|no|never|without|don't|doesn't|didn't|won't|wouldn't|can't|cannot)\b", re.I)
FIRST_PERSON = re.compile(r"\b(I|we|our|my|us)\b")
QUESTION_RE = re.compile(r"\?")


def importance_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        words = text.split()
        n = max(len(words), 1)
        rows.append([
            len(MODAL_VERBS.findall(text)), len(DECISION_KWS.findall(text)),
            len(ACTION_VERBS.findall(text)), len(NEGATION_KWS.findall(text)),
            len(FIRST_PERSON.findall(text)), int(bool(QUESTION_RE.search(text))),
            len(words), len(text), len(MODAL_VERBS.findall(text)) / n,
            sum(word[0].isupper() and len(word) > 1 for word in words) / n,
            int(text.endswith(".")), text.count(",") / n,
        ])
    return np.asarray(rows, dtype=float)


def score_importance(texts: list[str], registry: ModelRegistry) -> np.ndarray:
    x_tfidf = registry.importance_vectorizer.transform(texts)
    x_handcrafted = StandardScaler().fit_transform(importance_features(texts))
    x = hstack([x_tfidf, csr_matrix(x_handcrafted)])
    return registry.importance_model.predict_proba(x)[:, 1]


def _build_decision_contexts(texts: list[str], window: int) -> list[str]:
    """Concatenate neighbouring utterances around each text so the decision model can
    see whether a modal-verb sentence is a suggestion being rejected or a real decision.

    The target chunk is wrapped in angle brackets so the model can distinguish it from
    context. Each neighbour is separated by " | " to match conversational turn boundaries.
    """
    n = len(texts)
    contexts = []
    for i in range(n):
        before = texts[max(0, i - window):i]
        after = texts[i + 1:i + 1 + window]
        parts = []
        if before:
            parts.append(" | ".join(before))
        parts.append(f"<{texts[i]}>")
        if after:
            parts.append(" | ".join(after))
        contexts.append(" ".join(parts))
    return contexts


@torch.no_grad()
def score_decisions(texts: list[str], registry: ModelRegistry, settings: Settings) -> np.ndarray:
    # Build context-aware inputs so the model can distinguish a suggestion that is
    # immediately rejected from a genuine group decision.
    contexts = _build_decision_contexts(texts, settings.decision_context_window)
    values: list[float] = []
    for start in range(0, len(contexts), settings.classifier_batch_size):
        batch = contexts[start:start + settings.classifier_batch_size]
        encoded = registry.decision_tokenizer(
            batch, truncation=True, padding=True, max_length=256, return_tensors="pt"
        ).to(registry.device)
        logits = registry.decision_model(**encoded).logits
        values.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return np.asarray(values)


def decode_bio(tokens: list[str], labels: list[str]) -> list[tuple[str, int, int]]:
    spans, start, entity = [], None, None
    for index, label in enumerate(labels + ["O"]):
        if label.startswith("B-"):
            if entity is not None:
                spans.append((entity, start, index))
            entity, start = label[2:], index
        elif label.startswith("I-") and entity == label[2:]:
            continue
        else:
            if entity is not None:
                spans.append((entity, start, index))
            entity, start = None, None
    return spans


@torch.no_grad()
def extract_action_items(texts: list[str], registry: ModelRegistry, settings: Settings) -> list[list[dict]]:
    """Return one list of {action, assignee, deadline} dictionaries per text."""
    outputs: list[list[dict]] = []
    for start in range(0, len(texts), settings.classifier_batch_size):
        batch = texts[start:start + settings.classifier_batch_size]
        token_lists = [text.split() for text in batch]
        encoded = registry.action_tokenizer(
            token_lists, is_split_into_words=True, truncation=True,
            padding=True, max_length=128, return_tensors="pt",
        ).to(registry.device)
        predictions = registry.action_model(**encoded).logits.argmax(dim=-1).cpu().numpy()
        for row, tokens in enumerate(token_lists):
            labels, seen = [], set()
            for position, word_id in enumerate(encoded.word_ids(batch_index=row)):
                if word_id is not None and word_id not in seen:
                    seen.add(word_id)
                    labels.append(registry.action_id2label[int(predictions[row, position])])
            spans = decode_bio(tokens, labels[:len(tokens)])
            by_type: dict[str, list[str]] = {}
            for entity, span_start, span_end in spans:
                by_type.setdefault(entity, []).append(" ".join(tokens[span_start:span_end]))
            outputs.append([
                {
                    "action": action,
                    "assignee": (by_type.get("ASSIGNEE") or [None])[0],
                    "deadline": (by_type.get("DEADLINE") or [None])[0],
                }
                for action in by_type.get("ACTION", [])
            ])
    return outputs


def score_chunks(chunks: list[dict], registry: ModelRegistry, settings: Settings) -> list[dict]:
    """Attach importance/decision/action scores to each chunk in place, and return it."""
    texts = [chunk["text"] for chunk in chunks]
    importance_scores = score_importance(texts, registry)
    decision_scores = score_decisions(texts, registry, settings)
    action_items = extract_action_items(texts, registry, settings)

    for i, chunk in enumerate(chunks):
        chunk["importance"] = float(importance_scores[i])
        chunk["decision_score"] = float(decision_scores[i])
        chunk["is_decision"] = bool(decision_scores[i] >= settings.decision_threshold)

        # The BIO tagger operates on bare text, so it cannot resolve first-person pronouns
        # ("I'll do X") to a speaker name.  When the tagger finds an action but no ASSIGNEE,
        # the speaker of the utterance is the implicit assignee and is used as fallback.
        items = action_items[i]
        for item in items:
            if item["assignee"] is None and chunk.get("speaker"):
                item["assignee"] = chunk["speaker"]

        chunk["action_items"] = items
        chunk["is_action"] = bool(items)
    return chunks
