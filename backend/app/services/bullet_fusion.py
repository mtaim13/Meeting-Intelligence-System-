"""Groups adjacent flagged evidence and rewrites each group into one clean bullet.

Mirrors "5b. Group adjacent evidence and fuse into clean bullets" in notebooks/06_meeting_demo.ipynb,
using the T5 model trained in notebooks/07_bullet_fusion.ipynb.
"""
from __future__ import annotations

import re

import torch

from app.config import Settings
from app.services.model_registry import ModelRegistry

# AMI corpus role labels that the T5 and BART models learned to inject from their
# training targets.  Listed longest-first so the regex alternation is greedy and
# "user interface designer" matches before the shorter "interface designer" would.
_AMI_ROLES = [
    "user interface designer",
    "user interface specialist",
    "user interface expert",
    "industrial designer",
    "interface designer",       # shortened form seen in AMI abstracts
    "ui designer",
    "marketing expert",
    "marketing manager",
    "marketing specialist",
    "project manager",
    "design expert",
    "team",                     # catches both "team" and "the team" via optional-article prefix
]

# Matches an optional leading article ("the " / "a ") followed by any AMI role label.
# Capturing the article separately lets us decide whether to keep or drop it based on
# whether the replacement is a proper name (drop article) or a collective (keep article).
_AMI_ROLE_RE = re.compile(
    r"\b(the\s+|a\s+)?(" + "|".join(re.escape(role) for role in _AMI_ROLES) + r")\b",
    re.IGNORECASE,
)

# Pronouns that BART generates to refer back to a previously mentioned AMI role.
# After the role is replaced these pronouns become dangling ("He also discussed...").
# We normalise them to "the group" so they stay readable.
_DANGLING_PRONOUN_RE = re.compile(
    r"(?<![A-Za-z])(He|She|They|He/she)\b(?=\s+(also|then|also then|further|additionally|went on|continued|noted|added|suggested|discussed|proposed|decided|agreed|mentioned|explained|stated|said|asked|told|instructed|recommended|concluded)\b)",
    re.IGNORECASE,
)


def remove_ami_role_hallucinations(text: str, speaker_names: list[str]) -> str:
    """Replace AMI corpus role names in generated text with the actual speaker name(s).

    The T5 / BART models were fine-tuned on AMI abstracts that use role labels
    ("The marketing expert suggests...") instead of real names.  When these models
    process a remote-control-domain meeting they reproduce that style even though
    the input transcript contains real names.  This post-processing step:

    1. Replaces all AMI role labels with the real speaker name or "the group".
    2. Removes dangling pronouns (He/She/They) that referred to those roles.
    3. Collapses redundant "the group" repetitions within a sentence.
    """
    if not _AMI_ROLE_RE.search(text) and not _DANGLING_PRONOUN_RE.search(text):
        return text

    if len(speaker_names) == 1:
        name = speaker_names[0]
        result = _AMI_ROLE_RE.sub(lambda m: name, text)
        # Replace dangling pronouns that referred to the single speaker.
        result = _DANGLING_PRONOUN_RE.sub(name, result)
        return result

    # Multiple speakers — use "the group" as collective noun throughout.
    def _replace_group(m: re.Match) -> str:
        article = m.group(1) or "the "
        return f"{article.lower()}group"

    result = _AMI_ROLE_RE.sub(_replace_group, text)
    result = _DANGLING_PRONOUN_RE.sub("the group", result)

    # "The group instructed/asked/told them/the group to X" → "The group agreed to X"
    # This arises when both the assigning role and the assigned role collapse to "the group".
    result = re.sub(
        r"\bthe group\s+(?:instructed|asked|told|directed|requested)\s+(?:them|the group)\s+to\b",
        "The group agreed to",
        result,
        flags=re.IGNORECASE,
    )
    # Clean up any remaining "and the group to X" / "and them to X" fragments that
    # were the second branch of the same assignment sentence (e.g. "agreed to do A
    # and the group to do B" → "agreed to do A and B").
    result = re.sub(
        r"\s+and\s+(?:the group|them)\s+to\b",
        " and to",
        result,
        flags=re.IGNORECASE,
    )

    # For remaining repetitions within a sentence, replace subsequent "the group"
    # with "they" so "the group discussed the group's plan" →
    # "the group discussed their plan" (handled below for possessives too).
    def _dedupe_group_in_sentence(sentence: str) -> str:
        parts = re.split(r"\b(the group)\b", sentence, flags=re.IGNORECASE)
        found = False
        out = []
        for part in parts:
            if re.fullmatch(r"the group", part, re.IGNORECASE):
                if not found:
                    out.append(part)
                    found = True
                else:
                    out.append("they")
            else:
                out.append(part)
        return "".join(out)

    sentences = re.split(r"(?<=[.!?])\s+", result)
    result = " ".join(_dedupe_group_in_sentence(s) for s in sentences)
    # "the group and they" / "the group or they" → just "the group"
    result = re.sub(r"\bthe group\s+(?:and|or)\s+they\b", "the group", result, flags=re.IGNORECASE)
    # Re-capitalise "the group" when it lands at the start of a sentence.
    return re.sub(r"(^|(?<=[.!?]\s))the group", lambda m: m.group(0).capitalize(), result)


def group_consecutive(rows: list[dict], max_gap: int) -> list[list[dict]]:
    """Merge rows whose original transcript index is within max_gap of the previous row in the group."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for row in sorted(rows, key=lambda r: r["index"]):
        if current and row["index"] - current[-1]["index"] > max_gap:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


@torch.no_grad()
def fuse_bullets(
    evidence_sentences: list[str],
    task_type: str,
    registry: ModelRegistry,
    settings: Settings,
    speaker_names: list[str] | None = None,
) -> str:
    """Rewrite a list of noisy flagged sentences into one clean bullet using the T5 fusion model."""
    if not evidence_sentences:
        return ""
    prefixed = f"fuse {task_type}: " + " ".join(evidence_sentences)
    encoded = registry.fusion_tokenizer(
        prefixed, truncation=True, max_length=settings.fusion_max_source_tokens, return_tensors="pt"
    ).to(registry.device)
    generated = registry.fusion_model.generate(
        **encoded, num_beams=settings.fusion_num_beams, max_new_tokens=settings.fusion_max_target_tokens
    )
    raw = registry.fusion_tokenizer.decode(generated[0], skip_special_tokens=True)
    if speaker_names:
        raw = remove_ami_role_hallucinations(raw, speaker_names)
    return raw


def build_bullet(group: list[dict], task_type: str, registry: ModelRegistry, settings: Settings) -> str:
    evidence_sentences = [f"{row['speaker']}: {row['text']}" for row in group]
    speaker_names = list(dict.fromkeys(row["speaker"] for row in group))
    if registry.has_fusion_model:
        return fuse_bullets(evidence_sentences, task_type, registry, settings, speaker_names=speaker_names)
    return " / ".join(row["text"] for row in group)  # fallback: raw sentences, joined


def build_decision_bullets(selected_rows: list[dict], registry: ModelRegistry, settings: Settings) -> list[dict]:
    groups = group_consecutive([row for row in selected_rows if row["is_decision"]], settings.group_max_index_gap)
    return [
        {
            "bullet": build_bullet(group, "decision", registry, settings),
            "confidence": max(row["decision_score"] for row in group),
            "speakers": ", ".join(dict.fromkeys(row["speaker"] for row in group)),
            "evidence": [row["text"] for row in group],
        }
        for group in groups
    ]


def build_action_bullets(selected_rows: list[dict], registry: ModelRegistry, settings: Settings) -> list[dict]:
    groups = group_consecutive([row for row in selected_rows if row["action_items"]], settings.group_max_index_gap)
    results = []
    for group in groups:
        assignees = [item["assignee"] for row in group for item in row["action_items"] if item["assignee"]]
        deadlines = [item["deadline"] for row in group for item in row["action_items"] if item["deadline"]]
        results.append({
            "action": build_bullet(group, "action", registry, settings),
            "assignee": ", ".join(dict.fromkeys(assignees)) or None,
            "deadline": ", ".join(dict.fromkeys(deadlines)) or None,
            "speakers": ", ".join(dict.fromkeys(row["speaker"] for row in group)),
            "evidence": [row["text"] for row in group],
        })
    return results
