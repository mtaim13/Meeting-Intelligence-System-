"""End-to-end orchestration: request -> chunks -> scores -> summary -> bullets -> response."""
from __future__ import annotations

from app.config import Settings
from app.schemas.meeting import (
    ActionBullet,
    DecisionBullet,
    EvidenceChunk,
    MeetingIntelligenceResponse,
    MeetingRequest,
)
from app.services.bullet_fusion import build_action_bullets, build_decision_bullets, remove_ami_role_hallucinations
from app.services.evidence_selection import build_guided_input, generate_narrative_summary
from app.services.model_registry import ModelRegistry
from app.services.preprocessing import build_chunks
from app.services.scoring import score_chunks


def run_meeting_intelligence(
    request: MeetingRequest, registry: ModelRegistry, settings: Settings
) -> MeetingIntelligenceResponse:
    chunks = build_chunks(request, settings)
    chunks = score_chunks(chunks, registry, settings)

    guided_input, selected_indices = build_guided_input(chunks, registry, settings)
    raw_summary = generate_narrative_summary(guided_input, registry, settings)
    selected_rows = [chunks[i] for i in selected_indices]

    # The BART model was fine-tuned on AMI corpus summaries whose targets use role-based
    # language ("The marketing expert suggests...").  Strip any leaked role labels and
    # replace them with the actual speaker names that appeared in the selected evidence.
    all_speaker_names = list(dict.fromkeys(row["speaker"] for row in selected_rows))
    summary = remove_ami_role_hallucinations(raw_summary, all_speaker_names)

    decision_bullets = build_decision_bullets(selected_rows, registry, settings)
    action_bullets = build_action_bullets(selected_rows, registry, settings)

    return MeetingIntelligenceResponse(
        meeting_id=request.meeting_id or "unnamed-meeting",
        summary=summary,
        key_decisions=[DecisionBullet(**bullet) for bullet in decision_bullets],
        action_items=[ActionBullet(**bullet) for bullet in action_bullets],
        selected_evidence=[
            EvidenceChunk(
                speaker=row["speaker"],
                text=row["text"],
                importance=round(row["importance"], 4),
                decision_score=round(row["decision_score"], 4),
                is_decision=row["is_decision"],
                is_action=row["is_action"],
            )
            for row in selected_rows
        ],
        bullet_fusion_used=registry.has_fusion_model,
    )
