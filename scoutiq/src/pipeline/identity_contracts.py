"""Non-blocking identity attachments for anonymous player track IDs.

Tracking is authoritative and never waits for identity. Future roster/RAG,
jersey OCR, face, or ReID resolvers write evidence into this contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IdentityStatus = Literal["pending", "unresolved", "review", "resolved"]


class GameContext(BaseModel):
    league: str | None = None
    game_date: str | None = None
    home_team: str | None = None
    away_team: str | None = None


class IdentityEvidence(BaseModel):
    ocr_jersey: str | None = None
    ocr_consistency: float | None = None
    identity_evidence_status: str | None = None
    identity_evidence_reason: str | None = None
    team_hint: str | None = None
    face_score: float | None = None
    reid_score: float | None = None
    candidate_scores: list[dict] = Field(default_factory=list)
    keyframe_paths: list[str] = Field(default_factory=list)


class TrackIdentityAttachment(BaseModel):
    job_id: str
    track_id: int
    status: IdentityStatus = "pending"
    player_id: str | None = None
    player_name: str | None = None
    team_id: str | None = None
    jersey_number: str | None = None
    confidence: float | None = None
    score_margin: float | None = None
    reason: str | None = None
    evidence: IdentityEvidence | None = None


def pending_attachments(job_id: str, track_ids: list[int]) -> list[dict]:
    """Create safe placeholders; never invent names without evidence."""
    return [
        TrackIdentityAttachment(job_id=job_id, track_id=track_id).model_dump()
        for track_id in sorted(set(track_ids))
    ]
