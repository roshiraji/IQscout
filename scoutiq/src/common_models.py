"""Shared data contracts for identity-resolution pipeline stages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlayerRosterRecord(BaseModel):
    player_id: str
    league: str = "WNBA"
    season: int
    team_id: str
    team_name: str
    full_name: str
    aliases: list[str] = Field(default_factory=list)
    jersey_number: str | None = None
    position: str | None = None
    active_from: str | None = None
    active_to: str | None = None
    headshot_url: str | None = None
    roster_source_url: str | None = None


class PlayerVectorRecord(BaseModel):
    player_id: str
    face: list[float] | None = None
    reid_proto: list[float] | None = None
    name_text: list[float] | None = None


class TrackDetection(BaseModel):
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    bbox_xyxy: list[float]
    det_conf: float
    cls: int


class TrackletEvidence(BaseModel):
    video_id: str
    tracklet_id: str
    team_hint: str | None = None
    game_date: str | None = None
    keyframes: list[str] = Field(default_factory=list)
    ocr_reads: list[dict] = Field(default_factory=list)
    face_embeddings: list[list[float]] = Field(default_factory=list)
    reid_embeddings: list[list[float]] = Field(default_factory=list)
    ocr_consistency: float = 0.0
    num_keyframes: int = 0


class ResolvedTracklet(BaseModel):
    video_id: str
    tracklet_id: str
    resolved_player_id: str | None = None
    resolver_confidence: float | None = None
    resolver_score: float | None = None
    score_margin: float | None = None
    candidate_scores: list[dict] = Field(default_factory=list)
    status: str = "unresolved"
    reason: str | None = None


class ClipRecord(BaseModel):
    clip_id: str
    video_id: str
    tracklet_id: str
    resolved_player_id: str | None = None
    player_name: str | None = None
    team_id: str | None = None
    jersey_number: str | None = None
    game_date: str | None = None
    clip_path: str
    start_frame: int
    end_frame: int
    confidence: float | None = None
