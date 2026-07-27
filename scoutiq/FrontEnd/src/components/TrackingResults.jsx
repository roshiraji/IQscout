import { useEffect, useRef, useState } from "react";
import CoachInsights from "./CoachInsights";

function formatEta(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "estimating…";
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return `~${value}s`;
  return `~${Math.ceil(value / 60)}m`;
}

function TrackingResults({
  tracking,
  videoUrl,
  masksVideoUrl,
  isolatedVideoUrl,
  isolateStatus,
  onNewSelection,
  jobId,
  onPossessionUpdated,
  onPlayOutcomeUpdated,
  seekTime = null,
  seekNonce = 0,
}) {
  const [previewMode, setPreviewMode] = useState("full"); // "full" | "masks" | "isolate"
  const videoRef = useRef(null);

  const isolationAvailable =
    Boolean(isolatedVideoUrl) && tracking?.isolation_available !== false;
  const masksAvailable = Boolean(masksVideoUrl);
  const activeVideoUrl =
    previewMode === "isolate" && isolationAvailable
      ? isolatedVideoUrl
      : previewMode === "masks" && masksAvailable
        ? masksVideoUrl
        : videoUrl;

  const snapshotSeek =
    seekTime != null && Number.isFinite(Number(seekTime))
      ? Number(seekTime)
      : Number(tracking?.timestamp_seconds);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || !Number.isFinite(snapshotSeek)) return undefined;
    const apply = () => {
      try {
        el.currentTime = Math.max(0, snapshotSeek);
      } catch {
        // Ignore seek before metadata.
      }
    };
    apply();
    el.addEventListener("loadedmetadata", apply);
    return () => el.removeEventListener("loadedmetadata", apply);
  }, [snapshotSeek, seekNonce, activeVideoUrl]);

  if (!tracking) {
    return null;
  }

  const trackLabels = Object.keys(tracking.track_mapping || {});
  const isAuto = tracking.auto_tracked === true;
  const reusedCanonical = tracking.canonical_reuse === true;
  const retention = tracking.selected_player_retention
    ?? tracking.quality?.selected_player_retention
    ?? tracking.frame_coverage
    ?? 0;
  const rawTracks = tracking.unique_raw_tracks
    ?? tracking.quality?.unique_raw_tracks;
  const personGroups = tracking.unique_person_groups
    ?? tracking.quality?.unique_person_groups;
  const isolateBusy = ["queued", "processing"].includes(isolateStatus?.status);
  const isolateProgress = Math.round((isolateStatus?.progress || 0) * 100);
  const isolationMode =
    isolateStatus?.effective_mode || tracking?.isolation_mode || "temporal";
  const isolateBaseLabel =
    isolationMode === "hybrid"
      ? "Clean Isolate (hybrid)"
      : isolationMode === "ai"
        ? "Clean Isolate (AI)"
        : "Isolate";
  const isolateButtonLabel = isolateBusy
    ? `${isolateBaseLabel} · ${isolateProgress > 0 ? `${isolateProgress}%` : formatEta(isolateStatus?.eta_seconds)}`
    : isolateBaseLabel;

  return (
    <>
      <div className="card tracking-panel">
        <div className="tracking-panel-header">
          <h2>Player Tracking</h2>
          {typeof onNewSelection === "function" && (
            <button
              type="button"
              className="secondary-button"
              onClick={onNewSelection}
            >
              New selection
            </button>
          )}
        </div>
        <p className="status-text">
          {reusedCanonical
            ? `Following ${trackLabels.join(", ")} across the full clip. Spatial lock keeps each pick on the body you clicked (resists ID handoffs). Change checkboxes and Confirm again to rebuild — no full Analyze rerun.`
            : isAuto
            ? `Auto-tracked ${trackLabels.length} player(s) with the segmentation model (${tracking.model || "best_player_seg.pt"}) across the clip.`
            : `Following ${trackLabels.join(", ")} across the full clip with segmentation masks from ${tracking.model || "best_player_seg.pt"} (opens at snapshot frame ${tracking.snapshot_frame_number}; spatial lock sticks to the bodies you clicked).`}
        </p>

        <div className="metadata-grid">
          <div>
            <strong>Selected-player retention</strong>
            <span>{Math.round(retention * 100)}%</span>
          </div>
          <div>
            <strong>Players tracked</strong>
            <span>{trackLabels.length}</span>
          </div>
          <div>
            <strong>Frames tracked</strong>
            <span>
              {tracking.frames_tracked}
              {tracking.source_frame_count ? ` / ${tracking.source_frame_count}` : ""}
            </span>
          </div>
          <div>
            <strong>Position samples</strong>
            <span>{tracking.total_positions}</span>
          </div>
          {(rawTracks != null || personGroups != null) && (
            <div>
              <strong>Raw tracks → groups</strong>
              <span>
                {rawTracks ?? "—"} → {personGroups ?? "—"}
              </span>
            </div>
          )}
          <div>
            <strong>Runtime</strong>
            <span>{tracking.total_runtime_seconds}s</span>
          </div>
        </div>

        {tracking.unmatched_labels?.length > 0 && (
          <p className="tracking-warning">
            Could not anchor: {tracking.unmatched_labels.join(", ")} at the snapshot frame.
          </p>
        )}

        {tracking.quality?.canonical_fragmentation_warning && (
          <p className="tracking-warning">
            Some raw ByteTrack tracklets were short or crossed a camera cut. Stitched
            person groups are used for selection when available.
          </p>
        )}

        {tracking.quality?.gate && !tracking.quality.gate.passed && (
          <p className="tracking-warning">
            Quality gate: {tracking.quality.gate.limiting_factor}.{" "}
            {tracking.quality.gate.recommendation}
          </p>
        )}

        {videoUrl && (
          <div className="video-panel nested-video-panel">
            <h3>Tracked Players Preview</h3>
            <div
              className="snapshot-view-toggle"
              role="group"
              aria-label="Focused preview mode"
            >
              <button
                type="button"
                className={`view-toggle-btn ${previewMode === "full" ? "view-toggle-btn-active" : ""}`}
                onClick={() => setPreviewMode("full")}
              >
                Full scene
              </button>
              <button
                type="button"
                className={`view-toggle-btn ${previewMode === "masks" ? "view-toggle-btn-active" : ""}`}
                onClick={() => masksAvailable && setPreviewMode("masks")}
                disabled={!masksAvailable}
                title={
                  masksAvailable
                    ? "Jersey-colored segmentation only — no boxes or labels"
                    : "Masks preview not ready yet — Confirm again after updating the API"
                }
              >
                Segmentation only
              </button>
              <button
                type="button"
                className={`view-toggle-btn ${previewMode === "isolate" ? "view-toggle-btn-active" : ""}`}
                onClick={() => isolationAvailable && setPreviewMode("isolate")}
                disabled={!isolationAvailable}
                title={
                  isolationAvailable
                    ? "Isolate removes everyone except the selected players — court, hoop, and crowd stay visible."
                    : isolateStatus?.status === "failed"
                    ? isolateStatus.error || "Isolate processing failed. Re-Confirm to retry."
                    : isolateBusy
                    ? `Building a cleaner temporal background (${isolateProgress}%, ${formatEta(isolateStatus?.eta_seconds)} remaining).`
                    : "Re-Confirm to generate isolated preview."
                }
              >
                {isolateButtonLabel}
              </button>
            </div>
            {isolateBusy && (
              <div className="isolate-progress" role="status" aria-live="polite">
                <progress
                  value={isolateStatus?.progress || 0}
                  max="1"
                  aria-label="Isolate processing progress"
                />
                <span>
                  {isolateBaseLabel} · {isolateProgress}% ·{" "}
                  {formatEta(isolateStatus?.eta_seconds)} remaining
                  {isolateStatus?.stage
                    ? ` · ${String(isolateStatus.stage).replace(/_/g, " ")}`
                    : ""}
                </span>
              </div>
            )}
            {tracking?.isolation_ai_fallback && (
              <p className="status-text">
                AI isolate unavailable (GPU/LaMa) — temporal fallback was used.
              </p>
            )}
            {isolateStatus?.status === "failed" && (
              <p className="tracking-warning">
                Isolate processing failed: {isolateStatus.error || "Re-Confirm to retry."}
              </p>
            )}
            <p className="status-text">
              Full scene shows masks + boxes + labels. Segmentation only keeps
              jersey-colored masks. Isolate removes non-selected players so the
              court, hoop, and crowd stay in view.
            </p>
            <video
              key={activeVideoUrl}
              ref={videoRef}
              controls
              width="100%"
              preload="metadata"
              playsInline
              src={activeVideoUrl}
            >
              Your browser does not support video playback.
            </video>
          </div>
        )}

        <p className="status-text tracking-next-step">
          Retention is frames with at least one selected person / source frame count —
          the continuity selling point. Detection confidence is separate from identity lock.
        </p>
      </div>

      <CoachInsights
        insights={tracking.coach_insights}
        narrative={tracking.coach_narrative}
        jobId={jobId}
        onPossessionUpdated={onPossessionUpdated}
        onPlayOutcomeUpdated={onPlayOutcomeUpdated}
      />
    </>
  );
}

export default TrackingResults;
