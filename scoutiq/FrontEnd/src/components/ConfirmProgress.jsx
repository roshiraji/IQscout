const STAGE_ORDER = [
  "saving_selection",
  "loading_selection",
  "loading_tracks",
  "rendering_preview",
  "encoding_videos",
  "coach_insights",
  "queuing_isolate",
  "ready",
];

function stageIndex(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  return idx >= 0 ? idx : 0;
}

function formatIsolateStage(stage) {
  const map = {
    queued: "Queued…",
    loading_lama: "Loading LaMa on GPU (VRAM-max tile)…",
    hybrid_temporal_lama: "Hybrid isolate — temporal fill, then LaMa leftovers…",
    ai_inpainting: "Clean Isolate (AI) — erasing non-selected players…",
    preparing_temporal_background: "Preparing temporal background…",
    reconstructing: "Aligning clean neighbor frames…",
    encoding: "Encoding browser video…",
    ready: "Ready",
    failed: "Failed",
  };
  return map[stage] || (stage ? String(stage).replace(/_/g, " ") : "Processing…");
}

function isolateMethodLabel(isolateStatus, tracking) {
  if (isolateStatus?.method) return isolateStatus.method;
  if (tracking?.isolation_method) return tracking.isolation_method;
  const mode = isolateStatus?.effective_mode || tracking?.isolation_mode;
  if (mode === "hybrid") return "Clean Isolate (hybrid: temporal->LaMa)";
  if (mode === "ai") return "Clean Isolate (AI: LaMa only)";
  return "Isolate (temporal)";
}

function formatEta(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return null;
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return `~${value}s left`;
  return `~${Math.ceil(value / 60)}m left`;
}

/**
 * Visible Confirm + Isolate progress so the user always knows what is happening.
 */
function ConfirmProgress({ confirmStatus, isolateStatus, isSaving }) {
  const confirmBusy =
    isSaving || ["processing", "queued"].includes(confirmStatus?.status);
  const isolateBusy = ["queued", "processing"].includes(isolateStatus?.status);
  const failed =
    confirmStatus?.status === "failed" || isolateStatus?.status === "failed";

  if (!confirmBusy && !isolateBusy && !failed && confirmStatus?.status !== "ready") {
    return null;
  }

  const progress = Math.max(
    0,
    Math.min(1, Number(confirmBusy ? confirmStatus?.progress ?? 0.05 : 1))
  );
  const percent = Math.round(progress * 100);
  const detail =
    confirmStatus?.detail ||
    (isSaving
      ? "Building your focused player preview…"
      : isolateBusy
        ? "Focused preview ready — isolate still building in the background…"
        : confirmStatus?.status === "ready"
          ? "Focused preview ready."
          : "");

  const currentStage =
    confirmStatus?.stage || (isSaving ? "saving_selection" : "ready");
  const currentIdx = stageIndex(currentStage);
  const uniqueSteps = [
    {
      id: "lock",
      label: "Lock selection",
      match: ["saving_selection", "loading_selection"],
    },
    { id: "tracks", label: "Load tracks", match: ["loading_tracks"] },
    { id: "preview", label: "Build previews", match: ["rendering_preview"] },
    { id: "encode", label: "Encode videos", match: ["encoding_videos"] },
    { id: "insights", label: "Matchup Outcomes", match: ["coach_insights"] },
    { id: "done", label: "Ready", match: ["queuing_isolate", "ready"] },
  ];

  const isolateProgress = Math.round((isolateStatus?.progress || 0) * 100);
  const isolateEta = formatEta(isolateStatus?.eta_seconds);

  return (
    <div
      className={`card result-panel confirm-progress-panel ${
        failed ? "status-rejected" : "status-warning"
      }`}
      role="status"
      aria-live="polite"
    >
      <h2>
        {failed
          ? "Confirm failed"
          : confirmBusy
            ? "Confirming selection…"
            : isolateBusy
              ? "Focused preview ready"
              : "Selection ready"}
      </h2>
      <p className="status-text">{detail}</p>

      {(confirmBusy || confirmStatus?.status === "ready") && !failed && (
        <>
          <div className="confirm-progress-meter">
            <progress value={progress} max="1" aria-label="Confirm progress" />
            <strong>{percent}%</strong>
          </div>
          <ol className="confirm-stage-list">
            {uniqueSteps.map((step) => {
              const stepIdx = Math.min(...step.match.map((s) => stageIndex(s)));
              const done =
                confirmStatus?.status === "ready" || currentIdx > stepIdx;
              const active = confirmBusy && step.match.includes(currentStage);
              return (
                <li
                  key={step.id}
                  className={
                    done
                      ? "confirm-stage-done"
                      : active
                        ? "confirm-stage-active"
                        : "confirm-stage-pending"
                  }
                >
                  <span className="confirm-stage-marker" aria-hidden="true">
                    {done ? "✓" : active ? "●" : "○"}
                  </span>
                  {step.label}
                </li>
              );
            })}
          </ol>
        </>
      )}

      {failed && (
        <p className="tracking-warning">
          {confirmStatus?.error ||
            isolateStatus?.error ||
            "Something went wrong. Try Confirm again."}
        </p>
      )}

      {isolateBusy && !failed && (
        <div className="isolate-progress confirm-isolate-row">
          <progress
            value={isolateStatus?.progress || 0}
            max="1"
            aria-label="Isolate processing progress"
          />
          <span>
            {isolateMethodLabel(isolateStatus, null)} · {isolateProgress}%
            {isolateEta ? ` · ${isolateEta}` : ""}
            {isolateStatus?.stage
              ? ` · ${formatIsolateStage(isolateStatus.stage)}`
              : ""}
          </span>
        </div>
      )}

      {isolateStatus?.ai_fallback && !failed && (
        <p className="status-text">
          Clean Isolate (hybrid/AI) was requested but GPU/LaMa was unavailable — using
          temporal isolate instead. Install{" "}
          <code>simple-lama-inpainting</code> and use CUDA to enable AI mode.
        </p>
      )}
    </div>
  );
}

export default ConfirmProgress;
