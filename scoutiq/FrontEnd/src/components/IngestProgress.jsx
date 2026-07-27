const INGEST_STEPS = [
  { id: "upload", label: "Upload video", match: ["uploading"] },
  { id: "validate", label: "Validate clip", match: ["validating"] },
  { id: "queue", label: "Queue job", match: ["queued"] },
  { id: "model", label: "Load model", match: ["loading_model"] },
  { id: "track", label: "Track players", match: ["tracking"] },
  { id: "encode", label: "Encode preview", match: ["encoding"] },
  { id: "stitch", label: "Stitch IDs", match: ["stitching"] },
  { id: "snapshot", label: "Build snapshot", match: ["snapshot"] },
  { id: "ball", label: "Ball / possession", match: ["ball_tracking"] },
  { id: "ready", label: "Ready to select", match: ["ready", "ready_partial_smoke_test"] },
];

const STAGE_ORDER = [
  "uploading",
  "validating",
  "queued",
  "loading_model",
  "tracking",
  "encoding",
  "stitching",
  "snapshot",
  "ball_tracking",
  "ready",
  "ready_partial_smoke_test",
];

function stageIndex(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  return idx >= 0 ? idx : 0;
}

function formatIngestDetail(job, isBusy, ingestPhase, selectedFileName) {
  if (isBusy && ingestPhase === "uploading") {
    return selectedFileName
      ? `Uploading ${selectedFileName}...`
      : "Uploading video...";
  }
  if (isBusy && ingestPhase === "validating") {
    return "Checking duration, resolution, and FPS...";
  }
  if (!job) {
    return isBusy ? "Starting ingest..." : "";
  }
  if (job.status === "failed") {
    return "Full-clip tracking could not finish.";
  }
  if (job.status === "ready") {
    const frames = job.result?.frames_tracked ?? job.frames_processed;
    const total = job.result?.source_frame_count ?? job.total_frames;
    return `Full-clip tracking ready${
      frames != null && total != null ? ` (${frames} / ${total} frames)` : ""
    }. Continue to Select.`;
  }
  const stage = job.stage || job.status;
  const map = {
    queued: "Waiting for the tracking worker...",
    loading_model: "Loading player segmentation weights...",
    tracking: "Detecting and tracking players across the clip...",
    encoding: "Encoding browser preview (still working)...",
    stitching: "Stitching tracklets into stable person groups...",
    snapshot: "Building selection snapshot...",
    ball_tracking: "Running ball / possession pass...",
  };
  const base = map[stage] || String(stage).replace(/_/g, " ");
  if (
    stage === "tracking" &&
    job.frames_processed != null &&
    job.total_frames
  ) {
    return `${base} (${job.frames_processed} / ${job.total_frames} frames)`;
  }
  return base;
}

/**
 * Overall bar: never show 100% until job.status === "ready".
 * Backend now weights stages; frontend still caps as a safety net.
 */
function resolveIngestProgress(job, isBusy, ingestPhase, ready) {
  if (ready) return 1;
  if (isBusy) {
    return ingestPhase === "validating" ? 0.08 : 0.04;
  }
  if (!job) return 0;

  const raw = Number(job.progress);
  if (Number.isFinite(raw) && raw >= 0) {
    return Math.min(Math.max(raw, 0), 0.99);
  }

  // Fallback if an older API omits weighted progress.
  const stage = job.stage || job.status || "queued";
  const idx = stageIndex(stage);
  const base = idx / Math.max(STAGE_ORDER.length - 1, 1);
  if (
    stage === "tracking" &&
    job.frames_processed != null &&
    job.total_frames
  ) {
    const frameFrac = job.frames_processed / Math.max(job.total_frames, 1);
    return Math.min(0.05 + frameFrac * 0.63, 0.68);
  }
  return Math.min(base, 0.99);
}

/**
 * Live progress for Analyze / video ingest (upload → validate → full-clip track).
 */
function IngestProgress({
  isBusy = false,
  ingestPhase = null,
  selectedFileName = "",
  job = null,
  validation = null,
}) {
  const rejected =
    validation &&
    (validation.status === "rejected" || validation.usable === false);
  const failed = job?.status === "failed" || rejected;
  const ready = job?.status === "ready";
  const tracking =
    isBusy ||
    (job && ["queued", "tracking", "identity"].includes(job.status)) ||
    (job &&
      job.status !== "ready" &&
      job.status !== "failed" &&
      Boolean(job.stage));

  if (!isBusy && !job && !rejected) {
    return null;
  }

  const stage =
    isBusy && ingestPhase
      ? ingestPhase
      : job?.stage ||
        (job?.status === "ready"
          ? "ready"
          : job?.status === "failed"
            ? "failed"
            : job?.status || "queued");

  const progress = resolveIngestProgress(job, isBusy, ingestPhase, ready);
  const percent = ready ? 100 : Math.min(99, Math.round(progress * 100));
  const detail = formatIngestDetail(job, isBusy, ingestPhase, selectedFileName);
  const currentIdx = stageIndex(stage === "failed" ? "queued" : stage);
  const postTrack =
    !ready &&
    ["encoding", "stitching", "snapshot", "ball_tracking"].includes(stage);

  return (
    <div
      className={`card result-panel ingest-progress-panel ${
        failed ? "status-rejected" : "status-warning"
      }`}
      role="status"
      aria-live="polite"
    >
      <h2>
        {failed
          ? "Ingest failed"
          : ready
            ? "Ingest complete"
            : isBusy
              ? "Ingesting video..."
              : "Analyzing clip..."}
      </h2>
      <p className="status-text">{detail}</p>

      {!failed && (
        <>
          <div className="confirm-progress-meter">
            <progress
              value={progress}
              max="1"
              aria-label="Video ingest progress"
            />
            <strong>{percent}%</strong>
          </div>
          {postTrack && (
            <p className="status-text ingest-posttrack-note">
              Frame tracking finished — finishing encode, ID stitch, snapshot, and
              ball. Progress stays under 100% until selection unlocks.
            </p>
          )}
          <ol className="confirm-stage-list">
            {INGEST_STEPS.map((step) => {
              const stepIdx = Math.min(...step.match.map((s) => stageIndex(s)));
              const done = ready || currentIdx > stepIdx;
              const active =
                !ready &&
                !failed &&
                (step.match.includes(stage) ||
                  (isBusy && step.match.includes(ingestPhase)));
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
          {tracking && !ready && (
            <p className="status-text">
              Selection unlocks only when this reaches 100% and status is Ready.
              Keep this tab open — progress updates live.
            </p>
          )}
        </>
      )}

      {failed && (
        <p className="tracking-warning">
          {job?.error ||
            validation?.message ||
            "Video ingest failed. Try another clip or re-Analyze."}
        </p>
      )}
    </div>
  );
}

export default IngestProgress;
