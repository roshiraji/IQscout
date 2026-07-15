function statusClass(status) {
  if (status === "approved") return "status-approved";
  if (status === "rejected") return "status-rejected";
  return "status-warning";
}

function ProcessingStatus({ validation, labeling, error, snapshot, job }) {
  if (error) {
    return (
      <div className="card result-panel status-rejected">
        <h2>Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  if (!validation) {
    return null;
  }

  const { status, message, metadata, warnings = [], errors = [] } = validation;

  return (
    <div className={`card result-panel ${statusClass(status)}`}>
      <h2>Intake: {status}</h2>
      <p>{message}</p>

      <div className="metadata-grid">
        <div><strong>File</strong><span>{metadata.filename}</span></div>
        <div><strong>Duration</strong><span>{metadata.duration_seconds}s</span></div>
        <div><strong>Resolution</strong><span>{metadata.width}×{metadata.height}</span></div>
        <div><strong>FPS</strong><span>{metadata.fps}</span></div>
        <div><strong>Size</strong><span>{metadata.file_size_mb} MB</span></div>
        <div><strong>Frames</strong><span>{metadata.frame_count}</span></div>
      </div>

      {warnings.length > 0 && (
        <ul className="message-list">
          {warnings.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}

      {errors.length > 0 && (
        <ul className="message-list error-list">
          {errors.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}

      {labeling && (
        <div className="labeling-summary">
          <h3>YOLO labeling complete</h3>
          <p>
            {labeling.total_detections} detections · {labeling.frames_processed} frames ·{" "}
            {labeling.device_label} · {labeling.model_label}
          </p>
          <p>
            Runtime {labeling.total_runtime_seconds}s · pipeline {labeling.pipeline_fps} FPS
          </p>
        </div>
      )}

      {status === "approved" && job && job.status !== "ready" && (
        <div className="labeling-summary">
          <h3>Full-clip tracking: {job.stage || job.status}</h3>
          <p>
            {job.frames_processed || 0} / {job.total_frames || "?"} frames ·{" "}
            {Math.round((job.progress || 0) * 100)}%
          </p>
          <progress value={job.progress || 0} max="1" style={{ width: "100%" }} />
          <p>
            Selection unlocks only after the canonical model pass covers the full clip.
          </p>
        </div>
      )}

      {job?.status === "ready" && (
        <div className="labeling-summary">
          <h3>Full-clip canonical tracking ready</h3>
          <p>
            {job.result?.frames_tracked} / {job.result?.source_frame_count} frames ·{" "}
            {job.result?.unique_tracks} track IDs ·{" "}
            {Math.round((job.result?.frame_coverage || 0) * 100)}% frame coverage
          </p>
        </div>
      )}

      {status === "approved" && snapshot && (
        <div className="labeling-summary">
          <h3>Canonical track selection ready</h3>
          <p>
            Frame {snapshot.snapshot_frame_number} at {snapshot.timestamp_seconds}s ·{" "}
            {snapshot.player_count} mature tracks — select IDs below.
          </p>
        </div>
      )}
    </div>
  );
}

export default ProcessingStatus;
