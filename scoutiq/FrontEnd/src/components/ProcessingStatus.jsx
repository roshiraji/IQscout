function statusClass(status) {
  if (status === "approved") return "status-approved";
  if (status === "rejected") return "status-rejected";
  return "status-warning";
}

function ProcessingStatus({ validation, labeling, error, snapshot }) {
  if (error) {
    return (
      <div className="result-panel status-rejected">
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
    <div className={`result-panel ${statusClass(status)}`}>
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

      {status === "approved" && snapshot && (
        <div className="labeling-summary">
          <h3>Player snapshot ready</h3>
          <p>
            Frame {snapshot.snapshot_frame_number} at {snapshot.timestamp_seconds}s ·{" "}
            {snapshot.player_count} players detected — select boxes below.
          </p>
        </div>
      )}
    </div>
  );
}

export default ProcessingStatus;
