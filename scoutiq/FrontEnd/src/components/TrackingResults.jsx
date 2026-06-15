function TrackingResults({ tracking, videoUrl }) {
  if (!tracking) {
    return null;
  }

  const mappedLabels = Object.values(tracking.track_mapping || {});

  return (
    <div className="tracking-panel">
      <h2>Player Tracking</h2>
      <p className="status-text">
        Following {mappedLabels.join(", ")} from frame {tracking.snapshot_frame_number} through
        the end of the clip.
      </p>

      <div className="metadata-grid">
        <div>
          <strong>Players tracked</strong>
          <span>{mappedLabels.length}</span>
        </div>
        <div>
          <strong>Frames tracked</strong>
          <span>{tracking.frames_tracked}</span>
        </div>
        <div>
          <strong>Position samples</strong>
          <span>{tracking.total_positions}</span>
        </div>
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

      {videoUrl && (
        <div className="video-panel nested-video-panel">
          <h3>Tracked Players Preview</h3>
          <video controls width="100%" preload="metadata" playsInline src={videoUrl}>
            Your browser does not support video playback.
          </video>
        </div>
      )}

      <p className="status-text tracking-next-step">
        Tracking CSV saved for future metrics: movement path, speed, spacing, screens, and
        defensive pressure.
      </p>
    </div>
  );
}

export default TrackingResults;
