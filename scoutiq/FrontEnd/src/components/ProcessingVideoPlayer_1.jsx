import { useEffect, useRef, useState } from "react";

function ProcessingVideoPlayer({
  videoUrl,
  masksVideoUrl = "",
  seekTime = null,
  seekNonce = 0,
  title = "Processed Video",
}) {
  const videoRef = useRef(null);
  const [previewMode, setPreviewMode] = useState("full"); // full | masks
  const masksAvailable = Boolean(masksVideoUrl);
  const activeVideoUrl =
    previewMode === "masks" && masksAvailable ? masksVideoUrl : videoUrl;

  useEffect(() => {
    if (!masksAvailable && previewMode === "masks") {
      setPreviewMode("full");
    }
  }, [masksAvailable, previewMode]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || seekTime == null || !Number.isFinite(Number(seekTime))) return;
    try {
      el.currentTime = Math.max(0, Number(seekTime));
    } catch {
      // Ignore seek errors before metadata is ready.
    }
  }, [seekTime, seekNonce, activeVideoUrl]);

  if (!videoUrl) {
    return (
      <div className="card video-panel">
        <h2>{title}</h2>
        <p>No processed video yet. Upload and analyze an approved clip.</p>
      </div>
    );
  }

  return (
    <div className="card video-panel">
      <h2>{title}</h2>
      {masksAvailable ? (
        <div
          className="snapshot-view-toggle"
          role="group"
          aria-label="Processed video view mode"
        >
          <button
            type="button"
            className={`view-toggle-btn ${previewMode === "full" ? "view-toggle-btn-active" : ""}`}
            onClick={() => setPreviewMode("full")}
          >
            Boxes + labels
          </button>
          <button
            type="button"
            className={`view-toggle-btn ${previewMode === "masks" ? "view-toggle-btn-active" : ""}`}
            onClick={() => setPreviewMode("masks")}
          >
            Segmentation only
          </button>
        </div>
      ) : null}
      <p className="status-text">
        {previewMode === "masks"
          ? "Jersey-colored segmentation only — no bounding boxes or labels."
          : masksAvailable
            ? "Full overlay with boxes and labels. Switch to Segmentation only to hide boxes."
            : "Tracking overlay preview."}
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
  );
}

export default ProcessingVideoPlayer;
