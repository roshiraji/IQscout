import { useEffect, useRef } from "react";

function ProcessingVideoPlayer({ videoUrl, seekTime = null, seekNonce = 0 }) {
  const videoRef = useRef(null);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || seekTime == null || !Number.isFinite(Number(seekTime))) return;
    try {
      el.currentTime = Math.max(0, Number(seekTime));
    } catch {
      // Ignore seek errors before metadata is ready.
    }
  }, [seekTime, seekNonce]);

  if (!videoUrl) {
    return (
      <div className="card video-panel">
        <h2>Processed Video</h2>
        <p>No processed video yet. Upload and analyze an approved clip.</p>
      </div>
    );
  }

  return (
    <div className="card video-panel">
      <h2>Processed Video</h2>
      <video
        ref={videoRef}
        controls
        width="100%"
        preload="metadata"
        playsInline
        src={videoUrl}
      >
        Your browser does not support video playback.
      </video>
    </div>
  );
}

export default ProcessingVideoPlayer;
