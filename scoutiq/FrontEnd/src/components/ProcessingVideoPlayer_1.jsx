function ProcessingVideoPlayer({ videoUrl }) {
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
      <video controls width="100%" preload="metadata" playsInline src={videoUrl}>
        Your browser does not support video playback.
      </video>
    </div>
  );
}

export default ProcessingVideoPlayer;
