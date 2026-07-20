import { useRef, useState } from "react";

function VideoUpload({ onAnalyze, isBusy, isSaving = false, selectedFileName }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
  }

  function handleAnalyzeClick() {
    if (!file) {
      alert("Choose an MP4 file first.");
      return;
    }
    onAnalyze(file);
  }

  return (
    <div className="card upload-panel">
      <h2>Upload WNBA Video</h2>
      <p className="upload-hint">MP4 only · 5–15 seconds · then click players to track</p>
      <div className="upload-row">
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,.mp4"
          onChange={handleFileChange}
          disabled={isBusy}
        />
        <button type="button" onClick={handleAnalyzeClick} disabled={isBusy || !file}>
          {isBusy ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      {file && <p className="file-name">Selected: {file.name}</p>}
      {isBusy && selectedFileName && (
        <p className="status-text">
          Ingesting {selectedFileName} — watch the progress card below for upload,
          validation, and full-clip tracking status.
        </p>
      )}
      {isSaving && (
        <p className="status-text">
          Confirming your selection — building focused + segmentation previews and
          Matchup Outcomes. Watch the progress card below for live status.
        </p>
      )}
    </div>
  );
}

export default VideoUpload;
