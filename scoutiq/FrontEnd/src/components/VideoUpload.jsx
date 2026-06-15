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
    <div className="upload-panel">
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
          Validating {selectedFileName} and finding the best player snapshot… this can take 20–30
          seconds.
        </p>
      )}
      {isSaving && (
        <p className="status-text">
          Running ByteTrack on your selected players across the rest of the clip…
        </p>
      )}
    </div>
  );
}

export default VideoUpload;
