import { useState } from "react";

import VideoUpload from "../components/VideoUpload";

import HeatmapViewer from "../components/HeatmapViewer";

import ProcessingStatus from "../components/ProcessingStatus";

import ProcessingVideoPlayer from "../components/ProcessingVideoPlayer_1";

import PlayerSnapshotSelector from "../components/PlayerSnapshotSelector";
import TrackingResults from "../components/TrackingResults";



function Dashboard() {

  const [isBusy, setIsBusy] = useState(false);

  const [isSavingSelection, setIsSavingSelection] = useState(false);

  const [selectedFileName, setSelectedFileName] = useState("");

  const [savedVideoPath, setSavedVideoPath] = useState("");

  const [validation, setValidation] = useState(null);

  const [snapshot, setSnapshot] = useState(null);

  const [snapshotImageUrl, setSnapshotImageUrl] = useState("");

  const [labeling, setLabeling] = useState(null);

  const [videoUrl, setVideoUrl] = useState("");

  const [heatmapUrl, setHeatmapUrl] = useState("");

  const [tracking, setTracking] = useState(null);

  const [trackingVideoUrl, setTrackingVideoUrl] = useState("");

  const [error, setError] = useState("");



  async function handleAnalyze(file) {

    setIsBusy(true);

    setSelectedFileName(file.name);

    setSavedVideoPath("");

    setValidation(null);

    setSnapshot(null);

    setSnapshotImageUrl("");

    setLabeling(null);

    setVideoUrl("");

    setHeatmapUrl("");

    setTracking(null);

    setTrackingVideoUrl("");

    setError("");



    const formData = new FormData();

    formData.append("file", file);



    try {

      const response = await fetch("/api/analyze?run_snapshot=true&run_labeling=false", {

        method: "POST",

        body: formData,

      });



      const payload = await response.json().catch(() => ({}));



      if (!response.ok) {

        const detail = payload.detail;

        const message = Array.isArray(detail)

          ? detail.map((item) => item.msg || String(item)).join(", ")

          : detail || "Analysis request failed.";

        throw new Error(message);

      }



      setValidation(payload.validation);

      setSavedVideoPath(payload.saved_path || "");



      if (payload.snapshot) {

        setSnapshot(payload.snapshot);

        setSnapshotImageUrl(

          `/api/files/snapshot/${encodeURIComponent(payload.snapshot.image_filename)}`

        );

      } else if (payload.validation?.usable) {

        setError(

          "Video passed intake but no player snapshot was returned. Restart the API (run_scoutiq.bat or uvicorn on port 8001), then click Analyze again."

        );

      }



      if (payload.labeling?.annotated_video) {

        const filename = payload.labeling.annotated_video.split(/[/\\]/).pop();

        setVideoUrl(`/api/files/annotated/${encodeURIComponent(filename)}`);

      }



      if (payload.labeling?.heatmap_png) {

        const heatmapName = payload.labeling.heatmap_png.split(/[/\\]/).pop();

        setHeatmapUrl(`/api/files/heatmap/${encodeURIComponent(heatmapName)}`);

      }

    } catch (err) {

      setError(err.message || "Something went wrong during analysis.");

    } finally {

      setIsBusy(false);

    }

  }



  async function handleConfirmSelection(selectedPlayers) {

    if (!snapshot || !savedVideoPath) {

      throw new Error("Snapshot data is missing.");

    }



    setIsSavingSelection(true);

    setError("");



    try {

      const response = await fetch("/api/tracking/selection", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

          video_path: savedVideoPath,

          snapshot_frame_number: snapshot.snapshot_frame_number,

          timestamp_seconds: snapshot.timestamp_seconds,

          selected_players: selectedPlayers,

        }),

      });



      const payload = await response.json().catch(() => ({}));



      if (!response.ok) {
        throw new Error(payload.detail || "Failed to save tracking selection.");
      }

      if (payload.tracking) {
        setTracking(payload.tracking);
        const videoName = payload.tracking.tracking_video.split(/[/\\]/).pop();
        setTrackingVideoUrl(`/api/files/tracking/${encodeURIComponent(videoName)}`);
      }

      return payload;

    } catch (err) {

      setError(err.message || "Failed to save tracking selection.");

      return null;

    } finally {

      setIsSavingSelection(false);

    }

  }



  return (

    <div className="dashboard">

      <h1>ScoutIQ WNBA Dashboard</h1>



      <VideoUpload

        onAnalyze={handleAnalyze}

        isBusy={isBusy}

        isSaving={isSavingSelection}

        selectedFileName={selectedFileName}

      />



      <ProcessingStatus
        validation={validation}
        labeling={labeling}
        snapshot={snapshot}
        error={error}
      />



      <PlayerSnapshotSelector

        snapshot={snapshot}

        imageUrl={snapshotImageUrl}

        onConfirm={handleConfirmSelection}

        isSaving={isSavingSelection}

      />

      <TrackingResults tracking={tracking} videoUrl={trackingVideoUrl} />

      <ProcessingVideoPlayer videoUrl={videoUrl} />

      <HeatmapViewer imageUrl={heatmapUrl} />



      {!validation && !error && !isBusy && (

        <p className="status-text">

          Choose an MP4 and click Analyze to validate the clip and pick players to track.

        </p>

      )}

    </div>

  );

}



export default Dashboard;

