import { useEffect, useState } from "react";

import VideoUpload from "../components/VideoUpload";

import HeatmapViewer from "../components/HeatmapViewer";

import ProcessingStatus from "../components/ProcessingStatus";

import ProcessingVideoPlayer from "../components/ProcessingVideoPlayer_1";

import PlayerSnapshotSelector from "../components/PlayerSnapshotSelector";
import TrackingResults from "../components/TrackingResults";
import EventTable from "../components/EventTable";



function Dashboard() {

  const [isBusy, setIsBusy] = useState(false);

  const [isSavingSelection, setIsSavingSelection] = useState(false);

  const [selectedFileName, setSelectedFileName] = useState("");

  const [validation, setValidation] = useState(null);

  const [snapshot, setSnapshot] = useState(null);

  const [snapshotImageUrl, setSnapshotImageUrl] = useState("");
  const [snapshotMasksImageUrl, setSnapshotMasksImageUrl] = useState("");

  const [labeling, setLabeling] = useState(null);

  const [videoUrl, setVideoUrl] = useState("");

  const [heatmapUrl, setHeatmapUrl] = useState("");

  const [tracking, setTracking] = useState(null);

  const [trackingVideoUrl, setTrackingVideoUrl] = useState("");
  const [trackingIsolatedVideoUrl, setTrackingIsolatedVideoUrl] = useState("");
  const [isolateStatus, setIsolateStatus] = useState(null);

  const [error, setError] = useState("");

  const [jobId, setJobId] = useState("");

  const [job, setJob] = useState(null);

  const [gameContext, setGameContext] = useState({
    league: "WNBA",
    game_date: "",
    home_team: "",
    away_team: "",
  });



  useEffect(() => {
    if (!jobId || job?.status === "ready" || job?.status === "failed") {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || "Could not read tracking job.");
        }
        if (cancelled) return;
        setJob(payload);
        if (payload.status === "ready" && payload.result?.snapshot) {
          setSnapshot(payload.result.snapshot);
          setSnapshotImageUrl(
            `/api/jobs/${encodeURIComponent(jobId)}/files/snapshot`
          );
          setSnapshotMasksImageUrl(
            `/api/jobs/${encodeURIComponent(jobId)}/files/snapshot_masks`
          );
        } else if (payload.status === "failed") {
          setError(payload.error || "Full-clip tracking failed.");
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Job polling failed.");
      }
    };

    poll();
    const timer = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, job?.status]);

  useEffect(() => {
    const generation = tracking?.isolation_generation;
    if (
      !jobId
      || !generation
      || isolateStatus?.status === "ready"
      || isolateStatus?.status === "failed"
    ) {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(
          `/api/jobs/${encodeURIComponent(jobId)}/isolate-status?generation=${encodeURIComponent(generation)}`
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (!cancelled) {
            setIsolateStatus({
              status: "failed",
              stage: "failed",
              error: payload.detail || "Could not read isolate progress.",
            });
          }
          return;
        }
        if (cancelled) return;
        setIsolateStatus(payload);
        if (payload.status === "ready") {
          const stamp = Date.now();
          setTrackingIsolatedVideoUrl(
            `/api/jobs/${encodeURIComponent(jobId)}/files/focused_isolated?v=${stamp}&g=${encodeURIComponent(generation)}`
          );
          setTracking((current) => current ? {
            ...current,
            isolation_available: true,
            isolation_status: "ready",
          } : current);
        }
      } catch (err) {
        if (!cancelled) {
          // Keep polling through a transient connection interruption.
          setIsolateStatus((current) => ({
            ...current,
            transient_error: err.message || "Isolate progress temporarily unavailable.",
          }));
        }
      }
    };

    poll();
    const timer = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    jobId,
    tracking?.isolation_generation,
    isolateStatus?.status,
  ]);



  async function handleAnalyze(file) {

    setIsBusy(true);

    setSelectedFileName(file.name);

    setValidation(null);

    setSnapshot(null);

    setSnapshotImageUrl("");

    setSnapshotMasksImageUrl("");

    setLabeling(null);

    setVideoUrl("");

    setHeatmapUrl("");

    setTracking(null);

    setTrackingVideoUrl("");
    setTrackingIsolatedVideoUrl("");
    setIsolateStatus(null);
    setError("");

    setJobId("");

    setJob(null);



    const formData = new FormData();

    formData.append("file", file);
    Object.entries(gameContext).forEach(([key, value]) => {
      if (value) formData.append(key, value);
    });



    try {

      const response = await fetch("/api/analyze?run_snapshot=false&run_labeling=false", {

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

      if (payload.job_id) {
        setJobId(payload.job_id);
        setJob(payload.job);
      } else if (payload.validation?.usable) {
        setError("Video passed intake but no background tracking job was created.");
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

    if (!snapshot || !jobId) {

      throw new Error("Canonical tracking job or snapshot data is missing.");

    }



    setIsSavingSelection(true);

    setError("");

    // Drop the previous focused preview immediately so re-Confirm cannot
    // keep playing a stale cached segment while the new one renders.
    setTracking(null);
    setTrackingVideoUrl("");
    setTrackingIsolatedVideoUrl("");
    setIsolateStatus(null);

    try {

      const response = await fetch("/api/tracking/selection", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

          job_id: jobId,

          selected_person_group_ids: selectedPlayers.map(
            (player) => player.person_group_id ?? player.track_id
          ),

          selected_players: selectedPlayers,

        }),

      });



      const payload = await response.json().catch(() => ({}));



      if (!response.ok) {
        throw new Error(payload.detail || "Failed to save tracking selection.");
      }

      if (payload.tracking) {
        const stamp = Date.now();
        const groupKey = selectedPlayers
          .map((player) => player.person_group_id ?? player.track_id)
          .filter((id) => id != null)
          .sort((a, b) => Number(a) - Number(b))
          .join("-");
        setTracking(payload.tracking);
        // Cache-bust so a new Confirm replaces the previous focused preview.
        const bust = `v=${stamp}&g=${encodeURIComponent(groupKey)}`;
        setTrackingVideoUrl(
          `/api/jobs/${encodeURIComponent(jobId)}/files/focused?${bust}`
        );
        setIsolateStatus({
          status: payload.tracking.isolation_status || "queued",
          stage: payload.tracking.isolation_status || "queued",
          progress: 0,
          eta_seconds: null,
        });
        if (payload.tracking.isolation_available !== false) {
          setTrackingIsolatedVideoUrl(
            `/api/jobs/${encodeURIComponent(jobId)}/files/focused_isolated?${bust}`
          );
        } else {
          setTrackingIsolatedVideoUrl("");
        }
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
      <div className="hero">
        <h1>ScoutIQ</h1>
        <p>NBA Vision Analytics Platform</p>
      </div>

      <div className="card">
        <h2>Optional Game Context</h2>
        <p className="status-text">
          Team and date context will constrain future roster/RAG identity matches.
          Leave fields blank when unknown; tracking does not depend on identity.
        </p>
        <div className="metadata-grid">
          {[
            ["league", "League"],
            ["game_date", "Game date"],
            ["home_team", "Home team"],
            ["away_team", "Away team"],
          ].map(([key, label]) => (
            <label key={key}>
              <strong>{label}</strong>
              <input
                type={key === "game_date" ? "date" : "text"}
                value={gameContext[key]}
                onChange={(event) =>
                  setGameContext((current) => ({
                    ...current,
                    [key]: event.target.value,
                  }))
                }
              />
            </label>
          ))}
        </div>
      </div>

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
        job={job}
        error={error}
      />



      <PlayerSnapshotSelector
        snapshot={snapshot}
        imageUrl={snapshotImageUrl}
        masksImageUrl={snapshotMasksImageUrl}
        onConfirm={handleConfirmSelection}
        isSaving={isSavingSelection}
      />
      <TrackingResults
        key={tracking?.isolation_generation || "no-tracking"}
        tracking={tracking}
        videoUrl={trackingVideoUrl}
        isolatedVideoUrl={trackingIsolatedVideoUrl}
        isolateStatus={isolateStatus}
        onNewSelection={() => {
          setTracking(null);
          setTrackingVideoUrl("");
          setTrackingIsolatedVideoUrl("");
          setIsolateStatus(null);
          setError("");
        }}
      />

      <div className="card video-panel">
        <h2>Game Events</h2>
        <EventTable />
      </div>

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
