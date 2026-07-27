import { useEffect, useState } from "react";

import VideoUpload from "../components/VideoUpload";
import TrajectoryViewer from "../components/TrajectoryViewer";
import ProcessingStatus from "../components/ProcessingStatus";
import PlayerSnapshotSelector from "../components/PlayerSnapshotSelector";
import TrackingResults from "../components/TrackingResults";
import ConfirmProgress from "../components/ConfirmProgress";
import IngestProgress from "../components/IngestProgress";
import EventTable from "../components/EventTable";
import {
  analyzeVideo,
  getJob,
  getConfirmStatus,
  getIsolateStatus,
  getWnbaTeams,
  saveTrackingSelection,
  jobFileUrl,
} from "../API/scoutiqAPI";



function Dashboard() {

  const [isBusy, setIsBusy] = useState(false);

  const [isSavingSelection, setIsSavingSelection] = useState(false);

  const [selectedFileName, setSelectedFileName] = useState("");

  const [validation, setValidation] = useState(null);

  const [snapshot, setSnapshot] = useState(null);

  const [snapshotImageUrl, setSnapshotImageUrl] = useState("");
  const [snapshotMasksImageUrl, setSnapshotMasksImageUrl] = useState("");

  const [labeling, setLabeling] = useState(null);

  const [tracking, setTracking] = useState(null);

  const [trackingVideoUrl, setTrackingVideoUrl] = useState("");
  const [trackingMasksVideoUrl, setTrackingMasksVideoUrl] = useState("");
  const [trackingIsolatedVideoUrl, setTrackingIsolatedVideoUrl] = useState("");
  const [isolateStatus, setIsolateStatus] = useState(null);
  const [confirmStatus, setConfirmStatus] = useState(null);
  const [ingestPhase, setIngestPhase] = useState(null);

  const [error, setError] = useState("");

  const [jobId, setJobId] = useState("");

  const [job, setJob] = useState(null);

  const [gameContext, setGameContext] = useState({
    league: "WNBA",
    game_date: "",
    home_team: "",
    away_team: "",
  });
  const [wnbaTeams, setWnbaTeams] = useState([]);
  const [wnbaTeamsNote, setWnbaTeamsNote] = useState("");

  const [seekTime, setSeekTime] = useState(null);
  const [seekNonce, setSeekNonce] = useState(0);
  const [focusEvent, setFocusEvent] = useState(null);
  const [timelineRefreshKey, setTimelineRefreshKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const payload = await getWnbaTeams();
        if (cancelled) return;
        setWnbaTeams(Array.isArray(payload.teams) ? payload.teams : []);
        setWnbaTeamsNote(payload.note || "");
      } catch (err) {
        if (!cancelled) {
          setWnbaTeams([]);
          setWnbaTeamsNote(err.message || "Could not load WNBA teams.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!jobId || job?.status === "ready" || job?.status === "failed") {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const payload = await getJob(jobId);
        if (cancelled) return;
        setJob(payload);
        if (payload.status === "ready" && payload.result?.snapshot) {
          setSnapshot(payload.result.snapshot);
          setSnapshotImageUrl(jobFileUrl(jobId, "snapshot"));
          setSnapshotMasksImageUrl(jobFileUrl(jobId, "snapshot_masks"));
        } else if (payload.status === "failed") {
          // IngestProgress shows job.error — avoid duplicate Error panel below.
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
        const payload = await getIsolateStatus(jobId, generation);
        if (cancelled) return;
        setIsolateStatus(payload);
        if (payload.status === "ready") {
          const stamp = Date.now();
          setTrackingIsolatedVideoUrl(
            jobFileUrl(
              jobId,
              "focused_isolated",
              `v=${stamp}&g=${encodeURIComponent(generation)}`
            )
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

  // Poll Confirm progress while the selection POST is in flight (parallel requests).
  useEffect(() => {
    if (!jobId || !isSavingSelection) {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const payload = await getConfirmStatus(jobId);
        if (!cancelled) setConfirmStatus(payload);
      } catch {
        // Transient — keep polling.
      }
    };

    poll();
    const timer = window.setInterval(poll, 400);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, isSavingSelection]);



  async function handleAnalyze(file) {

    setIsBusy(true);
    setIngestPhase("uploading");

    setSelectedFileName(file.name);

    setValidation(null);

    setSnapshot(null);

    setSnapshotImageUrl("");

    setSnapshotMasksImageUrl("");

    setLabeling(null);

    setTracking(null);

    setTrackingVideoUrl("");
    setTrackingMasksVideoUrl("");
    setTrackingIsolatedVideoUrl("");
    setIsolateStatus(null);
    setConfirmStatus(null);
    setError("");

    setJobId("");

    setJob(null);



    try {
      const payload = await analyzeVideo(file, gameContext, {
        onUploadProgress: ({ phase }) => setIngestPhase(phase),
      });

      setValidation(payload.validation);
      setIngestPhase(null);

      if (payload.job_id) {
        setJobId(payload.job_id);
        setJob(payload.job);
      } else if (payload.validation?.usable) {
        setError("Video passed intake but no background tracking job was created.");
      }

    } catch (err) {

      setError(err.message || "Something went wrong during analysis.");
      setIngestPhase(null);

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
    setTrackingMasksVideoUrl("");
    setTrackingIsolatedVideoUrl("");
    setIsolateStatus(null);
    setConfirmStatus({
      status: "processing",
      stage: "saving_selection",
      progress: 0.02,
      detail: "Confirming your selected players…",
    });

    try {
      const payload = await saveTrackingSelection({
        job_id: jobId,
        selected_person_group_ids: selectedPlayers.map(
          (player) => player.person_group_id ?? player.track_id
        ),
        selected_players: selectedPlayers,
      });

      if (payload.tracking) {
        const stamp = Date.now();
        const groupKey = selectedPlayers
          .map((player) => player.person_group_id ?? player.track_id)
          .filter((id) => id != null)
          .sort((a, b) => Number(a) - Number(b))
          .join("-");
        setTracking(payload.tracking);
        const bust = `v=${stamp}&g=${encodeURIComponent(groupKey)}`;
        setTrackingVideoUrl(jobFileUrl(jobId, "focused", bust));
        setTrackingMasksVideoUrl(jobFileUrl(jobId, "focused_masks", bust));
        setConfirmStatus({
          status: "ready",
          stage: "ready",
          progress: 1,
          detail: "Focused preview ready. Isolate continues in the background.",
        });
        setIsolateStatus({
          status: payload.tracking.isolation_status || "queued",
          stage: payload.tracking.isolation_status || "queued",
          progress: 0,
          eta_seconds: null,
          effective_mode: payload.tracking.isolation_mode,
          requested_mode: payload.tracking.isolation_requested_mode,
          method: payload.tracking.isolation_method,
          ai_fallback: payload.tracking.isolation_ai_fallback,
        });
        if (payload.tracking.isolation_available !== false) {
          setTrackingIsolatedVideoUrl(
            jobFileUrl(jobId, "focused_isolated", bust)
          );
        } else {
          setTrackingIsolatedVideoUrl("");
        }
      }

      return payload;

    } catch (err) {

      setError(err.message || "Failed to save tracking selection.");
      setConfirmStatus({
        status: "failed",
        stage: "failed",
        progress: 0,
        detail: err.message || "Failed to save tracking selection.",
        error: err.message || "Failed to save tracking selection.",
      });
      return null;

    } finally {

      setIsSavingSelection(false);

    }

  }



  return (

    <div className="dashboard">
      <div className="hero">
        <h1>ScoutIQ</h1>
        <p>WNBA Vision Analytics Platform</p>
      </div>

      <div className="card game-context-card">
        <h2>Optional Game Context</h2>
        <p className="status-text">
          Team and date context will constrain future roster/RAG identity matches.
          Leave fields blank when unknown; tracking does not depend on identity.
        </p>
        <div className="game-context-grid">
          <div className="game-context-field">
            <span className="game-context-field-label">League</span>
            <input
              type="text"
              value={gameContext.league}
              onChange={(event) =>
                setGameContext((current) => ({
                  ...current,
                  league: event.target.value,
                }))
              }
            />
          </div>
          <div className="game-context-field">
            <span className="game-context-field-label">Game date</span>
            <input
              type="date"
              className="game-context-date"
              value={gameContext.game_date}
              onChange={(event) =>
                setGameContext((current) => ({
                  ...current,
                  game_date: event.target.value,
                }))
              }
            />
          </div>
          <div className="game-context-field">
            <span className="game-context-field-label">Home team</span>
            {wnbaTeams.length > 0 ? (
              <select
                value={gameContext.home_team}
                onChange={(event) =>
                  setGameContext((current) => ({
                    ...current,
                    home_team: event.target.value,
                  }))
                }
              >
                <option value="">Select home team</option>
                {wnbaTeams.map((team) => (
                  <option key={`home-${team.abbreviation}`} value={team.abbreviation}>
                    {team.display_name} ({team.abbreviation})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                placeholder="e.g. IND"
                value={gameContext.home_team}
                onChange={(event) =>
                  setGameContext((current) => ({
                    ...current,
                    home_team: event.target.value,
                  }))
                }
              />
            )}
          </div>
          <div className="game-context-field">
            <span className="game-context-field-label">Away team</span>
            {wnbaTeams.length > 0 ? (
              <select
                value={gameContext.away_team}
                onChange={(event) =>
                  setGameContext((current) => ({
                    ...current,
                    away_team: event.target.value,
                  }))
                }
              >
                <option value="">Select away team</option>
                {wnbaTeams.map((team) => (
                  <option key={`away-${team.abbreviation}`} value={team.abbreviation}>
                    {team.display_name} ({team.abbreviation})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                placeholder="e.g. ATL"
                value={gameContext.away_team}
                onChange={(event) =>
                  setGameContext((current) => ({
                    ...current,
                    away_team: event.target.value,
                  }))
                }
              />
            )}
          </div>
        </div>
        {wnbaTeamsNote ? (
          <p className="status-text game-context-teams-note">{wnbaTeamsNote}</p>
        ) : null}
      </div>

      <VideoUpload

        onAnalyze={handleAnalyze}

        isBusy={isBusy}

        isSaving={isSavingSelection}

        selectedFileName={selectedFileName}

      />

      <IngestProgress
        isBusy={isBusy}
        ingestPhase={ingestPhase}
        selectedFileName={selectedFileName}
        job={job}
        validation={validation}
      />

      <ProcessingStatus
        validation={validation}
        labeling={labeling}
        snapshot={snapshot}
        job={job}
        error={error}
        hideJobProgress
      />


      <PlayerSnapshotSelector
        snapshot={snapshot}
        imageUrl={snapshotImageUrl}
        masksImageUrl={snapshotMasksImageUrl}
        jobId={jobId}
        onConfirm={handleConfirmSelection}
        isSaving={isSavingSelection}
      />
      <ConfirmProgress
        confirmStatus={confirmStatus}
        isolateStatus={isolateStatus}
        isSaving={isSavingSelection}
      />
      <TrackingResults
        key={tracking?.isolation_generation || "no-tracking"}
        tracking={tracking}
        videoUrl={trackingVideoUrl}
        masksVideoUrl={trackingMasksVideoUrl}
        isolatedVideoUrl={trackingIsolatedVideoUrl}
        isolateStatus={isolateStatus}
        jobId={jobId}
        seekTime={seekTime}
        seekNonce={seekNonce}
        onPossessionUpdated={(payload) => {
          if (!payload?.coach_insights) return;
          setTracking((current) =>
            current
              ? {
                  ...current,
                  coach_insights: payload.coach_insights,
                  coach_narrative:
                    payload.coach_narrative || current.coach_narrative,
                  possession: payload.possession || current.possession,
                  offense_team_id: payload.possession?.offense_team_id,
                  defense_team_id: payload.possession?.defense_team_id,
                }
              : current
          );
        }}
        onNewSelection={() => {
          setTracking(null);
          setTrackingVideoUrl("");
          setTrackingMasksVideoUrl("");
          setTrackingIsolatedVideoUrl("");
          setIsolateStatus(null);
          setConfirmStatus(null);
          setError("");
        }}
      />

      <div className="card video-panel">
        <h2>Game Events</h2>
        <EventTable
          jobId={jobId}
          enabled={Boolean(jobId && job?.status === "ready")}
          refreshKey={timelineRefreshKey}
          onEventSelect={(event) => {
            const t = Number(event?.start_time);
            if (Number.isFinite(t)) {
              setSeekTime(t);
              setSeekNonce((n) => n + 1);
            }
            setFocusEvent(event);
          }}
        />
      </div>

      <TrajectoryViewer
        jobId={jobId}
        enabled={Boolean(jobId && job?.status === "ready")}
        focusEvent={focusEvent}
        onGenerateComplete={() => setTimelineRefreshKey((k) => k + 1)}
      />

    </div>

  );

}



export default Dashboard;
