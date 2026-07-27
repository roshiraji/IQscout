import { useEffect, useState } from "react";

import VideoUpload from "../components/VideoUpload";
import TrajectoryViewer from "../components/TrajectoryViewer";
import ProcessingStatus from "../components/ProcessingStatus";
import ProcessingVideoPlayer from "../components/ProcessingVideoPlayer_1";
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
  saveTrackingSelection,
  jobFileUrl,
} from "../API/scoutiqAPI";



function Dashboard() {

  const [isBusy, setIsBusy] = useState(false);

  const [isSavingSelection, setIsSavingSelection] = useState(false);

  const [selectedFileName, setSelectedFileName] = useState("");

  const [activeTab, setActiveTab] = useState("analyze");

  const [validation, setValidation] = useState(null);

  const [snapshot, setSnapshot] = useState(null);

  const [snapshotImageUrl, setSnapshotImageUrl] = useState("");
  const [snapshotMasksImageUrl, setSnapshotMasksImageUrl] = useState("");

  const [labeling, setLabeling] = useState(null);

  const [videoUrl, setVideoUrl] = useState("");
  const [allPlayersVideoUrl, setAllPlayersVideoUrl] = useState("");

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

  const [seekTime, setSeekTime] = useState(null);
  const [seekNonce, setSeekNonce] = useState(0);
  const [focusEvent, setFocusEvent] = useState(null);
  const [timelineRefreshKey, setTimelineRefreshKey] = useState(0);


  useEffect(() => {
    if (!jobId || job?.status === "ready" || job?.status === "failed") {
      return undefined;
    }

    let cancelled = false;
    let consecutivePollFailures = 0;
    const poll = async () => {
      try {
        const payload = await getJob(jobId);
        if (cancelled) return;
        consecutivePollFailures = 0;
        setError((current) =>
          current === "Tracking job not found." ? "" : current
        );
        setJob(payload);
        if (payload.status === "ready" && payload.result?.snapshot) {
          setSnapshot(payload.result.snapshot);
          setSnapshotImageUrl(jobFileUrl(jobId, "snapshot"));
          setSnapshotMasksImageUrl(jobFileUrl(jobId, "snapshot_masks"));
          setAllPlayersVideoUrl(jobFileUrl(jobId, "all"));
          setVideoUrl(jobFileUrl(jobId, "all"));
        } else if (payload.status === "failed") {
          // IngestProgress shows job.error — avoid duplicate Error panel below.
        }
      } catch (err) {
        if (cancelled) return;
        consecutivePollFailures += 1;
        // Job state is atomically replaced while tracking. On mapped Windows
        // workspaces a poll can briefly see no JSON; do not paint a false
        // fatal error over a live job.
        if (consecutivePollFailures < 5) return;
        setError(err.message || "Job polling failed.");
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

  // Advance workspace tabs only on pipeline milestones (don't fight manual clicks).
  useEffect(() => {
    if (isBusy || ingestPhase) {
      setActiveTab("analyze");
    }
  }, [isBusy, ingestPhase]);

  useEffect(() => {
    if (snapshot && job?.status === "ready" && !tracking && !isSavingSelection) {
      setActiveTab("select");
    }
  }, [snapshot, job?.status, tracking, isSavingSelection]);

  useEffect(() => {
    if (tracking) {
      setActiveTab("results");
    }
  }, [tracking]);



  async function handleAnalyze(file) {

    setIsBusy(true);
    setIngestPhase("uploading");

    setSelectedFileName(file.name);

    setValidation(null);

    setSnapshot(null);

    setSnapshotImageUrl("");

    setSnapshotMasksImageUrl("");

    setLabeling(null);

    setVideoUrl("");
    setAllPlayersVideoUrl("");

    setTracking(null);

    setTrackingVideoUrl("");
    setTrackingMasksVideoUrl("");
    setTrackingIsolatedVideoUrl("");
    setIsolateStatus(null);
    setConfirmStatus(null);
    setError("");
    setSeekTime(null);
    setFocusEvent(null);
    setTimelineRefreshKey(0);

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



      if (payload.labeling?.annotated_video) {

        const filename = payload.labeling.annotated_video.split(/[/\\]/).pop();

        setVideoUrl(`/api/files/annotated/${encodeURIComponent(filename)}`);

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
        // Open focused film on the snapshot moment — not t=0 where backward
        // lock may have gaps or older bodies.
        const snapT = Number(
          payload.tracking.timestamp_seconds ?? snapshot?.timestamp_seconds
        );
        if (Number.isFinite(snapT)) {
          setSeekTime(snapT);
          setSeekNonce((n) => n + 1);
        }
        setActiveTab("results");
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

      <div className="workspace-shell">
        <div className="workspace-tabs" role="tablist" aria-label="ScoutIQ workspace">
          {[
            {
              id: "analyze",
              label: "Analyze",
              hint: "Upload & track",
              ready: true,
              badge: isBusy || job?.status === "tracking" ? "…" : job?.status === "ready" ? "✓" : null,
            },
            {
              id: "select",
              label: "Select",
              hint: "Pick players",
              ready: Boolean(snapshot) || isSavingSelection,
              badge: isSavingSelection ? "…" : tracking ? "✓" : snapshot ? "•" : null,
            },
            {
              id: "results",
              label: "Results",
              hint: "Film & insights",
              ready: Boolean(tracking),
              badge: isolateStatus?.status === "processing" ? "…" : tracking ? "✓" : null,
            },
              {
              id: "more",
              label: "Events",
              hint: "Timeline & paths",
              ready: true,
              badge: job?.status === "ready" ? "•" : null,
            },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`workspace-tab${activeTab === tab.id ? " workspace-tab-active" : ""}${
                tab.ready ? "" : " workspace-tab-locked"
              }`}
              disabled={!tab.ready && tab.id !== "analyze" && tab.id !== "more"}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="workspace-tab-label">
                {tab.label}
                {tab.badge ? <span className="workspace-tab-badge">{tab.badge}</span> : null}
              </span>
              <span className="workspace-tab-hint">{tab.hint}</span>
            </button>
          ))}
        </div>

        <div className="workspace-panel" role="tabpanel">
          {activeTab === "analyze" && (
            <>
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

              {!validation && !error && !isBusy && (
                <p className="status-text workspace-empty-hint">
                  Choose an MP4 and click Analyze to validate the clip and pick players to track.
                </p>
              )}

              {job?.status === "ready" && allPlayersVideoUrl && (
                <div className="card">
                  <h2>Full-clip tracking preview</h2>
                  <p className="status-text">
                    Boxes and IDs from frame 0 through the end of the clip. Continue to
                    Select to pick players for a focused preview.
                  </p>
                  <ProcessingVideoPlayer videoUrl={allPlayersVideoUrl} />
                </div>
              )}

              {job?.status === "ready" && snapshot && (
                <div className="workspace-next-row">
                  <button type="button" onClick={() => setActiveTab("select")}>
                    Continue to Select →
                  </button>
                </div>
              )}
            </>
          )}

          {activeTab === "select" && (
            <>
              {!snapshot && !isSavingSelection ? (
                <div className="card workspace-empty-card">
                  <h2>Select players</h2>
                  <p className="status-text">
                    Finish Analyze first. When full-clip tracking is ready, pick athletes here.
                  </p>
                  <button type="button" onClick={() => setActiveTab("analyze")}>
                    ← Back to Analyze
                  </button>
                </div>
              ) : (
                <>
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
                  {tracking && (
                    <div className="workspace-next-row">
                      <button type="button" onClick={() => setActiveTab("results")}>
                        View Results →
                      </button>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {activeTab === "results" && (
            <>
              {!tracking ? (
                <div className="card workspace-empty-card">
                  <h2>Results</h2>
                  <p className="status-text">
                    Confirm a player selection to unlock focused film, isolate, and Matchup Outcomes.
                  </p>
                  <button type="button" onClick={() => setActiveTab("select")}>
                    ← Back to Select
                  </button>
                </div>
              ) : (
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
                  onPlayOutcomeUpdated={(payload) => {
                    if (!payload?.coach_insights) return;
                    setTracking((current) =>
                      current
                        ? {
                            ...current,
                            coach_insights: payload.coach_insights,
                            coach_narrative:
                              payload.coach_narrative || current.coach_narrative,
                            play_outcome: payload.play_outcome ?? current.play_outcome,
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
                    setFocusEvent(null);
                    setActiveTab("select");
                  }}
                />
              )}
            </>
          )}

          {activeTab === "more" && (
            <>
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
                masksImageUrl={snapshotMasksImageUrl}
                playVideoUrl={allPlayersVideoUrl || videoUrl}
                masksVideoUrl={trackingMasksVideoUrl}
                confirmSelectedIds={tracking?.selected_person_group_ids || null}
                onGenerateComplete={() => setTimelineRefreshKey((k) => k + 1)}
              />

              <ProcessingVideoPlayer
                videoUrl={trackingVideoUrl || videoUrl}
                masksVideoUrl={trackingMasksVideoUrl}
                seekTime={seekTime}
                seekNonce={seekNonce}
                title={trackingVideoUrl ? "Focused Players" : "Processed Video"}
              />

              {!jobId && (
                <p className="status-text workspace-empty-hint">
                  After Analyze finishes, generate trajectories here and browse Game Events.
                  Click an event to seek video and highlight paths.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>

  );

}



export default Dashboard;
