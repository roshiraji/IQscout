import { useEffect, useMemo, useState } from "react";

import { getJobTimeline } from "../API/scoutiqAPI";

const CAMERA_TYPES = new Set([
  "gameplay_segment_started",
  "gameplay_segment_ended",
  "camera_orientation_changed",
  "closeup_started",
  "closeup_ended",
  "gameplay_resumed",
]);
const TRACKING_TYPES = new Set([
  "tracking_quality_dropped",
  "tracking_quality_recovered",
]);
const MOVEMENT_TYPES = new Set([
  "player_became_stationary",
  "player_started_moving",
  "rapid_acceleration",
  "sharp_direction_change",
  "players_converged",
  "players_separated",
]);
const DEFENSE_TYPES = new Set([
  "close_defensive_pressure_started",
  "close_defensive_pressure_ended",
  "separation_threshold_exceeded",
  "possible_help_defense",
  "possible_double_team",
  "likely_matchup_changed",
]);
const BALL_TYPES = new Set([
  "probable_possession_change",
  "ball_near_player",
  "ball_moved_toward_hoop",
  "possible_shot_attempt",
]);

function eventCategory(eventType) {
  if (CAMERA_TYPES.has(eventType)) return "camera";
  if (TRACKING_TYPES.has(eventType)) return "tracking";
  if (MOVEMENT_TYPES.has(eventType)) return "movement";
  if (DEFENSE_TYPES.has(eventType)) return "defense";
  if (BALL_TYPES.has(eventType)) return "ball";
  return "other";
}

function formatTime(seconds) {
  const t = Number(seconds);
  if (!Number.isFinite(t) || t < 0) return "—";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function prettyEventType(type) {
  return String(type || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function EventTable({
  jobId,
  enabled = false,
  refreshKey = 0,
  onEventSelect,
}) {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [events, setEvents] = useState([]);
  const [summary, setSummary] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedDetails, setSelectedDetails] = useState(null);

  const [catCamera, setCatCamera] = useState(true);
  const [catTracking, setCatTracking] = useState(true);
  const [catMovement, setCatMovement] = useState(true);
  const [catDefense, setCatDefense] = useState(true);
  const [catBall, setCatBall] = useState(true);
  const [minConfidence, setMinConfidence] = useState(0);
  const [playerFilter, setPlayerFilter] = useState("all");
  const [segmentFilter, setSegmentFilter] = useState("all");

  useEffect(() => {
    if (!enabled || !jobId) {
      setStatus("idle");
      setEvents([]);
      setSummary(null);
      setError("");
      return undefined;
    }
    let cancelled = false;
    (async () => {
      setStatus("loading");
      setError("");
      try {
        const payload = await getJobTimeline(jobId);
        if (cancelled) return;
        setEvents(payload.events || []);
        setSummary(payload.summary || null);
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        setEvents([]);
        setSummary(null);
        const msg = err.message || "Could not load timeline.";
        if (String(msg).toLowerCase().includes("not found")) {
          setStatus("needs_generate");
          setError("");
        } else {
          setStatus("error");
          setError(msg);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled, jobId, refreshKey]);

  const playerOptions = useMemo(() => {
    const ids = new Set();
    for (const e of events) {
      for (const id of e.person_group_ids || []) ids.add(Number(id));
    }
    return Array.from(ids).sort((a, b) => a - b);
  }, [events]);

  const segmentOptions = useMemo(() => {
    const ids = new Set();
    for (const e of events) {
      if (e.segment_id != null) ids.add(Number(e.segment_id));
    }
    return Array.from(ids).sort((a, b) => a - b);
  }, [events]);

  const filtered = useMemo(() => {
    return events.filter((e) => {
      const cat = eventCategory(e.event_type);
      if (cat === "camera" && !catCamera) return false;
      if (cat === "tracking" && !catTracking) return false;
      if (cat === "movement" && !catMovement) return false;
      if (cat === "defense" && !catDefense) return false;
      if (cat === "ball" && !catBall) return false;
      if (cat === "other") return true;
      if (Number(e.confidence || 0) < minConfidence) return false;
      if (playerFilter !== "all") {
        const want = Number(playerFilter);
        if (!(e.person_group_ids || []).map(Number).includes(want)) return false;
      }
      if (segmentFilter !== "all") {
        if (Number(e.segment_id) !== Number(segmentFilter)) return false;
      }
      return true;
    });
  }, [
    events,
    catCamera,
    catTracking,
    catMovement,
    catDefense,
    catBall,
    minConfidence,
    playerFilter,
    segmentFilter,
  ]);

  const handleSelect = (event) => {
    setSelectedId(event.event_id);
    setSelectedDetails(event);
    if (typeof onEventSelect === "function") {
      onEventSelect(event);
    }
  };

  return (
    <div className="event-table-panel">
      <p className="status-text event-table-note">
        Probable / possible events from screen-space analytics — not official
        play-by-play. Generate trajectories to build the timeline.
      </p>

      {status === "loading" ? <p className="status-text">Loading events…</p> : null}
      {status === "needs_generate" ? (
        <p className="status-text">
          No timeline yet. Use <strong>Generate trajectories</strong> in Player
          Movement to create game events.
        </p>
      ) : null}
      {error ? <p className="trajectory-error">{error}</p> : null}

      {status === "ready" ? (
        <>
          <div className="event-filters">
            <div className="event-filter-group">
              <span className="trajectory-control-title">Categories</span>
              <div className="event-filter-chips">
                {[
                  ["camera", catCamera, setCatCamera, "Camera"],
                  ["tracking", catTracking, setCatTracking, "Tracking"],
                  ["movement", catMovement, setCatMovement, "Movement"],
                  ["defense", catDefense, setCatDefense, "Defensive"],
                  ["ball", catBall, setCatBall, "Ball"],
                ].map(([key, checked, setter, label]) => (
                  <label key={key} className="trajectory-check">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => setter(e.target.checked)}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>

            <div className="event-filter-group event-filter-row">
              <label className="trajectory-inline">
                Min confidence {minConfidence.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(Number(e.target.value))}
                />
              </label>
              <label className="trajectory-inline">
                Player
                <select
                  value={playerFilter}
                  onChange={(e) => setPlayerFilter(e.target.value)}
                >
                  <option value="all">All</option>
                  {playerOptions.map((id) => (
                    <option key={id} value={String(id)}>
                      P{id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="trajectory-inline">
                Segment
                <select
                  value={segmentFilter}
                  onChange={(e) => setSegmentFilter(e.target.value)}
                >
                  <option value="all">All</option>
                  {segmentOptions.map((id) => (
                    <option key={id} value={String(id)}>
                      #{id}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <p className="status-text">
            Showing {filtered.length} of {events.length} events
            {summary
              ? ` · camera ${summary.camera_event_count ?? 0} · movement ${summary.movement_event_count ?? 0} · defense ${summary.defense_event_count ?? 0} · ball ${summary.ball_event_count ?? 0}`
              : ""}
          </p>

          <div className="event-table-wrap">
            <table className="event-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Event</th>
                  <th>Players</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={4}>No events match the current filters.</td>
                  </tr>
                ) : (
                  filtered.map((e) => (
                    <tr
                      key={e.event_id}
                      className={
                        selectedId === e.event_id ? "is-selected" : undefined
                      }
                      onClick={() => handleSelect(e)}
                    >
                      <td>{formatTime(e.start_time)}</td>
                      <td>
                        <span className={`event-cat event-cat-${eventCategory(e.event_type)}`}>
                          {eventCategory(e.event_type)}
                        </span>{" "}
                        {prettyEventType(e.event_type)}
                      </td>
                      <td>
                        {(e.person_group_ids || []).length
                          ? (e.person_group_ids || []).map((id) => `P${id}`).join(", ")
                          : "—"}
                      </td>
                      <td>
                        {Number(e.confidence || 0).toFixed(2)}
                        {e.confidence_label ? ` (${e.confidence_label})` : ""}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {selectedDetails ? (
            <div className="event-details">
              <h3>Event details</h3>
              <p className="status-text">
                {prettyEventType(selectedDetails.event_type)} · frames{" "}
                {selectedDetails.start_frame}–{selectedDetails.end_frame}
                {selectedDetails.segment_id != null
                  ? ` · segment #${selectedDetails.segment_id}`
                  : ""}
              </p>
              <pre className="event-details-json">
                {JSON.stringify(selectedDetails.details || {}, null, 2)}
              </pre>
            </div>
          ) : (
            <p className="status-text">Click an event to seek video and focus trajectories.</p>
          )}
        </>
      ) : null}
    </div>
  );
}

export default EventTable;
