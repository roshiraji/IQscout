import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  generateJobTrajectories,
  getJobSegments,
  getJobTrajectories,
  jobFileUrl,
} from "../API/scoutiqAPI";

const TEAM_COLORS = {
  0: "#e74c3c",
  1: "#3498db",
};

function teamColor(teamId, personGroupId) {
  if (teamId === 0 || teamId === 1) return TEAM_COLORS[teamId];
  const palette = ["#f1c40f", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#ecf0f1"];
  const idx = Math.abs(Number(personGroupId) || 0) % palette.length;
  return palette[idx];
}

function collectFragments(playerRow) {
  if (Array.isArray(playerRow.path_fragments) && playerRow.path_fragments.length) {
    return playerRow.path_fragments;
  }
  if (Array.isArray(playerRow.points) && playerRow.points.length) {
    return [
      {
        fragment_id: 1,
        segment_id: playerRow.points[0]?.segment_id ?? null,
        usable_for_analytics: true,
        points: playerRow.points,
      },
    ];
  }
  return [];
}

function filterFragments(fragments, { segmentId, timeStart, timeEnd, analyticsSafeOnly = false }) {
  return fragments
    .map((frag) => {
      if (analyticsSafeOnly && frag.usable_for_analytics === false) {
        return { ...frag, points: [] };
      }
      const points = (frag.points || []).filter((p) => {
        if (analyticsSafeOnly && p.usable_for_analytics === false) return false;
        const t = Number(p.time_seconds);
        if (Number.isFinite(timeStart) && t < timeStart) return false;
        if (Number.isFinite(timeEnd) && t > timeEnd) return false;
        if (segmentId != null && Number(p.segment_id) !== Number(segmentId)) {
          return false;
        }
        return true;
      });
      return { ...frag, points };
    })
    .filter((frag) => frag.points.length > 0);
}

function buildDensityGrid(points, binsX, binsY) {
  const grid = Array.from({ length: binsY }, () => Array(binsX).fill(0));
  let max = 0;
  let count = 0;
  for (const p of points) {
    const xn = Number(p.x_normalized);
    const yn = Number(p.y_normalized);
    if (!Number.isFinite(xn) || !Number.isFinite(yn)) continue;
    const ix = Math.min(binsX - 1, Math.max(0, Math.floor(xn * binsX)));
    const iy = Math.min(binsY - 1, Math.max(0, Math.floor(yn * binsY)));
    grid[iy][ix] += 1;
    max = Math.max(max, grid[iy][ix]);
    count += 1;
  }
  return { grid, max, count };
}

function densityFill(t) {
  // Cool → warm ramp in normalized screen-space density.
  const r = Math.round(40 + 215 * t);
  const g = Math.round(180 * (1 - t) + 40 * t);
  const b = Math.round(255 * (1 - t));
  const a = 0.12 + 0.5 * t;
  return `rgba(${r},${g},${b},${a})`;
}

function drawDensityOverlay(ctx, width, height, grid, maxCount) {
  if (!maxCount) return;
  const binsY = grid.length;
  const binsX = grid[0]?.length || 0;
  if (!binsX || !binsY) return;
  const cellW = width / binsX;
  const cellH = height / binsY;
  for (let y = 0; y < binsY; y += 1) {
    for (let x = 0; x < binsX; x += 1) {
      const value = grid[y][x];
      if (!value) continue;
      ctx.fillStyle = densityFill(value / maxCount);
      ctx.fillRect(x * cellW, y * cellH, cellW + 0.6, cellH + 0.6);
    }
  }
}

function drawArrow(ctx, x0, y0, x1, y1, color) {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const size = 8;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(
    x1 - size * Math.cos(angle - Math.PI / 6),
    y1 - size * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x1 - size * Math.cos(angle + Math.PI / 6),
    y1 - size * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

function TrajectoryViewer({
  jobId,
  enabled = false,
  focusEvent = null,
  onGenerateComplete,
}) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState([]);
  const [summary, setSummary] = useState(null);
  const [segments, setSegments] = useState([]);
  const [players, setPlayers] = useState([]);
  const [bgReady, setBgReady] = useState(false);
  const bgImageRef = useRef(null);

  const [selectionMode, setSelectionMode] = useState("all"); // all | team_red | team_blue | custom
  const [selectedIds, setSelectedIds] = useState([]);
  const [segmentFilter, setSegmentFilter] = useState("all");
  const [timeStart, setTimeStart] = useState(0);
  const [timeEnd, setTimeEnd] = useState(0);

  const [showPaths, setShowPaths] = useState(true);
  const [showArrows, setShowArrows] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const [fadeByTime, setFadeByTime] = useState(true);
  const [variant, setVariant] = useState("analytics");
  const [highlightIds, setHighlightIds] = useState([]);
  const [focusMarker, setFocusMarker] = useState(null);

  const frameWidth = summary?.frame_width || 1280;
  const frameHeight = summary?.frame_height || 720;
  const duration = Number(summary?.duration_seconds) || 0;

  const loadBackground = useCallback((id) => {
    setBgReady(false);
    const img = new Image();
    img.onload = () => {
      bgImageRef.current = img;
      setBgReady(true);
    };
    img.onerror = () => {
      bgImageRef.current = null;
      setBgReady(true);
    };
    img.src = jobFileUrl(id, "snapshot");
  }, []);

  const loadExisting = useCallback(
    async (id, nextVariant = variant) => {
      const [traj, segs] = await Promise.all([
        getJobTrajectories(id, nextVariant),
        getJobSegments(id),
      ]);
      setSummary(traj.summary || null);
      setPlayers(traj.trajectories || []);
      setSegments(segs.segments || []);
      const dur = Number(traj.summary?.duration_seconds) || 0;
      setTimeStart(0);
      setTimeEnd(dur);
      setSelectedIds(
        (traj.trajectories || []).map((p) => Number(p.person_group_id))
      );
      setStatus("ready");
      setError("");
    },
    [variant]
  );

  const handleGenerate = useCallback(async () => {
    if (!jobId) return;
    setStatus("generating");
    setError("");
    setWarnings([]);
    try {
      const result = await generateJobTrajectories(jobId);
      setWarnings(result.warnings || []);
      loadBackground(jobId);
      await loadExisting(jobId, variant);
      if (typeof onGenerateComplete === "function") {
        onGenerateComplete(result);
      }
    } catch (err) {
      setStatus("error");
      setError(err.message || "Trajectory generation failed.");
    }
  }, [jobId, loadBackground, loadExisting, variant, onGenerateComplete]);

  useEffect(() => {
    if (!focusEvent) return;
    const ids = (focusEvent.person_group_ids || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id));
    if (ids.length) {
      setSelectionMode("custom");
      setSelectedIds(ids);
      setHighlightIds(ids);
    } else {
      setHighlightIds([]);
    }
    if (focusEvent.segment_id != null) {
      setSegmentFilter(String(focusEvent.segment_id));
    }
    const t0 = Number(focusEvent.start_time);
    const t1 = Number(focusEvent.end_time ?? focusEvent.start_time);
    if (Number.isFinite(t0)) {
      const pad = 1.5;
      const dur = Number(summary?.duration_seconds) || Math.max(t1, t0) + pad;
      setTimeStart(Math.max(0, t0 - pad));
      setTimeEnd(Math.min(dur, (Number.isFinite(t1) ? t1 : t0) + pad));
      setFocusMarker({ t: t0, ids });
    }
    setShowEvents(true);
  }, [focusEvent, summary?.duration_seconds]);

  useEffect(() => {
    if (!enabled || !jobId) {
      setStatus("idle");
      return;
    }
    let cancelled = false;
    (async () => {
      setStatus("loading");
      setError("");
      loadBackground(jobId);
      try {
        await loadExisting(jobId, variant);
        if (cancelled) return;
      } catch (err) {
        if (cancelled) return;
        // Not generated yet — stay idle with generate CTA.
        setStatus("needs_generate");
        setPlayers([]);
        setSegments([]);
        setSummary(null);
        if (!String(err.message || "").toLowerCase().includes("not found")) {
          setError(err.message || "Could not load trajectories.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled, jobId, loadBackground, loadExisting, variant]);

  const visiblePlayers = useMemo(() => {
    return players.filter((row) => {
      const gid = Number(row.person_group_id);
      const teamId = row.team_id;
      if (selectionMode === "all") return true;
      if (selectionMode === "team_red") return teamId === 0;
      if (selectionMode === "team_blue") return teamId === 1;
      return selectedIds.includes(gid);
    });
  }, [players, selectionMode, selectedIds]);

  const segmentIdFilter =
    segmentFilter === "all" ? null : Number(segmentFilter);

  const filteredDensityPoints = useMemo(() => {
    const points = [];
    for (const row of visiblePlayers) {
      for (const frag of filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart,
        timeEnd,
        analyticsSafeOnly: true,
      })) {
        points.push(...frag.points);
      }
    }
    return points;
  }, [visiblePlayers, segmentIdFilter, timeStart, timeEnd]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || status !== "ready") return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (bgImageRef.current) {
      ctx.drawImage(bgImageRef.current, 0, 0, w, h);
      ctx.fillStyle = "rgba(0,0,0,0.28)";
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.fillStyle = "#12141a";
      ctx.fillRect(0, 0, w, h);
    }

    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "12px sans-serif";
    ctx.fillText("Screen-space trajectory preview", 12, 18);

    if (showHeatmap) {
      const binsX = 48;
      const binsY = Math.max(1, Math.round(binsX * (h / Math.max(w, 1))));
      const { grid, max } = buildDensityGrid(filteredDensityPoints, binsX, binsY);
      drawDensityOverlay(ctx, w, h, grid, max);
    }

    if (!showPaths) {
      ctx.globalAlpha = 1;
      return;
    }

    for (const row of visiblePlayers) {
      const gid = Number(row.person_group_id);
      const isHighlight = highlightIds.includes(gid);
      const color = teamColor(row.team_id, row.person_group_id);
      const fragments = filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart,
        timeEnd,
        analyticsSafeOnly: variant === "analytics",
      });

      for (const frag of fragments) {
        const pts = frag.points;
        if (pts.length < 1) continue;

        const mapped = pts.map((p) => ({
          x: Number(p.x_normalized) * w,
          y: Number(p.y_normalized) * h,
          t: Number(p.time_seconds) || 0,
        }));

        if (mapped.length >= 2) {
          for (let i = 1; i < mapped.length; i += 1) {
            const a = mapped[i - 1];
            const b = mapped[i];
            let alpha = 0.9;
            if (fadeByTime && duration > 0) {
              alpha = 0.25 + 0.75 * (b.t / duration);
            }
            if (highlightIds.length && !isHighlight) {
              alpha *= 0.25;
            }
            ctx.strokeStyle = color;
            ctx.globalAlpha = Math.min(1, Math.max(0.15, alpha));
            ctx.lineWidth = isHighlight ? 3.6 : 2.2;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();

            if (showArrows && i % 8 === 0) {
              ctx.globalAlpha = Math.min(1, alpha);
              drawArrow(ctx, a.x, a.y, b.x, b.y, color);
            }
          }
        }

        ctx.globalAlpha = highlightIds.length && !isHighlight ? 0.35 : 1;
        const start = mapped[0];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(start.x, start.y, isHighlight ? 6 : 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();

        const end = mapped[mapped.length - 1];
        ctx.fillStyle = color;
        const endSize = isHighlight ? 10 : 8;
        ctx.fillRect(end.x - endSize / 2, end.y - endSize / 2, endSize, endSize);
        ctx.strokeStyle = "#fff";
        ctx.strokeRect(end.x - endSize / 2, end.y - endSize / 2, endSize, endSize);

        if (showArrows && mapped.length >= 2) {
          const a = mapped[mapped.length - 2];
          const b = mapped[mapped.length - 1];
          drawArrow(ctx, a.x, a.y, b.x, b.y, color);
        }

        if (showEvents && focusMarker && isHighlight) {
          let best = null;
          let bestDt = Infinity;
          for (const p of mapped) {
            const dt = Math.abs(p.t - focusMarker.t);
            if (dt < bestDt) {
              bestDt = dt;
              best = p;
            }
          }
          if (best && bestDt < 2.5) {
            ctx.globalAlpha = 1;
            ctx.strokeStyle = "#ffe08a";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(best.x, best.y, 12, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }
    ctx.globalAlpha = 1;
  }, [
    status,
    visiblePlayers,
    filteredDensityPoints,
    segmentIdFilter,
    timeStart,
    timeEnd,
    showPaths,
    showArrows,
    showHeatmap,
    showEvents,
    fadeByTime,
    duration,
    highlightIds,
    focusMarker,
    variant,
    bgReady,
    frameWidth,
    frameHeight,
  ]);

  function togglePlayer(gid) {
    setSelectionMode("custom");
    setSelectedIds((current) =>
      current.includes(gid)
        ? current.filter((id) => id !== gid)
        : [...current, gid]
    );
  }

  if (!enabled) {
    return (
      <div className="card trajectory-panel">
        <h2>Player Movement</h2>
        <p className="status-text">
          Finish Analyze (tracking ready) to generate a screen-space trajectory preview.
        </p>
      </div>
    );
  }

  return (
    <div className="card trajectory-panel">
      <h2>Player Movement</h2>
      <p className="status-text trajectory-label">
        Screen-space trajectory preview — normalized video coordinates, not true court
        positioning.
      </p>

      <div className="trajectory-toolbar">
        <button
          type="button"
          onClick={handleGenerate}
          disabled={status === "generating"}
        >
          {status === "generating"
            ? "Generating…"
            : status === "ready"
              ? "Regenerate trajectories"
              : "Generate trajectories"}
        </button>
        <label className="trajectory-inline">
          Data
          <select
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
            disabled={status === "generating"}
          >
            <option value="analytics">Analytics-safe</option>
            <option value="raw">Raw (all frames)</option>
          </select>
        </label>
      </div>

      {error ? <p className="trajectory-error">{error}</p> : null}
      {warnings.length > 0 ? (
        <ul className="trajectory-warnings">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      ) : null}

      {status === "needs_generate" || status === "idle" ? (
        <p className="status-text">
          Click Generate to build segments and trajectories from existing tracking
          (no YOLO re-run).
        </p>
      ) : null}

      {status === "loading" || status === "generating" ? (
        <p className="status-text">Working…</p>
      ) : null}

      {status === "ready" ? (
        <>
          <div className="trajectory-controls">
            <div className="trajectory-control-group">
              <span className="trajectory-control-title">Players</span>
              <div className="trajectory-chip-row">
                {[
                  ["all", "All players"],
                  ["team_red", "Team Red"],
                  ["team_blue", "Team Blue"],
                  ["custom", "Selected"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={`secondary-button${selectionMode === value ? " is-active" : ""}`}
                    onClick={() => setSelectionMode(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="trajectory-player-list">
                {players.map((row) => {
                  const gid = Number(row.person_group_id);
                  const checked =
                    selectionMode === "all"
                    || (selectionMode === "team_red" && row.team_id === 0)
                    || (selectionMode === "team_blue" && row.team_id === 1)
                    || (selectionMode === "custom" && selectedIds.includes(gid));
                  return (
                    <label key={gid} className="trajectory-player-item">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => togglePlayer(gid)}
                      />
                      <span
                        className="trajectory-swatch"
                        style={{ background: teamColor(row.team_id, gid) }}
                      />
                      P{gid}
                      {row.team_id === 0
                        ? " · Red"
                        : row.team_id === 1
                          ? " · Blue"
                          : ""}
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="trajectory-control-group">
              <span className="trajectory-control-title">Segment / time</span>
              <label className="trajectory-inline">
                Gameplay segment
                <select
                  value={segmentFilter}
                  onChange={(e) => setSegmentFilter(e.target.value)}
                >
                  <option value="all">Full video range</option>
                  {segments.map((seg) => (
                    <option key={seg.segment_id} value={String(seg.segment_id)}>
                      #{seg.segment_id} {seg.camera_state}
                      {seg.usable_for_analytics ? "" : " (excluded)"} ·{" "}
                      {Number(seg.start_time).toFixed(1)}s–
                      {Number(seg.end_time).toFixed(1)}s
                    </option>
                  ))}
                </select>
              </label>
              <label className="trajectory-inline">
                Start {timeStart.toFixed(1)}s
                <input
                  type="range"
                  min={0}
                  max={duration || 0}
                  step={0.1}
                  value={timeStart}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setTimeStart(Math.min(v, timeEnd));
                  }}
                />
              </label>
              <label className="trajectory-inline">
                End {timeEnd.toFixed(1)}s
                <input
                  type="range"
                  min={0}
                  max={duration || 0}
                  step={0.1}
                  value={timeEnd}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setTimeEnd(Math.max(v, timeStart));
                  }}
                />
              </label>
            </div>

            <div className="trajectory-control-group">
              <span className="trajectory-control-title">Layers</span>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showPaths}
                  onChange={(e) => setShowPaths(e.target.checked)}
                />
                Player paths
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showArrows}
                  onChange={(e) => setShowArrows(e.target.checked)}
                />
                Direction arrows
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={fadeByTime}
                  onChange={(e) => setFadeByTime(e.target.checked)}
                />
                Fade path by time
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
                Heatmap density
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showEvents}
                  onChange={(e) => setShowEvents(e.target.checked)}
                />
                Event focus marker
              </label>
            </div>
          </div>

          <div className="trajectory-canvas-wrap">
            <canvas
              ref={canvasRef}
              className="trajectory-canvas"
              width={frameWidth}
              height={frameHeight}
            />
          </div>

          <p className="status-text trajectory-legend">
            ○ start · ■ end · colors by team (Red / Blue). Heatmap density uses the same
            analytics-safe points (closeups/transitions excluded) in normalized
            screen-space.
          </p>
          {summary ? (
            <p className="status-text">
              {summary.total_analytics_safe_points ?? summary.trajectory_points}{" "}
              analytics-safe points · {summary.path_fragment_count ?? "—"} fragments
              {showHeatmap
                ? ` · density from ${filteredDensityPoints.length} filtered points`
                : ""}{" "}
              · {summary.coordinate_space || "normalized screen-space coordinates"}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export default TrajectoryViewer;
