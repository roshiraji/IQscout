import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  generateJobTrajectories,
  getJobBallPreview,
  getJobSegments,
  getJobTrajectories,
  jobFileUrl,
} from "../API/scoutiqAPI";

/** Dashboard-wide screen-space preview palette (Segment 4). */
const TEAM_COLORS = {
  0: "#E74C3C", // Team Red
  1: "#4A90E2", // Team Blue
  2: "#F1C40F", // Officials
};
const COLOR_UNKNOWN = "#9E9E9E";
const COLOR_BALL = "#FF8C1A";
const COLOR_BALL_OUTLINE = "#FFFFFF";

/** Default trail length when Focus/history controls are not yet wired (Segment 5). */
const DEFAULT_HISTORY_SECONDS = 3;
/** Ball trail is shorter than player trails. */
const BALL_HISTORY_SECONDS = 1.25;
/** Break ball stroke when detections teleport or skip many frames. */
const BALL_MAX_FRAME_GAP = 6;
const BALL_MAX_JUMP_NORM = 0.10;
/** Used when FPS / timestamps cannot be inferred. */
const FALLBACK_HISTORY_FRAMES = 60;
/** Trajectory stroke: ~1.5–2px. Opacity varies by age (Segment 3). */
const TRAIL_LINE_WIDTH = 1.75;
const TRAIL_OPACITY = 0.35;
/** Age fade: oldest faint → newest clearly visible (last ~25–30%). */
const TRAIL_FADE_OLD = 0.09;
const TRAIL_FADE_MID = 0.22;
const TRAIL_FADE_NEW = 0.52;
const TRAIL_FADE_HEAD_FRACTION = 0.28;

function teamColor(teamId, personGroupId, role) {
  if (role === "official" || Number(teamId) === 2) return TEAM_COLORS[2];
  if (teamId === 0 || teamId === 1) return TEAM_COLORS[teamId];
  if (personGroupId == null) return COLOR_UNKNOWN;
  return COLOR_UNKNOWN;
}

function roleLabel(teamId, role) {
  if (role === "official" || Number(teamId) === 2) return "Official";
  if (teamId === 0) return "Red";
  if (teamId === 1) return "Blue";
  return "Unknown";
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

/**
 * Prefer summary/segment FPS; else estimate from point timestamps; else null.
 */
function resolveFps(summary, segments, samplePoints) {
  const fromSummary = Number(summary?.fps ?? summary?.fps_estimate);
  if (Number.isFinite(fromSummary) && fromSummary > 1) return fromSummary;

  for (const seg of segments || []) {
    const v = Number(seg?.fps_estimate ?? seg?.fps);
    if (Number.isFinite(v) && v > 1) return v;
  }

  const pts = samplePoints || [];
  if (pts.length >= 3) {
    const a = pts[0];
    const b = pts[pts.length - 1];
    const dt = Number(b.time_seconds) - Number(a.time_seconds);
    const df = Number(b.frame) - Number(a.frame);
    if (dt > 1e-3 && Number.isFinite(df) && df > 0) {
      const est = df / dt;
      if (est > 1 && est < 240) return est;
    }
  }
  return null;
}

/**
 * Keep only the newest historySeconds (or ~60 frames) of a fragment's points.
 * Pass historySeconds=null for full retained history (no trim).
 * Does not mutate stored trajectory data.
 */
function trimToRecentHistory(points, { historySeconds = DEFAULT_HISTORY_SECONDS, fps = null } = {}) {
  if (!Array.isArray(points) || points.length <= 1) return points || [];

  const ordered = [...points].sort(
    (a, b) =>
      (Number(a.frame) - Number(b.frame)) ||
      (Number(a.time_seconds) - Number(b.time_seconds))
  );

  // Full history — no window trim.
  if (historySeconds == null || historySeconds === "full" || historySeconds <= 0) {
    return ordered;
  }
  const windowSec = Number(historySeconds);
  if (!Number.isFinite(windowSec) || windowSec <= 0) return ordered;

  const last = ordered[ordered.length - 1];
  const lastT = Number(last.time_seconds);
  const lastFrame = Number(last.frame);

  if (Number.isFinite(lastT)) {
    const tMin = lastT - windowSec;
    const trimmed = ordered.filter((p) => Number(p.time_seconds) >= tMin);
    if (trimmed.length >= 1) return trimmed;
  }

  const frameWindow =
    Number.isFinite(fps) && fps > 0
      ? Math.max(2, Math.round(fps * windowSec))
      : FALLBACK_HISTORY_FRAMES;
  if (Number.isFinite(lastFrame)) {
    const fMin = lastFrame - frameWindow;
    return ordered.filter((p) => Number(p.frame) >= fMin);
  }

  return ordered.slice(-frameWindow);
}

/**
 * Opacity for trail segment ending at mapped[segmentIndex].
 * Progression: faint (old) → medium → bright (newest ~28%).
 * Short trails stay readable without over-fading.
 */
function trailSegmentAlpha(mapped, segmentIndex, { fadeEnabled = true } = {}) {
  const n = mapped.length;
  if (n < 2) return TRAIL_OPACITY;
  if (!fadeEnabled) return TRAIL_OPACITY;

  // Very short: keep nearly even, slight boost toward the tip.
  if (n <= 3) {
    const u = segmentIndex / Math.max(1, n - 1);
    return TRAIL_OPACITY * (0.75 + 0.35 * u);
  }

  const t0 = Number(mapped[0].t);
  const tN = Number(mapped[n - 1].t);
  const tB = Number(mapped[segmentIndex].t);
  let u;
  if (
    Number.isFinite(t0) &&
    Number.isFinite(tN) &&
    Number.isFinite(tB) &&
    tN > t0 + 1e-6
  ) {
    u = (tB - t0) / (tN - t0);
  } else {
    u = segmentIndex / (n - 1);
  }
  u = Math.min(1, Math.max(0, u));

  const headStart = 1 - TRAIL_FADE_HEAD_FRACTION;
  if (u < headStart) {
    const v = u / headStart;
    return TRAIL_FADE_OLD + (TRAIL_FADE_MID - TRAIL_FADE_OLD) * v;
  }
  const v = (u - headStart) / TRAIL_FADE_HEAD_FRACTION;
  return TRAIL_FADE_MID + (TRAIL_FADE_NEW - TRAIL_FADE_MID) * v;
}

/** Light moving-average on canvas coords only — does not alter stored tracks. */
const SMOOTH_WINDOW = 3;

function smoothMappedPoints(mapped, { enabled = true, windowSize = SMOOTH_WINDOW } = {}) {
  if (!enabled || !Array.isArray(mapped) || mapped.length < 4) {
    return mapped;
  }
  const half = Math.floor(windowSize / 2);
  const out = mapped.map((p) => ({ ...p }));
  // Preserve endpoints so current foot position stays exact.
  for (let i = 1; i < mapped.length - 1; i += 1) {
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let j = i - half; j <= i + half; j += 1) {
      if (j < 0 || j >= mapped.length) continue;
      sx += mapped[j].x;
      sy += mapped[j].y;
      n += 1;
    }
    if (n > 0) {
      out[i].x = sx / n;
      out[i].y = sy / n;
    }
  }
  return out;
}

/** Split ball points into contiguous strokes (no cross-teleport smoothing). */
function splitBallTrailSegments(points, { maxFrameGap = BALL_MAX_FRAME_GAP, maxJumpNorm = BALL_MAX_JUMP_NORM, width = 1, height = 1 } = {}) {
  const segments = [];
  let current = [];
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    if (!current.length) {
      current.push(p);
      continue;
    }
    const prev = current[current.length - 1];
    const frameGap = Math.abs((p.frame || 0) - (prev.frame || 0));
    const dx = (p.x - prev.x) / Math.max(width, 1);
    const dy = (p.y - prev.y) / Math.max(height, 1);
    const jump = Math.hypot(dx, dy);
    if (p.trailBreakBefore || frameGap > maxFrameGap || jump > maxJumpNorm) {
      if (current.length) segments.push(current);
      current = [p];
    } else {
      current.push(p);
    }
  }
  if (current.length) segments.push(current);
  return segments;
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

function drawArrow(ctx, x0, y0, x1, y1, color, size = 6) {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const tipX = x1;
  const tipY = y1;
  const leftX = x1 - size * Math.cos(angle - Math.PI / 6);
  const leftY = y1 - size * Math.sin(angle - Math.PI / 6);
  const rightX = x1 - size * Math.cos(angle + Math.PI / 6);
  const rightY = y1 - size * Math.sin(angle + Math.PI / 6);

  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(leftX, leftY);
  ctx.lineTo(rightX, rightY);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.stroke();
}

/** True when the path has enough motion to warrant a direction arrow. */
function hasDirectionalMotion(mapped, { minPathPx = 8, minLastSegPx = 2.5 } = {}) {
  if (!mapped || mapped.length < 3) return false;
  let pathLen = 0;
  for (let i = 1; i < mapped.length; i += 1) {
    pathLen += Math.hypot(
      mapped[i].x - mapped[i - 1].x,
      mapped[i].y - mapped[i - 1].y
    );
  }
  const last = mapped[mapped.length - 1];
  const prev = mapped[mapped.length - 2];
  const lastSeg = Math.hypot(last.x - prev.x, last.y - prev.y);
  return pathLen >= minPathPx && lastSeg >= minLastSegPx;
}

/**
 * One small arrowhead near the newest point, pulled back so it does not
 * sit on top of the current-position marker.
 */
function drawEndDirectionArrow(ctx, mapped, color, opacity = 0.8) {
  if (!hasDirectionalMotion(mapped)) return;

  const b = mapped[mapped.length - 1];
  const lookback = Math.min(4, mapped.length - 1);
  const a = mapped[mapped.length - 1 - lookback];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist;
  const uy = dy / dist;
  const pull = 5;
  const tipX = b.x - ux * pull;
  const tipY = b.y - uy * pull;
  const baseX = tipX - ux * 7;
  const baseY = tipY - uy * 7;

  ctx.globalAlpha = opacity;
  drawArrow(ctx, baseX, baseY, tipX, tipY, color, 6);
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
  const [ballPoints, setBallPoints] = useState([]);
  const [roleByGroup, setRoleByGroup] = useState({});
  const [bgReady, setBgReady] = useState(false);
  const bgImageRef = useRef(null);

  const [selectionMode, setSelectionMode] = useState("all"); // all | team_red | team_blue | custom
  const [selectedIds, setSelectedIds] = useState([]);
  const [segmentFilter, setSegmentFilter] = useState("all");
  const [timeStart, setTimeStart] = useState(0);
  const [timeEnd, setTimeEnd] = useState(0);

  const [showPaths, setShowPaths] = useState(true);
  const [showArrows, setShowArrows] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  const [showBall, setShowBall] = useState(true);
  const [showMasks, setShowMasks] = useState(false);
  const [showOfficials, setShowOfficials] = useState(true);
  const [fadeByTime, setFadeByTime] = useState(true);
  const [smoothPaths, setSmoothPaths] = useState(true);
  /** 1 | 2 | 3 | "full" — default 3s */
  const [historyMode, setHistoryMode] = useState("3");
  const [variant, setVariant] = useState("analytics");
  const [highlightIds, setHighlightIds] = useState([]);
  const [focusMarker, setFocusMarker] = useState(null);

  const frameWidth = summary?.frame_width || 1280;
  const frameHeight = summary?.frame_height || 720;
  const duration = Number(summary?.duration_seconds) || 0;
  const historySeconds =
    historyMode === "full" ? null : Number(historyMode) || DEFAULT_HISTORY_SECONDS;
  const historyLabel =
    historyMode === "full" ? "full history" : `last ~${historySeconds}s`;

  const loadBackground = useCallback((id) => {
    setBgReady(false);
    const img = new Image();
    let triedFallback = false;
    img.onload = () => {
      bgImageRef.current = img;
      setBgReady(true);
    };
    img.onerror = () => {
      // Prefer masks-only snapshot (no boxes/labels). Fall back if missing.
      if (!triedFallback) {
        triedFallback = true;
        img.src = jobFileUrl(id, "snapshot");
        return;
      }
      bgImageRef.current = null;
      setBgReady(true);
    };
    img.src = jobFileUrl(id, "snapshot_masks");
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

      // Soft-load roles + ball (missing artifacts must not break the preview).
      try {
        const groupsRes = await fetch(jobFileUrl(id, "person_groups"));
        if (groupsRes.ok) {
          const groupsPayload = await groupsRes.json();
          const map = {};
          for (const g of groupsPayload.person_groups || []) {
            const gid = Number(g.person_group_id);
            if (Number.isFinite(gid)) {
              map[gid] = g.role || (Number(g.team_id) === 2 ? "official" : "player");
            }
          }
          setRoleByGroup(map);
        } else {
          setRoleByGroup({});
        }
      } catch {
        setRoleByGroup({});
      }

      try {
        const ball = await getJobBallPreview(id);
        setBallPoints(ball.available ? ball.points || [] : []);
      } catch {
        setBallPoints([]);
      }

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
      const role = roleByGroup[gid] || row.role;
      const isOfficial = role === "official" || teamId === 2;

      if (!showOfficials && selectionMode !== "officials" && isOfficial) {
        return false;
      }
      if (selectionMode === "all") return true;
      if (selectionMode === "team_red") return teamId === 0;
      if (selectionMode === "team_blue") return teamId === 1;
      if (selectionMode === "officials") return isOfficial;
      // Individual focus: keep everyone for spatial context; opacity handles emphasis.
      if (selectionMode === "custom") return true;
      return selectedIds.includes(gid);
    });
  }, [players, selectionMode, selectedIds, roleByGroup, showOfficials]);

  /** Person-group IDs to emphasize (event click or Individual checkboxes). */
  const focusIds = useMemo(() => {
    if (highlightIds.length > 0) return highlightIds;
    if (selectionMode === "custom" && selectedIds.length > 0) {
      return selectedIds;
    }
    return [];
  }, [highlightIds, selectionMode, selectedIds]);

  const focusActive = focusIds.length > 0;
  const focusHeader =
    focusActive
      ? ` · focus ${focusIds.map((id) => `P${id}`).join(", ")}`
      : "";

  const segmentIdFilter =
    segmentFilter === "all" ? null : Number(segmentFilter);

  const filteredDensityPoints = useMemo(() => {
    const points = [];
    let sample = null;
    for (const row of visiblePlayers) {
      for (const frag of filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart,
        timeEnd,
        analyticsSafeOnly: true,
      })) {
        if (!sample && frag.points?.length) sample = frag.points;
        const fps = resolveFps(summary, segments, sample || frag.points);
        points.push(
          ...trimToRecentHistory(frag.points, {
            historySeconds,
            fps,
          })
        );
      }
    }
    return points;
  }, [visiblePlayers, segmentIdFilter, timeStart, timeEnd, summary, segments, historySeconds]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || status !== "ready") return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (bgImageRef.current && showMasks) {
      // Masks are baked into snapshot_masks.jpg — show plate when enabled.
      ctx.drawImage(bgImageRef.current, 0, 0, w, h);
      ctx.fillStyle = "rgba(0,0,0,0.08)";
      ctx.fillRect(0, 0, w, h);
    } else if (bgImageRef.current) {
      // Masks off (default): faint court context only so trails stay primary.
      ctx.fillStyle = "#12141a";
      ctx.fillRect(0, 0, w, h);
      ctx.globalAlpha = 0.18;
      ctx.drawImage(bgImageRef.current, 0, 0, w, h);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.fillStyle = "#12141a";
      ctx.fillRect(0, 0, w, h);
    }

    // Labels live in the HTML chrome (not over the court action).
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

    // Shared FPS estimate for this paint (from summary / segments / samples).
    let fpsSample = null;
    for (const row of visiblePlayers) {
      const frags = collectFragments(row);
      if (frags[0]?.points?.length) {
        fpsSample = frags[0].points;
        break;
      }
    }
    const fps = resolveFps(summary, segments, fpsSample);

    // Draw context players first, then focused on top.
    const drawRows = [...visiblePlayers].sort((a, b) => {
      const aFocus = focusIds.includes(Number(a.person_group_id)) ? 1 : 0;
      const bFocus = focusIds.includes(Number(b.person_group_id)) ? 1 : 0;
      return aFocus - bFocus;
    });

    for (const row of drawRows) {
      const gid = Number(row.person_group_id);
      const isFocused = focusActive && focusIds.includes(gid);
      const isContext = focusActive && !isFocused;
      const color = teamColor(
        row.team_id,
        row.person_group_id,
        roleByGroup[gid] || row.role
      );
      const fragments = filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart,
        timeEnd,
        analyticsSafeOnly: variant === "analytics",
      });

      for (const frag of fragments) {
        const recent = trimToRecentHistory(frag.points, {
          historySeconds,
          fps,
        });
        if (recent.length < 1) continue;

        const mapped = smoothMappedPoints(
          recent.map((p) => ({
            x: Number(p.x_normalized) * w,
            y: Number(p.y_normalized) * h,
            t: Number(p.time_seconds) || 0,
            frame: Number(p.frame) || 0,
          })),
          { enabled: smoothPaths }
        );

        if (mapped.length >= 2) {
          for (let i = 1; i < mapped.length; i += 1) {
            const a = mapped[i - 1];
            const b = mapped[i];
            let alpha = trailSegmentAlpha(mapped, i, { fadeEnabled: fadeByTime });
            if (isContext) {
              alpha *= 0.15;
            } else if (isFocused) {
              alpha = Math.min(0.95, alpha * 1.35 + 0.12);
            }
            const lineW = isFocused ? TRAIL_LINE_WIDTH + 1.1 : TRAIL_LINE_WIDTH;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.globalAlpha = Math.min(1, Math.max(0.05, alpha));
            ctx.strokeStyle = color;
            ctx.lineWidth = lineW;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }

        if (showArrows) {
          const arrowOpacity = isContext ? 0.18 : isFocused ? 1 : 0.85;
          drawEndDirectionArrow(ctx, mapped, color, arrowOpacity);
        }

        ctx.globalAlpha = isContext ? 0.18 : 1;
        const end = mapped[mapped.length - 1];
        ctx.fillStyle = color;
        const endR = isFocused ? 4.5 : 2.75;
        ctx.beginPath();
        ctx.arc(end.x, end.y, endR, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = isFocused ? 2 : 1.25;
        ctx.strokeStyle = isFocused ? "#ffe08a" : "#ffffff";
        ctx.stroke();
        if (isFocused) {
          ctx.beginPath();
          ctx.arc(end.x, end.y, endR + 3.5, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,224,138,0.7)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        if (mapped.length >= 2) {
          const start = mapped[0];
          const startAlpha = trailSegmentAlpha(mapped, 0, { fadeEnabled: fadeByTime });
          ctx.globalAlpha = isContext
            ? 0.08
            : Math.min(0.35, startAlpha + 0.05);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(start.x, start.y, 1.75, 0, Math.PI * 2);
          ctx.fill();
        }

        if (showEvents && focusMarker && isFocused) {
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
            ctx.arc(best.x, best.y, 10, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }

    // Ball: short trail + high-contrast current marker (soft-skip if missing).
    if (showBall && ballPoints.length) {
      const ballFiltered = ballPoints.filter((p) => {
        const t = Number(p.time_seconds);
        if (Number.isFinite(timeStart) && t < timeStart) return false;
        if (Number.isFinite(timeEnd) && t > timeEnd) return false;
        return true;
      });
      const ballRecent = trimToRecentHistory(ballFiltered, {
        historySeconds:
          historySeconds == null
            ? BALL_HISTORY_SECONDS
            : Math.min(BALL_HISTORY_SECONDS, historySeconds),
        fps,
      });
      const mappedRaw = ballRecent.map((p) => ({
        x: Number(p.x_normalized) * w,
        y: Number(p.y_normalized) * h,
        t: Number(p.time_seconds) || 0,
        frame: Number(p.frame) || 0,
        trailBreakBefore: Boolean(p.trail_break_before),
        conf: Number(p.confidence) || 0,
      }));
      const ballSegments = splitBallTrailSegments(mappedRaw, {
        maxFrameGap: BALL_MAX_FRAME_GAP,
        maxJumpNorm: BALL_MAX_JUMP_NORM,
        width: w,
        height: h,
      }).map((seg) => smoothMappedPoints(seg, { enabled: smoothPaths, windowSize: 3 }));

      for (const mappedBall of ballSegments) {
        if (mappedBall.length < 2) continue;
        for (let i = 1; i < mappedBall.length; i += 1) {
          const a = mappedBall[i - 1];
          const b = mappedBall[i];
          const alpha = trailSegmentAlpha(mappedBall, i, { fadeEnabled: fadeByTime });
          ctx.globalAlpha = Math.min(0.7, Math.max(0.08, alpha * 0.85));
          ctx.strokeStyle = COLOR_BALL;
          ctx.lineWidth = 1.5;
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
      const tipSeg = [...ballSegments].reverse().find((seg) => seg.length >= 1);
      if (tipSeg) {
        const tip = tipSeg[tipSeg.length - 1];
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = COLOR_BALL;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = COLOR_BALL_OUTLINE;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "#111";
        ctx.fill();
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
    showBall,
    showMasks,
    fadeByTime,
    smoothPaths,
    duration,
    highlightIds,
    focusMarker,
    variant,
    bgReady,
    frameWidth,
    frameHeight,
    summary,
    segments,
    roleByGroup,
    ballPoints,
    historySeconds,
    focusIds,
    focusActive,
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
          Finish Analyze (tracking ready) to open the screen-space movement preview.
        </p>
      </div>
    );
  }

  return (
    <div className="card trajectory-panel">
      <h2>Screen-Space Movement Preview</h2>
      <p className="status-text trajectory-label">
        Normalized video coordinates for scouting context — not true court positioning.
      </p>
      <p className="trajectory-calib-note">
        Screen-space preview — calibrate the court for court-coordinate analytics.
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
              <span className="trajectory-control-title">Focus</span>
              <div className="trajectory-chip-row">
                {[
                  ["all", "All tracks"],
                  ["team_red", "Team Red"],
                  ["team_blue", "Team Blue"],
                  ["officials", "Officials"],
                  ["custom", "Individual"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={`secondary-button${selectionMode === value ? " is-active" : ""}`}
                    onClick={() => {
                      setSelectionMode(value);
                      if (value === "custom") {
                        // Start empty so checking a player activates focus dimming.
                        setSelectedIds([]);
                        setHighlightIds([]);
                      }
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="trajectory-player-list">
                {players.map((row) => {
                  const gid = Number(row.person_group_id);
                  const role = roleByGroup[gid] || row.role;
                  const isOfficial = role === "official" || row.team_id === 2;
                  if (!showOfficials && selectionMode !== "officials" && isOfficial) {
                    return null;
                  }
                  const checked =
                    selectionMode === "all"
                    || (selectionMode === "team_red" && row.team_id === 0)
                    || (selectionMode === "team_blue" && row.team_id === 1)
                    || (selectionMode === "officials" && isOfficial)
                    || (selectionMode === "custom" && selectedIds.includes(gid));
                  const isFocusChip =
                    selectionMode === "custom" && selectedIds.includes(gid);
                  return (
                    <label
                      key={gid}
                      className={`trajectory-player-item${isFocusChip ? " is-focused" : ""}`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => togglePlayer(gid)}
                      />
                      <span
                        className="trajectory-swatch"
                        style={{ background: teamColor(row.team_id, gid, role) }}
                      />
                      P{gid}
                      {" · "}
                      {roleLabel(row.team_id, role)}
                      {isFocusChip ? " · focus" : ""}
                    </label>
                  );
                })}
              </div>
              {selectionMode === "custom" ? (
                <p className="status-text" style={{ margin: "4px 0 0" }}>
                  Individual focus keeps other players visible at low opacity for context.
                  {focusActive
                    ? ` Focusing ${focusIds.map((id) => `P${id}`).join(", ")}.`
                    : " Check one or more players to focus."}
                </p>
              ) : null}
            </div>

            <div className="trajectory-control-group">
              <span className="trajectory-control-title">History / segment</span>
              <label className="trajectory-inline">
                Trail history
                <select
                  value={historyMode}
                  onChange={(e) => setHistoryMode(e.target.value)}
                >
                  <option value="1">1 second</option>
                  <option value="2">2 seconds</option>
                  <option value="3">3 seconds</option>
                  <option value="full">Full history</option>
                </select>
              </label>
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
                Trajectory lines
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
                  checked={showMasks}
                  onChange={(e) => setShowMasks(e.target.checked)}
                />
                Segmentation masks
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showOfficials}
                  onChange={(e) => setShowOfficials(e.target.checked)}
                />
                Officials
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={showBall}
                  onChange={(e) => setShowBall(e.target.checked)}
                />
                Ball trail
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={fadeByTime}
                  onChange={(e) => setFadeByTime(e.target.checked)}
                />
                Fade trail by age
              </label>
              <label className="trajectory-check">
                <input
                  type="checkbox"
                  checked={smoothPaths}
                  onChange={(e) => setSmoothPaths(e.target.checked)}
                />
                Smooth trails
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
            {(historyLabel || focusHeader) && (
              <div className="trajectory-status-bar" aria-live="polite">
                <span>{historyLabel}</span>
                {focusActive ? <span>{focusHeader.replace(/^ · /, "")}</span> : null}
              </div>
            )}
            <canvas
              ref={canvasRef}
              className="trajectory-canvas"
              width={frameWidth}
              height={frameHeight}
            />
          </div>

          <ul className="trajectory-legend" aria-label="Movement preview legend">
            <li>
              <span className="trajectory-swatch" style={{ background: TEAM_COLORS[1] }} />
              Blue: Team Blue
            </li>
            <li>
              <span className="trajectory-swatch" style={{ background: TEAM_COLORS[0] }} />
              Red: Team Red
            </li>
            <li>
              <span className="trajectory-swatch" style={{ background: TEAM_COLORS[2] }} />
              Yellow: Officials
            </li>
            <li>
              <span
                className="trajectory-swatch trajectory-swatch-ball"
                style={{ background: COLOR_BALL }}
              />
              Orange: Ball
            </li>
            <li>
              <span className="trajectory-swatch" style={{ background: COLOR_UNKNOWN }} />
              Gray: Unknown
            </li>
          </ul>
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
