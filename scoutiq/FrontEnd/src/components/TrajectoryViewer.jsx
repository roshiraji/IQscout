import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  generateJobTrajectories,
  getJobCameraMotion,
  getJobSegments,
  getJobTrajectories,
  jobFileUrl,
} from "../API/scoutiqAPI";
import PlayheadScrubber from "./PlayheadScrubber";
import { formatClock } from "./VideoScrubber";

const TEAM_COLORS = {
  0: "#e74c3c",
  1: "#3498db",
};

function teamIdOf(row) {
  const raw = row?.team_id;
  if (raw === 0 || raw === 1) return raw;
  if (raw === "0" || raw === "1") return Number(raw);
  const jersey = String(row?.jersey_color_name || "").trim().toLowerCase();
  if (jersey === "red") return 0;
  if (jersey === "blue") return 1;
  return null;
}

function teamColor(teamId, personGroupId) {
  const id = teamId === "0" || teamId === "1" ? Number(teamId) : teamId;
  if (id === 0 || id === 1) return TEAM_COLORS[id];
  const palette = ["#f1c40f", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#ecf0f1"];
  const idx = Math.abs(Number(personGroupId) || 0) % palette.length;
  return palette[idx];
}

/** Trail stroke width; opacity comes from age-fade + focus dimming. */
const TRAIL_LINE_WIDTH = 2.2;
const TRAIL_OPACITY = 0.55;
/** Age fade: oldest faint → newest clearly visible (last ~28%). */
const TRAIL_FADE_OLD = 0.09;
const TRAIL_FADE_MID = 0.22;
const TRAIL_FADE_NEW = 0.52;
const TRAIL_FADE_HEAD_FRACTION = 0.28;
/** Default context dim (0–100). Higher = fainter non-focused trails. */
const DEFAULT_CONTEXT_DIM = 85;
/** Default background dim (0–100). Higher = darker film under the trails. */
const DEFAULT_BACKGROUND_DIM = 20;
/** How far to look for a neighboring frame's camera transform. */
const MOTION_FRAME_SEARCH = 12;

/** camera_motion frame row → 3x3 matrix (row-major, h22 == 1). */
function mat3FromMotionRow(row) {
  return [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], 1];
}

function mat3Invert(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  return [
    A / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    B / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    C / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ];
}

/** Apply a homography to a normalized point; null when it projects to infinity. */
function applyMat3(m, x, y) {
  const w = m[6] * x + m[7] * y + m[8];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-9) return null;
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

/**
 * Opacity for trail segment ending at mapped[segmentIndex].
 * Progression: faint (old) → medium → bright (newest ~28%).
 */
function trailSegmentAlpha(mapped, segmentIndex, { fadeEnabled = true } = {}) {
  const n = mapped.length;
  if (n < 2) return TRAIL_OPACITY;
  if (!fadeEnabled) return TRAIL_OPACITY;

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

function idsForMode(players, mode, confirmSelectedIds = []) {
  if (mode === "all") {
    return players.map((row) => Number(row.person_group_id));
  }
  if (mode === "team_red") {
    return players
      .filter((row) => teamIdOf(row) === 0)
      .map((row) => Number(row.person_group_id));
  }
  if (mode === "team_blue") {
    return players
      .filter((row) => teamIdOf(row) === 1)
      .map((row) => Number(row.person_group_id));
  }
  // "Selected" = Confirm picks when available, else whatever the user checked.
  if (Array.isArray(confirmSelectedIds) && confirmSelectedIds.length) {
    const allowed = new Set(confirmSelectedIds.map(Number));
    return players
      .map((row) => Number(row.person_group_id))
      .filter((gid) => allowed.has(gid));
  }
  return [];
}

function playerMatchesSelection(row, mode, selectedIds) {
  const gid = Number(row.person_group_id);
  if (mode === "all") return true;
  if (mode === "team_red") return teamIdOf(row) === 0;
  if (mode === "team_blue") return teamIdOf(row) === 1;
  // Individual focus: keep everyone drawn; opacity handles emphasis.
  if (mode === "custom") return true;
  return selectedIds.includes(gid);
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
  masksImageUrl = "",
  playVideoUrl = "",
  masksVideoUrl = "",
  confirmSelectedIds = null,
}) {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
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
  /** 0 = others full brightness, 100 = others nearly invisible (when focus is on). */
  const [contextDim, setContextDim] = useState(DEFAULT_CONTEXT_DIM);
  /** Darkens the film/snapshot under the canvas so trails read as high contrast. */
  const [backgroundDim, setBackgroundDim] = useState(DEFAULT_BACKGROUND_DIM);
  const [cameraMotion, setCameraMotion] = useState(null);
  const [anchorToCourt, setAnchorToCourt] = useState(false);
  const [variant, setVariant] = useState("analytics");
  const [highlightIds, setHighlightIds] = useState([]);
  const [focusMarker, setFocusMarker] = useState(null);
  const [bgMode, setBgMode] = useState("masks"); // labels | masks — prefer no boxes
  const [playhead, setPlayhead] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playTimerRef = useRef(null);
  const videoDrivenRef = useRef(false);

  const frameWidth = summary?.frame_width || 1280;
  const frameHeight = summary?.frame_height || 720;
  const duration = Number(summary?.duration_seconds) || 0;
  const masksAvailable = Boolean(masksImageUrl);
  const activePlayUrl =
    bgMode === "masks" && masksVideoUrl
      ? masksVideoUrl
      : playVideoUrl;
  const hasPlayVideo = Boolean(activePlayUrl);

  useEffect(() => {
    setIsPlaying(false);
  }, [activePlayUrl]);

  const loadBackground = useCallback(
    (id, mode = bgMode) => {
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
      const useMasks = mode === "masks" && masksImageUrl;
      img.src = useMasks ? masksImageUrl : jobFileUrl(id, "snapshot");
    },
    [bgMode, masksImageUrl]
  );

  const loadExisting = useCallback(
    async (id, nextVariant = variant) => {
      const [traj, segs, motion] = await Promise.all([
        getJobTrajectories(id, nextVariant),
        getJobSegments(id),
        // Court anchoring is optional polish — never fail the load over it.
        getJobCameraMotion(id).catch(() => null),
      ]);
      const rows = traj.trajectories || [];
      setSummary(traj.summary || null);
      setPlayers(rows);
      setSegments(segs.segments || []);
      setCameraMotion(motion?.available ? motion : null);
      const dur = Number(traj.summary?.duration_seconds) || 0;
      setTimeStart(0);
      setTimeEnd(dur);
      setPlayhead(0);
      setIsPlaying(false);
      setStatus("ready");
      setError("");
      return rows;
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
      loadBackground(jobId, bgMode);
      await loadExisting(jobId, variant);
      if (typeof onGenerateComplete === "function") {
        onGenerateComplete(result);
      }
    } catch (err) {
      setStatus("error");
      setError(err.message || "Trajectory generation failed.");
    }
  }, [
    jobId,
    loadBackground,
    loadExisting,
    variant,
    bgMode,
    onGenerateComplete,
  ]);

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
      const start = Math.max(0, t0 - pad);
      const end = Math.min(dur, (Number.isFinite(t1) ? t1 : t0) + pad);
      setTimeStart(start);
      setTimeEnd(end);
      setPlayhead(Number.isFinite(t1) ? t1 : t0);
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
      loadBackground(jobId, bgMode);
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
    // bgMode is handled by its own effect once ready — don't re-fetch JSON on toggle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, jobId, loadExisting, variant]);

  // Reload the canvas background when Boxes ↔ Segmentation toggles.
  useEffect(() => {
    if (!enabled || !jobId || status !== "ready") return;
    loadBackground(jobId, bgMode);
  }, [bgMode, enabled, jobId, status, loadBackground]);

  // Keep the active filter's ID set in sync whenever players reload or the
  // Confirm selection changes. Custom mode is owned by the checkboxes.
  useEffect(() => {
    if (status !== "ready") return;
    if (selectionMode === "custom") return;
    setSelectedIds(idsForMode(players, selectionMode, confirmSelectedIds));
  }, [players, selectionMode, confirmSelectedIds, status]);

  const applySelectionMode = useCallback(
    (mode) => {
      setSelectionMode(mode);
      if (mode === "custom") {
        // Prefer Confirm picks as the focus set; otherwise start empty so
        // checking a player activates focus dimming.
        const seeded = idsForMode(players, "custom", confirmSelectedIds);
        setSelectedIds(seeded);
      } else {
        setSelectedIds(idsForMode(players, mode, confirmSelectedIds));
      }
      setHighlightIds([]);
    },
    [players, confirmSelectedIds]
  );

  const visiblePlayers = useMemo(() => {
    return players.filter((row) =>
      playerMatchesSelection(row, selectionMode, selectedIds)
    );
  }, [players, selectionMode, selectedIds]);

  /** Person-group IDs to emphasize (event click or Individual checkboxes). */
  const focusIds = useMemo(() => {
    if (highlightIds.length > 0) return highlightIds;
    if (selectionMode === "custom" && selectedIds.length > 0) {
      return selectedIds;
    }
    return [];
  }, [highlightIds, selectionMode, selectedIds]);

  const focusActive = focusIds.length > 0;

  const motionIndex = useMemo(() => {
    if (!cameraMotion?.available) return null;
    const byFrame = new Map();
    for (const row of cameraMotion.frames || []) {
      if (!Array.isArray(row) || row.length < 10) continue;
      byFrame.set(Number(row[0]), row);
    }
    if (byFrame.size === 0) return null;
    return { byFrame, fps: Number(cameraMotion.fps) || 30 };
  }, [cameraMotion]);

  // The playhead frame is the view every trail gets warped into, so marks land
  // on the court the camera is showing right now.
  const displayWarp = useMemo(() => {
    if (!anchorToCourt || !motionIndex) return null;
    const target = Math.round(playhead * motionIndex.fps);
    let row = null;
    for (let delta = 0; delta <= MOTION_FRAME_SEARCH && !row; delta += 1) {
      row =
        motionIndex.byFrame.get(target - delta) ||
        motionIndex.byFrame.get(target + delta) ||
        null;
    }
    if (!row) return null;
    const inverse = mat3Invert(mat3FromMotionRow(row));
    if (!inverse) return null;
    return { anchorId: Number(row[1]), inverse };
  }, [anchorToCourt, motionIndex, playhead]);

  // Map slider → opacity scale for non-focused trails / markers.
  const contextTrailScale = Math.max(0.02, 1 - contextDim / 100);
  const contextMarkerAlpha = Math.max(0.04, contextTrailScale * 1.15);

  // Player chips: for team filters only list that team; for Selected list
  // everyone so the user can pick who to focus (others stay dimmed on canvas).
  const listedPlayers = useMemo(() => {
    if (selectionMode === "all" || selectionMode === "custom") return players;
    if (selectionMode === "team_red" || selectionMode === "team_blue") {
      return players.filter((row) =>
        playerMatchesSelection(row, selectionMode, selectedIds)
      );
    }
    if (selectedIds.length) {
      return players.filter((row) =>
        selectedIds.includes(Number(row.person_group_id))
      );
    }
    return players;
  }, [players, selectionMode, selectedIds]);

  const segmentIdFilter =
    segmentFilter === "all" ? null : Number(segmentFilter);

  // Trails grow with the playhead — only moments up to the head are drawn.
  const scrubEnd = Math.min(
    timeEnd,
    Number.isFinite(playhead) ? playhead : timeEnd
  );
  const scrubStart = Math.min(timeStart, scrubEnd);

  const filteredDensityPoints = useMemo(() => {
    const points = [];
    for (const row of visiblePlayers) {
      for (const frag of filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart: scrubStart,
        timeEnd: scrubEnd,
        analyticsSafeOnly: true,
      })) {
        points.push(...frag.points);
      }
    }
    return points;
  }, [visiblePlayers, segmentIdFilter, scrubStart, scrubEnd]);

  useEffect(() => {
    if (playhead < scrubStart) setPlayhead(scrubStart);
    if (playhead > timeEnd) setPlayhead(timeEnd);
  }, [timeStart, timeEnd, scrubStart, playhead]);

  // Keep the clip locked to the playhead when scrubbing / seeking.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hasPlayVideo) return;
    if (videoDrivenRef.current) return;
    if (!Number.isFinite(playhead)) return;
    if (Math.abs((Number(video.currentTime) || 0) - playhead) < 0.05) return;
    try {
      video.currentTime = Math.max(0, playhead);
    } catch {
      // Ignore seeks before metadata is ready.
    }
  }, [playhead, hasPlayVideo, activePlayUrl]);

  // Drive the playhead from the real clip while it plays.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hasPlayVideo) return undefined;

    const onTime = () => {
      const t = Number(video.currentTime) || 0;
      videoDrivenRef.current = true;
      if (t >= timeEnd - 0.04) {
        video.pause();
        setIsPlaying(false);
        setPlayhead(timeEnd);
      } else if (t < scrubStart) {
        setPlayhead(scrubStart);
      } else {
        setPlayhead(t);
      }
      window.requestAnimationFrame(() => {
        videoDrivenRef.current = false;
      });
    };
    const onEnded = () => {
      setIsPlaying(false);
      setPlayhead(timeEnd);
    };
    const onPause = () => {
      if (!videoDrivenRef.current) setIsPlaying(false);
    };

    video.addEventListener("timeupdate", onTime);
    video.addEventListener("ended", onEnded);
    video.addEventListener("pause", onPause);
    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("pause", onPause);
    };
  }, [hasPlayVideo, activePlayUrl, timeEnd, scrubStart]);

  // Play / pause the clip (fallback timer only when no video is available).
  useEffect(() => {
    const video = videoRef.current;
    if (hasPlayVideo && video) {
      if (playTimerRef.current) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
      if (isPlaying) {
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => setIsPlaying(false));
        }
      } else {
        video.pause();
      }
      return undefined;
    }

    if (!isPlaying) {
      if (playTimerRef.current) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
      return undefined;
    }
    playTimerRef.current = window.setInterval(() => {
      setPlayhead((current) => {
        const next = current + 0.1;
        if (next >= timeEnd) {
          setIsPlaying(false);
          return timeEnd;
        }
        return next;
      });
    }, 100);
    return () => {
      if (playTimerRef.current) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [isPlaying, timeEnd, hasPlayVideo, activePlayUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || status !== "ready") return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Slider-controlled scrim: the film darkens, the trails do not, so the
    // contrast between court footage and marks is the user's call.
    const scrim = Math.min(0.95, Math.max(0, backgroundDim / 100));
    if (hasPlayVideo) {
      // Video sits under the canvas — the scrim is the only thing we can dim.
      ctx.fillStyle = `rgba(0,0,0,${scrim})`;
      ctx.fillRect(0, 0, w, h);
    } else if (bgImageRef.current) {
      ctx.drawImage(bgImageRef.current, 0, 0, w, h);
      ctx.fillStyle = `rgba(0,0,0,${scrim})`;
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.fillStyle = "#12141a";
      ctx.fillRect(0, 0, w, h);
    }

    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "12px sans-serif";
    ctx.fillText(
      displayWarp
        ? "Court-anchored trajectory preview (camera motion removed)"
        : "Screen-space trajectory preview",
      12,
      18
    );

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
      const isFocused = focusActive && focusIds.includes(gid);
      const isContext = focusActive && !isFocused;
      const color = teamColor(teamIdOf(row), row.person_group_id);
      const fragments = filterFragments(collectFragments(row), {
        segmentId: segmentIdFilter,
        timeStart: scrubStart,
        timeEnd: scrubEnd,
        analyticsSafeOnly: variant === "analytics",
      });

      for (const frag of fragments) {
        const pts = frag.points;
        if (pts.length < 1) continue;

        const mapped = [];
        let pendingBreak = false;
        for (const p of pts) {
          let xn = Number(p.x_normalized);
          let yn = Number(p.y_normalized);
          if (displayWarp) {
            const row = motionIndex.byFrame.get(Number(p.frame));
            // A different anchor means a different camera setup entirely —
            // those points cannot be placed in this view.
            if (!row || Number(row[1]) !== displayWarp.anchorId) {
              pendingBreak = true;
              continue;
            }
            const onAnchor = applyMat3(mat3FromMotionRow(row), xn, yn);
            const inView =
              onAnchor && applyMat3(displayWarp.inverse, onAnchor[0], onAnchor[1]);
            if (!inView) {
              pendingBreak = true;
              continue;
            }
            [xn, yn] = inView;
          }
          mapped.push({
            x: xn * w,
            y: yn * h,
            t: Number(p.time_seconds) || 0,
            interp: Boolean(p.interpolated),
            brk: pendingBreak,
          });
          pendingBreak = false;
        }
        if (mapped.length < 1) continue;

        if (mapped.length >= 2) {
          for (let i = 1; i < mapped.length; i += 1) {
            const a = mapped[i - 1];
            const b = mapped[i];
            if (b.brk) continue;
            let alpha = trailSegmentAlpha(mapped, i, { fadeEnabled: fadeByTime });
            if (isContext) {
              alpha *= contextTrailScale;
            } else if (isFocused) {
              alpha = Math.min(0.95, alpha * 1.35 + 0.12);
            }
            const segAlpha = Math.min(1, Math.max(0.05, alpha));
            const segInterpolated = a.interp || b.interp;
            const dash = segInterpolated ? [5, 5] : [];
            const lineWidth = isFocused
              ? TRAIL_LINE_WIDTH + 1.1
              : TRAIL_LINE_WIDTH;

            // Dark halo under the trail keeps it legible over any court color.
            ctx.setLineDash(dash);
            ctx.strokeStyle = "rgba(0,0,0,0.55)";
            ctx.globalAlpha = segAlpha;
            ctx.lineWidth = lineWidth + 1.6;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();

            ctx.strokeStyle = color;
            ctx.globalAlpha = segInterpolated ? segAlpha * 0.85 : segAlpha;
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
            ctx.setLineDash([]);

            if (showArrows && i % 8 === 0) {
              ctx.globalAlpha = isContext
                ? contextMarkerAlpha
                : Math.min(1, alpha);
              drawArrow(ctx, a.x, a.y, b.x, b.y, color);
            }
          }
        }

        ctx.globalAlpha = isContext ? contextMarkerAlpha : 1;
        const start = mapped[0];
        const startAlpha = trailSegmentAlpha(mapped, 0, { fadeEnabled: fadeByTime });
        ctx.globalAlpha = isContext
          ? contextTrailScale * 0.55
          : Math.min(0.45, startAlpha + 0.08);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(start.x, start.y, isFocused ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.globalAlpha = isContext ? contextMarkerAlpha : 1;
        ctx.stroke();

        const end = mapped[mapped.length - 1];
        ctx.globalAlpha = isContext ? contextMarkerAlpha : 1;
        ctx.fillStyle = color;
        const endSize = isFocused ? 11 : 8;
        ctx.fillRect(end.x - endSize / 2, end.y - endSize / 2, endSize, endSize);
        ctx.strokeStyle = isFocused ? "#ffe08a" : "#fff";
        ctx.lineWidth = isFocused ? 2 : 1;
        ctx.strokeRect(end.x - endSize / 2, end.y - endSize / 2, endSize, endSize);
        if (isFocused) {
          ctx.beginPath();
          ctx.arc(end.x, end.y, endSize / 2 + 3.5, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,224,138,0.7)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        if (
          showArrows &&
          mapped.length >= 2 &&
          !isContext &&
          !mapped[mapped.length - 1].brk
        ) {
          const a = mapped[mapped.length - 2];
          const b = mapped[mapped.length - 1];
          ctx.globalAlpha = isFocused ? 1 : 0.85;
          drawArrow(ctx, a.x, a.y, b.x, b.y, color);
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
    scrubStart,
    scrubEnd,
    showPaths,
    showArrows,
    showHeatmap,
    showEvents,
    fadeByTime,
    backgroundDim,
    displayWarp,
    motionIndex,
    contextTrailScale,
    contextMarkerAlpha,
    focusIds,
    focusActive,
    focusMarker,
    variant,
    bgReady,
    hasPlayVideo,
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
    setHighlightIds([]);
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
          (no YOLO re-run). After that, Team Red / Team Blue / Selected hide every
          other trail.
        </p>
      ) : null}

      {status === "loading" || status === "generating" ? (
        <p className="status-text">Working…</p>
      ) : null}

      {status === "ready" ? (
        <>
          <div
            className="snapshot-view-toggle"
            role="group"
            aria-label="Trajectory background mode"
          >
            <button
              type="button"
              className={`view-toggle-btn ${bgMode === "labels" ? "view-toggle-btn-active" : ""}`}
              onClick={() => setBgMode("labels")}
            >
              Boxes + labels
            </button>
            <button
              type="button"
              className={`view-toggle-btn ${bgMode === "masks" ? "view-toggle-btn-active" : ""}`}
              onClick={() => setBgMode("masks")}
              disabled={!masksVideoUrl && !masksAvailable}
              title={
                masksVideoUrl
                  ? "Play the segmentation-only clip under the marks"
                  : masksAvailable
                    ? "Snapshot masks only — Confirm to unlock masks video"
                    : "Masks not available — re-Analyze the clip"
              }
            >
              Segmentation only
            </button>
          </div>

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
                    onClick={() => applySelectionMode(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <p className="status-text">
                {selectionMode === "custom"
                  ? focusActive
                    ? `Focusing ${focusIds.map((id) => `P${id}`).join(", ")} — other players stay visible but dimmed.`
                    : "Individual focus: check players to emphasize them. Everyone else stays visible at low opacity."
                  : selectionMode === "team_red"
                    ? `Showing Team Red only (${visiblePlayers.length} of ${players.length}).`
                    : selectionMode === "team_blue"
                      ? `Showing Team Blue only (${visiblePlayers.length} of ${players.length}).`
                      : `Showing ${visiblePlayers.length} of ${players.length} players.`}
                {focusActive && selectionMode !== "custom"
                  ? ` · focus ${focusIds.map((id) => `P${id}`).join(", ")}`
                  : ""}
              </p>
              <label
                className={`trajectory-inline${focusActive ? "" : " is-disabled"}`}
                title={
                  focusActive
                    ? "How faint non-focused player paths appear"
                    : "Select players (Selected mode) or an event to dim the others"
                }
              >
                Dim others {contextDim}%
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={contextDim}
                  disabled={!focusActive}
                  onChange={(e) => setContextDim(Number(e.target.value))}
                />
              </label>
              <div className="trajectory-player-list">
                {listedPlayers.map((row) => {
                  const gid = Number(row.person_group_id);
                  const checked =
                    selectionMode === "all"
                    || selectedIds.includes(gid);
                  const focused =
                    selectionMode === "custom" && selectedIds.includes(gid);
                  const teamId = teamIdOf(row);
                  return (
                    <label
                      key={gid}
                      className={`trajectory-player-item${
                        focused ? " is-focused" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => togglePlayer(gid)}
                      />
                      <span
                        className="trajectory-swatch"
                        style={{ background: teamColor(teamId, gid) }}
                      />
                      P{gid}
                      {teamId === 0
                        ? " · Red"
                        : teamId === 1
                          ? " · Blue"
                          : ""}
                      {focused ? " · focus" : ""}
                    </label>
                  );
                })}
                {listedPlayers.length === 0 ? (
                  <p className="status-text">
                    {selectionMode === "custom"
                      ? "No players loaded yet."
                      : "No players match this team filter."}
                  </p>
                ) : null}
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
              <label
                className={`trajectory-check${motionIndex ? "" : " is-disabled"}`}
                title={
                  motionIndex
                    ? "Remove camera pan/zoom so trails stay on the same spot of floor"
                    : "Camera motion has not been estimated — re-generate trajectories"
                }
              >
                <input
                  type="checkbox"
                  checked={anchorToCourt && Boolean(motionIndex)}
                  disabled={!motionIndex}
                  onChange={(e) => setAnchorToCourt(e.target.checked)}
                />
                Anchor trails to court
              </label>
              <label
                className="trajectory-inline"
                title="Darken the film under the marks so trails read at higher contrast"
              >
                Dim film {backgroundDim}%
                <input
                  type="range"
                  min={0}
                  max={90}
                  step={1}
                  value={backgroundDim}
                  onChange={(e) => setBackgroundDim(Number(e.target.value))}
                />
              </label>
            </div>
          </div>

          <div className={`trajectory-canvas-wrap${hasPlayVideo ? " has-play-video" : ""}`}>
            <div
              className="trajectory-stage"
              style={{ aspectRatio: `${frameWidth} / ${frameHeight}` }}
            >
              {hasPlayVideo ? (
                <video
                  key={activePlayUrl}
                  ref={videoRef}
                  className="trajectory-play-video"
                  src={activePlayUrl}
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={(event) => {
                    try {
                      event.currentTarget.currentTime = Math.max(0, playhead);
                    } catch {
                      /* ignore */
                    }
                  }}
                />
              ) : null}
              <canvas
                ref={canvasRef}
                className="trajectory-canvas"
                width={frameWidth}
                height={frameHeight}
              />
            </div>
            <PlayheadScrubber
              value={playhead}
              min={timeStart}
              max={timeEnd || duration}
              onChange={(t) => {
                setIsPlaying(false);
                setPlayhead(t);
              }}
              playing={isPlaying}
              onTogglePlay={() => {
                if (isPlaying) {
                  setIsPlaying(false);
                  return;
                }
                if (playhead >= (timeEnd || duration) - 0.05) {
                  setPlayhead(timeStart);
                  const video = videoRef.current;
                  if (video) {
                    try {
                      video.currentTime = timeStart;
                    } catch {
                      /* ignore */
                    }
                  }
                }
                setIsPlaying(true);
              }}
              markerSeconds={
                focusMarker && Number.isFinite(focusMarker.t)
                  ? focusMarker.t
                  : null
              }
              markerLabel="Event"
            />
            <p className="status-text">
              Playhead at {formatClock(playhead)} — the clip and marks move together.
            </p>
          </div>

          <p className="status-text trajectory-legend">
            ○ start · ■ end · dashed = interpolated across a short missed-detection
            gap · trails are smoothed to cancel per-frame box/mask jitter · colors
            by team (Red / Blue). Heatmap density uses the same analytics-safe
            points (closeups/transitions excluded) in normalized screen-space.
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
          {summary ? (
            <p className="status-text">
              {summary.identity_locked
                ? "Foot position from confirmed player masks (SAM 2 + ReID identity-locked tracks)."
                : "Foot position from raw detections — confirm player selection first for identity-locked, higher-accuracy trails."}
              {summary.mask_stance_points
                ? ` ${summary.mask_stance_points} of ${
                    (summary.mask_stance_points || 0) +
                    (summary.bbox_fallback_points || 0)
                  } points used the player's segmentation mask.`
                : ""}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export default TrajectoryViewer;
