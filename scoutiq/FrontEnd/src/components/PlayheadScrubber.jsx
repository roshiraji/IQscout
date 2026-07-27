import { formatClock } from "./VideoScrubber";

/**
 * Controlled playhead for the Player Movement (trajectory) preview.
 * Scrubs clip time — trails on the canvas grow/shrink with the head.
 */
function PlayheadScrubber({
  value,
  min = 0,
  max = 0,
  onChange,
  markerSeconds = null,
  markerLabel = "Snapshot",
  playing = false,
  onTogglePlay = null,
}) {
  const safeMax = Number.isFinite(max) && max > 0 ? max : 0;
  const safeMin = Number.isFinite(min) ? min : 0;
  const current = Math.max(safeMin, Math.min(Number(value) || safeMin, safeMax || safeMin));
  const markerPct =
    safeMax > safeMin &&
    markerSeconds != null &&
    Number.isFinite(Number(markerSeconds))
      ? Math.max(
          0,
          Math.min(
            100,
            ((Number(markerSeconds) - safeMin) / (safeMax - safeMin)) * 100
          )
        )
      : null;

  return (
    <div className="video-scrubber" role="group" aria-label="Play moment scrubber">
      {typeof onTogglePlay === "function" ? (
        <button
          type="button"
          className="video-scrubber-play"
          onClick={onTogglePlay}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "Pause" : "Play"}
        </button>
      ) : (
        <span className="video-scrubber-play video-scrubber-play-label">Playhead</span>
      )}

      <div className="video-scrubber-track-wrap">
        {markerPct != null ? (
          <span
            className="video-scrubber-marker"
            style={{ left: `${markerPct}%` }}
            title={`${markerLabel} · ${formatClock(Number(markerSeconds))}`}
            aria-hidden="true"
          />
        ) : null}
        <input
          type="range"
          className="video-scrubber-range"
          min={safeMin}
          max={safeMax || safeMin}
          step={0.05}
          value={current}
          disabled={safeMax <= safeMin}
          aria-label="Scrub play moment"
          aria-valuemin={safeMin}
          aria-valuemax={safeMax}
          aria-valuenow={current}
          aria-valuetext={`${formatClock(current)} of ${formatClock(safeMax)}`}
          onChange={(event) => {
            if (typeof onChange === "function") {
              onChange(Number(event.target.value));
            }
          }}
        />
      </div>

      <span className="video-scrubber-time" aria-live="off">
        {formatClock(current)}
        <span className="video-scrubber-time-sep"> / </span>
        {formatClock(safeMax)}
      </span>
    </div>
  );
}

export default PlayheadScrubber;
