import { useEffect, useRef, useState } from "react";

function formatClock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Playhead scrubber bound to a <video> element.
 * Dragging seeks the clip; the thumb tracks playback in real time.
 */
function VideoScrubber({
  videoRef,
  markerSeconds = null,
  markerLabel = "Snapshot",
  mediaKey = null,
}) {
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const draggingRef = useRef(false);

  useEffect(() => {
    const video = videoRef?.current;
    if (!video) return undefined;

    setCurrent(0);
    setDuration(0);
    setPlaying(false);

    const syncMeta = () => {
      const dur = Number(video.duration);
      if (Number.isFinite(dur) && dur > 0) setDuration(dur);
      if (!draggingRef.current) setCurrent(Number(video.currentTime) || 0);
      setPlaying(!video.paused && !video.ended);
    };

    const onTime = () => {
      if (draggingRef.current) return;
      setCurrent(Number(video.currentTime) || 0);
    };

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => setPlaying(false);

    syncMeta();
    video.addEventListener("loadedmetadata", syncMeta);
    video.addEventListener("durationchange", syncMeta);
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", onTime);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);

    return () => {
      video.removeEventListener("loadedmetadata", syncMeta);
      video.removeEventListener("durationchange", syncMeta);
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("seeked", onTime);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onEnded);
    };
  }, [videoRef, mediaKey]);

  function seekTo(next) {
    const video = videoRef?.current;
    if (!video) return;
    const dur = Number.isFinite(video.duration) ? video.duration : duration;
    const clamped = Math.max(0, Math.min(Number(next) || 0, dur || 0));
    try {
      video.currentTime = clamped;
    } catch {
      // Ignore seeks before metadata is ready.
    }
    setCurrent(clamped);
  }

  async function togglePlay() {
    const video = videoRef?.current;
    if (!video) return;
    try {
      if (video.paused || video.ended) {
        await video.play();
      } else {
        video.pause();
      }
    } catch {
      // Autoplay / gesture restrictions — ignore.
    }
  }

  const max = duration > 0 ? duration : 0;
  const markerPct =
    max > 0 &&
    markerSeconds != null &&
    Number.isFinite(Number(markerSeconds))
      ? Math.max(0, Math.min(100, (Number(markerSeconds) / max) * 100))
      : null;

  return (
    <div className="video-scrubber" role="group" aria-label="Play scrubber">
      <button
        type="button"
        className="video-scrubber-play"
        onClick={togglePlay}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? "Pause" : "Play"}
      </button>

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
          min={0}
          max={max || 0}
          step={0.01}
          value={Math.min(current, max || 0)}
          disabled={!max}
          aria-label="Scrub playhead"
          aria-valuemin={0}
          aria-valuemax={max || 0}
          aria-valuenow={current}
          aria-valuetext={`${formatClock(current)} of ${formatClock(max)}`}
          onPointerDown={() => {
            draggingRef.current = true;
          }}
          onPointerUp={(event) => {
            draggingRef.current = false;
            seekTo(event.currentTarget.value);
          }}
          onChange={(event) => {
            const value = Number(event.target.value);
            setCurrent(value);
            if (draggingRef.current) {
              seekTo(value);
            }
          }}
        />
      </div>

      <span className="video-scrubber-time" aria-live="off">
        {formatClock(current)}
        <span className="video-scrubber-time-sep"> / </span>
        {formatClock(max)}
      </span>
    </div>
  );
}

export default VideoScrubber;
export { formatClock };
