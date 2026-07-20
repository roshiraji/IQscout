import { useEffect, useState } from "react";
import { getHealth } from "../API/scoutiqAPI";

/**
 * Story beats that explain how ScoutIQ turns a clip into coachable insight.
 * Cycles while the local backend / dashboard finish coming online.
 */
const STORY_BEATS = [
  {
    id: "ingest",
    step: 1,
    title: "Ingest the clip",
    detail: "Drop a WNBA highlight. ScoutIQ validates duration, FPS, and resolution.",
  },
  {
    id: "detect",
    step: 2,
    title: "See every player",
    detail: "Segmentation tracks athletes across the floor — not just boxes, full silhouettes.",
  },
  {
    id: "stitch",
    step: 3,
    title: "Keep identities stable",
    detail: "Tracklets stitch into person groups so the same player stays selected over time.",
  },
  {
    id: "select",
    step: 4,
    title: "Pick the matchup",
    detail: "Click the players that matter. Confirm locks the focused preview for that pair.",
  },
  {
    id: "isolate",
    step: 5,
    title: "Clear the noise",
    detail: "Clean Isolate removes everyone else so you study spacing and footwork alone.",
  },
  {
    id: "ball",
    step: 6,
    title: "Read possession",
    detail: "Ball tracking + fusion signals mark offense, defense, and who holds the rock.",
  },
  {
    id: "coach",
    step: 7,
    title: "Coach the film",
    detail: "Matchup Outcomes score stay-close pressure, space creation, and role clarity.",
  },
];

/**
 * Full-screen boot experience: teaches the ScoutIQ workflow while services warm up.
 */
function BootScreen({ onReady }) {
  const [beatIndex, setBeatIndex] = useState(0);
  const [bootPhase, setBootPhase] = useState("warming");
  const [error, setError] = useState("");
  const [fadeKey, setFadeKey] = useState(0);

  useEffect(() => {
    const tick = window.setInterval(() => {
      setBeatIndex((i) => {
        const next = (i + 1) % STORY_BEATS.length;
        setFadeKey((k) => k + 1);
        return next;
      });
    }, 2800);
    return () => window.clearInterval(tick);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;

    async function poll() {
      attempts += 1;
      try {
        const health = await getHealth();
        if (!cancelled && health?.status === "ok") {
          setBootPhase("ready");
          window.setTimeout(() => {
            if (!cancelled) onReady?.();
          }, 650);
          return;
        }
      } catch {
        if (attempts < 8) setBootPhase("warming");
        else setBootPhase("connecting");
      }

      if (!cancelled && attempts > 90) {
        setError(
          "ScoutIQ did not finish starting. Close this window and run run_scoutiq.bat again."
        );
        setBootPhase("error");
        return;
      }
      if (!cancelled) {
        window.setTimeout(poll, 450);
      }
    }

    poll();
    return () => {
      cancelled = true;
    };
  }, [onReady]);

  const beat = STORY_BEATS[beatIndex];
  const phaseLabel =
    bootPhase === "ready"
      ? "Ready — opening your workspace"
      : bootPhase === "connecting"
        ? "Connecting to local vision engine..."
        : bootPhase === "error"
          ? "Startup interrupted"
          : "Warming local models and dashboard...";

  return (
    <div className="boot-screen" role="status" aria-live="polite">
      <div className="boot-orbit" aria-hidden="true" />
      <div className="boot-screen-inner">
        <p className="boot-eyebrow">WNBA Vision Analytics</p>
        <h1 className="boot-brand">ScoutIQ</h1>
        <p className="boot-tagline">
          Film in. Matchup clarity out — track, isolate, and coach from one clip.
        </p>

        <ol className="boot-pipeline" aria-label="How ScoutIQ works">
          {STORY_BEATS.map((item, index) => {
            const active = index === beatIndex;
            const done = index < beatIndex || bootPhase === "ready";
            return (
              <li
                key={item.id}
                className={
                  done
                    ? "boot-step boot-step-done"
                    : active
                      ? "boot-step boot-step-active"
                      : "boot-step"
                }
              >
                <span className="boot-step-num">{item.step}</span>
                <span className="boot-step-label">{item.title}</span>
              </li>
            );
          })}
        </ol>

        {error ? (
          <p className="boot-error">{error}</p>
        ) : (
          <div className="boot-story" key={fadeKey}>
            <div className="boot-story-card">
              <p className="boot-story-kicker">How it works</p>
              <h2 className="boot-story-title">{beat.title}</h2>
              <p className="boot-story-detail">{beat.detail}</p>
            </div>
            <div className="boot-progress-row">
              <div className="boot-spinner" aria-hidden="true" />
              <p className="boot-message">{phaseLabel}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BootScreen;
