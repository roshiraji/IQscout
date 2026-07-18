function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value))}%`;
}

function num(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function score100(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Math.round(Number(value) * 100);
}

function scoreTone(score) {
  if (score == null) return "neutral";
  if (score >= 70) return "strong";
  if (score >= 45) return "mid";
  return "low";
}

function MetricTile({ label, value, hint }) {
  if (value == null || value === "—") return null;
  return (
    <div className="scorecard-tile">
      <strong>{label}</strong>
      <span className="scorecard-tile-value">{value}</span>
      {hint ? <em className="scorecard-tile-hint">{hint}</em> : null}
    </div>
  );
}

function ComponentBar({ label, value }) {
  if (value == null || Number.isNaN(Number(value))) return null;
  const pctValue = Math.round(Number(value) * 100);
  return (
    <div className="scorecard-bar-row">
      <div className="scorecard-bar-label">
        <span>{label}</span>
        <span>{pctValue}%</span>
      </div>
      <div className="scorecard-bar-track">
        <div
          className="scorecard-bar-fill"
          style={{ width: `${Math.max(0, Math.min(100, pctValue))}%` }}
        />
      </div>
    </div>
  );
}

function OutcomeHero({ label, score, subtitle, tone }) {
  const display =
    score == null || Number.isNaN(Number(score)) ? 0 : Number(score);
  return (
    <div className={`scorecard-hero scorecard-hero-${tone || scoreTone(display)}`}>
      <span className="scorecard-hero-label">{label}</span>
      <strong className="scorecard-hero-value">{display}</strong>
      <span className="scorecard-hero-unit">/ 100</span>
      {subtitle ? <span className="scorecard-hero-sub">{subtitle}</span> : null}
    </div>
  );
}

function GapSparkline({ series, personIds }) {
  if (!Array.isArray(series) || series.length === 0 || !Array.isArray(personIds)) {
    return null;
  }
  const [a, b] = [...personIds].sort((x, y) => x - y);
  const pair =
    series.find((p) => {
      const ids = [p.person_group_a, p.person_group_b].sort((x, y) => x - y);
      return ids[0] === a && ids[1] === b;
    }) || series[0];
  const frames = (pair?.frames || [])
    .map((f) => Number(f.gap_norm))
    .filter((v) => !Number.isNaN(v));
  if (frames.length < 2) return null;

  const maxGap = Math.max(...frames, 0.01);
  const w = 280;
  const h = 48;
  const points = frames
    .map((gap, i) => {
      const x = (i / (frames.length - 1)) * w;
      const y = h - (gap / maxGap) * (h - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="scorecard-sparkline">
      <div className="scorecard-sparkline-head">
        <strong>Separation over time</strong>
        <span>lower = tighter defense</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="scorecard-sparkline-svg" aria-hidden>
        <polyline fill="none" stroke="currentColor" strokeWidth="2" points={points} />
      </svg>
    </div>
  );
}

function CoachInsights({ insights, narrative }) {
  if (!insights || !Array.isArray(insights.pairs) || insights.pairs.length === 0) {
    return null;
  }

  const glance = insights.glance || {};
  const primary = insights.pairs[0] || {};
  const scorecard = primary.matchup_scorecard || {};
  const components = scorecard.components || {};
  const sep = primary.pairwise_separation || {};
  const recovery = primary.recovery || {};
  const relationship = primary.relationship || glance.relationship || "unknown";
  const isOpponent = relationship === "opponent";
  const isSameTeam = relationship === "same_team";
  const pairLabel =
    (primary.labels || glance.players?.map((p) => p.label) || []).join(" vs ") ||
    "Selected pair";

  const stayClose = score100(
    scorecard.stay_close_score ??
      glance.stay_close_score ??
      scorecard.pressure_index ??
      glance.pressure_index ??
      0
  );
  const createSpace = score100(
    scorecard.create_space_score ??
      glance.create_space_score ??
      scorecard.secondary_score ??
      0
  );
  const offBall = score100(
    scorecard.off_ball_score ??
      glance.off_ball_score ??
      scorecard.offense_index ??
      glance.offense_index ??
      scorecard.primary_score ??
      glance.primary_score ??
      0
  );

  const defenderLabel =
    scorecard.defender_label || glance.defender_label || "Stay Close";
  const attackerLabel =
    scorecard.attacker_label || glance.attacker_label || "Create Space";

  // Same jersey color / same team → off-ball score. Otherwise dual matchup scores.
  const showMatchup = !isSameTeam;

  const attachedPct =
    glance.time_attached_pct ??
    (sep.pct_frames_within_near != null ? sep.pct_frames_within_near * 100 : null);
  const separatedPct =
    glance.time_separated_pct ??
    (attachedPct != null ? Math.max(0, 100 - attachedPct) : null);

  const recoveryValue =
    recovery.mean_lag_seconds != null || glance.recovery_mean_seconds != null
      ? `${num(glance.recovery_mean_seconds ?? recovery.mean_lag_seconds, 2)}s`
      : null;

  return (
    <div className="card coach-insights-panel scorecard-panel">
      <div className="scorecard-header">
        <div>
          <h2>Matchup Outcomes</h2>
          <p className="status-text scorecard-subhead">
            {isOpponent
              ? "Who won the gap on this clip: stay attached, or create space and get away."
              : isSameTeam
                ? "Off-ball action on this clip: spacing, pick shape, and cut timing."
                : "Select opposite-team players for defense/offense outcome scores."}
          </p>
          <p className="coach-pair-label">{pairLabel}</p>
        </div>

        {showMatchup ? (
          <div className="scorecard-hero-pair">
            <OutcomeHero
              label={defenderLabel}
              score={stayClose}
              subtitle="Stay Close"
              tone={scoreTone(stayClose)}
            />
            <OutcomeHero
              label={attackerLabel}
              score={createSpace}
              subtitle="Create Space"
              tone={scoreTone(createSpace)}
            />
          </div>
        ) : (
          <OutcomeHero
            label="Off-Ball Action"
            score={offBall}
            subtitle="Pick / cut / spacing"
            tone={scoreTone(offBall)}
          />
        )}
      </div>

      <div className="scorecard-tile-grid">
        <MetricTile
          label="Time attached"
          value={pct(attachedPct)}
          hint="defender stayed close"
        />
        <MetricTile
          label="Time separated"
          value={pct(separatedPct)}
          hint="attacker got away"
        />
        <MetricTile
          label="Closeouts"
          value={glance.closeout_count ?? primary.closeout_bursts?.count}
          hint="defense closed the gap"
        />
        <MetricTile
          label="Get-away bursts"
          value={
            glance.separation_burst_count ?? primary.separation_bursts?.count
          }
          hint="offense opened the gap"
        />
        <MetricTile
          label="Tightest gap"
          value={
            (glance.p10_gap_norm ?? sep.p10_gap_norm) != null
              ? `${num(glance.p10_gap_norm ?? sep.p10_gap_norm)} diag`
              : null
          }
          hint="closest stretches"
        />
        <MetricTile
          label="Recovery"
          value={recoveryValue}
          hint="time to re-attach"
        />
      </div>

      <div className="scorecard-body">
        <div className="scorecard-components">
          <h3>{isOpponent ? "Why these scores" : "Action breakdown"}</h3>
          {isOpponent ? (
            <>
              <ComponentBar label="Time attached" value={components.time_attached} />
              <ComponentBar
                label="Very tight stretches"
                value={components.time_very_tight}
              />
              <ComponentBar label="Closeout activity" value={components.closeout_activity} />
              <ComponentBar label="Time separated" value={components.time_separated} />
              <ComponentBar
                label="Get-away bursts"
                value={components.separation_bursts}
              />
              <ComponentBar
                label="Max separation stretch"
                value={components.max_gap_stretch}
              />
            </>
          ) : isSameTeam ? (
            <>
              <ComponentBar label="Screen shape" value={components.screen_score} />
              <ComponentBar label="Cut timing" value={components.cut_overlap} />
              <ComponentBar
                label="Spacing quality"
                value={components.spacing_quality}
              />
            </>
          ) : (
            <p className="status-text">
              Opposite-team pairs unlock Stay Close vs Create Space scores.
            </p>
          )}
        </div>

        <GapSparkline
          series={insights.pair_distance_series}
          personIds={primary.person_group_ids}
        />
      </div>

      {(glance.players || []).length > 0 && (
        <div className="scorecard-players">
          {(glance.players || []).map((player) => (
            <div key={player.person_group_id} className="scorecard-player-chip">
              <strong>
                {player.label || `Person #${player.person_group_id}`}
              </strong>
              <span>
                work rate {pct((player.high_effort_fraction || 0) * 100)}
              </span>
            </div>
          ))}
        </div>
      )}

      {(narrative ||
        insights.coach_narrative ||
        scorecard.outcome_summary ||
        glance.outcome_summary) && (
        <p className="coach-narrative">
          {narrative ||
            insights.coach_narrative ||
            scorecard.outcome_summary ||
            glance.outcome_summary}
        </p>
      )}

      <p className="scorecard-disclaimer">
        {scorecard.disclaimer ||
          "Create Space means got away on distance — not points scored. Shot make/miss needs a ball model."}
      </p>

      <details className="coach-details">
        <summary>Raw detail stats</summary>
        <div className="coach-detail-block">
          <ul>
            <li>
              Mean / max gap: {num(sep.mean_gap_norm)} / {num(sep.max_gap_norm)}{" "}
              diag
            </li>
            <li>
              Max close rate:{" "}
              {num(primary.closeout_bursts?.max_close_rate_norm)} diag/s
            </li>
            <li>
              Max open rate:{" "}
              {num(primary.separation_bursts?.max_open_rate_norm)} diag/s
            </li>
            {isSameTeam && (
              <li>
                Screen shape:{" "}
                {(primary.screen_shape_proxy?.soft_label || "—").replaceAll(
                  "_",
                  " "
                )}
              </li>
            )}
          </ul>
        </div>
      </details>
    </div>
  );
}

export default CoachInsights;
