function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value))}%`;
}

function num(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function CoachInsights({ insights, narrative }) {
  if (!insights || !Array.isArray(insights.pairs) || insights.pairs.length === 0) {
    return null;
  }

  const glance = insights.glance || {};
  const primary = insights.pairs[0] || {};
  const relationship = primary.relationship || glance.relationship || "unknown";
  const isOpponent = relationship === "opponent";
  const isSameTeam = relationship === "same_team";
  const pairLabel =
    (primary.labels || glance.players?.map((p) => p.label) || []).join(" vs ") ||
    "Selected pair";

  return (
    <div className="card coach-insights-panel">
      <h2>Coach Insights</h2>
      <p className="status-text">
        Clip proxies from locked person groups (normalized to frame size — not court
        feet or box-score substitutes).{" "}
        {isOpponent
          ? "Opposite teams → pressure / matchup view."
          : isSameTeam
            ? "Same team → off-ball action view."
            : "Team relationship unknown for this pair."}
      </p>

      <p className="coach-pair-label">{pairLabel}</p>

      <div className="metadata-grid coach-glance-grid">
        <div>
          <strong>Both visible</strong>
          <span>{pct(glance.both_visible_pct)}</span>
        </div>
        <div>
          <strong>Mean gap</strong>
          <span>{num(glance.mean_gap_norm)} diag</span>
        </div>
        <div>
          <strong>{isOpponent ? "Closeouts" : "Close bursts"}</strong>
          <span>{glance.closeout_count ?? "—"}</span>
        </div>
        {(glance.players || []).map((player) => (
          <div key={player.person_group_id}>
            <strong>{player.label || `Person #${player.person_group_id}`}</strong>
            <span>
              effort {pct((player.high_effort_fraction || 0) * 100)} · cov{" "}
              {pct((player.coverage || 0) * 100)}
            </span>
          </div>
        ))}
      </div>

      {(narrative || insights.coach_narrative) && (
        <p className="coach-narrative">{narrative || insights.coach_narrative}</p>
      )}

      <details className="coach-details">
        <summary>Detail stats</summary>
        <div className="coach-detail-block">
          <h3>Pair</h3>
          <ul>
            <li>
              Near-band frames:{" "}
              {pct(
                (primary.pairwise_separation?.pct_frames_within_near || 0) * 100
              )}
            </li>
            <li>
              Close-band frames:{" "}
              {pct(
                (primary.pairwise_separation?.pct_frames_within_close || 0) * 100
              )}
            </li>
            <li>
              Max close rate:{" "}
              {num(primary.closeout_bursts?.max_close_rate_norm)} diag/s
            </li>
            {isOpponent && (
              <>
                <li>
                  Chase leader: Person #
                  {primary.chase_role?.leader_person_group_id ?? "—"}
                </li>
                <li>
                  Sticky fraction:{" "}
                  {pct((primary.sticky_matchup?.sticky_fraction || 0) * 100)}
                </li>
              </>
            )}
            {isSameTeam && (
              <>
                <li>
                  Screen shape:{" "}
                  {(primary.screen_shape_proxy?.soft_label || "—").replaceAll(
                    "_",
                    " "
                  )}
                </li>
                <li>
                  Cut-timing overlap:{" "}
                  {pct(
                    (primary.cut_timing_alignment?.high_effort_overlap || 0) * 100
                  )}
                </li>
              </>
            )}
          </ul>
          <h3>Per player</h3>
          <ul>
            {(insights.players || []).map((player) => (
              <li key={player.person_group_id}>
                {player.label}: path{" "}
                {num(player.movement_intensity?.path_length_norm, 2)} · p90 speed{" "}
                {num(player.movement_intensity?.p90_speed_norm)} · longest hold{" "}
                {num(player.coverage_trust?.longest_continuous_seconds, 1)}s
              </li>
            ))}
          </ul>
        </div>
      </details>
    </div>
  );
}

export default CoachInsights;
