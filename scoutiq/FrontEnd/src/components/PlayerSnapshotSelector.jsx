import { useEffect, useMemo, useState } from "react";

function selectionKey(player) {
  return player?.person_group_id ?? player?.track_id;
}

function PlayerSnapshotSelector({
  snapshot,
  imageUrl,
  masksImageUrl,
  onConfirm,
  isSaving,
}) {
  const [selectedKeys, setSelectedKeys] = useState(() => new Set());
  const [saveMessage, setSaveMessage] = useState("");
  const [viewMode, setViewMode] = useState("labels"); // labels | masks

  const frameWidth = snapshot?.frame_width || 1;
  const frameHeight = snapshot?.frame_height || 1;
  const players = snapshot?.players || [];
  const selectionUnit = snapshot?.selection_unit || "person_group_id";
  const showLabels = viewMode === "labels";
  const masksAvailable = Boolean(
    snapshot?.masks_image_filename || snapshot?.masks_image_path
  );
  const activeImageUrl =
    !showLabels && masksImageUrl && masksAvailable ? masksImageUrl : imageUrl;

  const selectedPlayers = useMemo(
    () => players.filter((player) => selectedKeys.has(selectionKey(player))),
    [players, selectedKeys]
  );

  useEffect(() => {
    setSelectedKeys(new Set());
    setSaveMessage("");
    setViewMode("labels");
  }, [snapshot?.job_id]);

  function togglePlayer(key) {
    setSaveMessage("");
    setSelectedKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  async function handleConfirm() {
    if (selectedPlayers.length < 2) {
      setSaveMessage("Select at least two players for coach insights.");
      return;
    }

    setSaveMessage("");
    const result = await onConfirm(selectedPlayers);
    if (result?.message) {
      setSaveMessage(result.message);
    }
  }

  if (!snapshot || !imageUrl) {
    return null;
  }

  return (
    <div className="card snapshot-panel">
      <h2>Choose Players to Track</h2>
      <p className="status-text">
        Click a stitched person group ({selectionUnit}) on the snapshot. Masks are
        colored by jersey team (e.g. red vs blue) so home/away is easy to see.
        Select at least two players, then Confirm for a focused preview and coach
        insights — the model is not rerun. Change the selection and Confirm again
        anytime to rebuild the focused preview for a new set of players. Frame{" "}
        {snapshot.snapshot_frame_number} at {snapshot.timestamp_seconds}s —{" "}
        {players.length} selectable people.
      </p>

      {Array.isArray(snapshot.teams) && snapshot.teams.length > 0 && (
        <div className="team-legend-row">
          {snapshot.teams.map((team) => (
            <span
              key={team.team_id}
              className="team-legend-chip"
              style={{
                borderColor: team.overlay_color_hex || "#888",
                background: `${team.overlay_color_hex || "#888"}33`,
                color: team.jersey_color_name === "white" ? "#111" : "#f4f4f4",
              }}
            >
              {team.label || `Team ${team.jersey_color_name || team.team_id}`}
              {Array.isArray(team.member_group_ids)
                ? ` (${team.member_group_ids.length})`
                : ""}
            </span>
          ))}
        </div>
      )}

      <div className="snapshot-view-toggle" role="group" aria-label="Snapshot view mode">
        <button
          type="button"
          className={`view-toggle-btn ${showLabels ? "view-toggle-btn-active" : ""}`}
          aria-pressed={showLabels}
          onClick={() => setViewMode("labels")}
        >
          Boxes + labels
        </button>
        <button
          type="button"
          className={`view-toggle-btn ${!showLabels ? "view-toggle-btn-active" : ""}`}
          aria-pressed={!showLabels}
          onClick={() => setViewMode("masks")}
          disabled={!masksAvailable || !masksImageUrl}
          title={
            masksAvailable
              ? "Show jersey-colored segmentation only"
              : "Masks-only snapshot not available for this job — re-analyze the clip"
          }
        >
          Segmentation only
        </button>
      </div>

      <div className={`snapshot-stage ${showLabels ? "" : "snapshot-stage-masks"}`}>
        <img
          src={activeImageUrl}
          alt={
            showLabels
              ? "Player snapshot with boxes and labels"
              : "Player snapshot with jersey segmentation only"
          }
          className="snapshot-image"
        />
        <div className="snapshot-overlay">
          {players.map((player) => {
            const key = selectionKey(player);
            const isSelected = selectedKeys.has(key);
            const left = (player.x1 / frameWidth) * 100;
            const top = (player.y1 / frameHeight) * 100;
            const width = ((player.x2 - player.x1) / frameWidth) * 100;
            const height = ((player.y2 - player.y1) / frameHeight) * 100;

            const teamColor = player.overlay_color_hex || "#ffffff";
            return (
              <button
                key={key}
                type="button"
                className={`player-box ${isSelected ? "player-box-selected" : ""} ${
                  showLabels ? "" : "player-box-hit-only"
                }`}
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                  borderColor: showLabels
                    ? isSelected
                      ? "#00d26a"
                      : teamColor
                    : isSelected
                      ? "#00d26a"
                      : "transparent",
                  boxShadow: showLabels
                    ? `inset 0 0 0 2px ${teamColor}55`
                    : isSelected
                      ? `inset 0 0 0 2px #00d26a88`
                      : "none",
                }}
                onClick={() => togglePlayer(key)}
                title={`${player.player_name || player.label} (${player.confidence})`}
                aria-pressed={isSelected}
              >
                {showLabels && (
                  <span
                    className="player-box-label"
                    style={{ background: teamColor, color: "#0a0a0a" }}
                  >
                    {player.player_name || player.label}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="player-chip-row">
        {players.map((player) => {
          const key = selectionKey(player);
          const isSelected = selectedKeys.has(key);
          const teamColor = player.overlay_color_hex || undefined;
          return (
            <button
              key={key}
              type="button"
              className={`player-chip ${isSelected ? "player-chip-selected" : ""}`}
              style={
                teamColor
                  ? {
                      borderColor: isSelected ? "#00d26a" : teamColor,
                      boxShadow: `inset 4px 0 0 ${teamColor}`,
                    }
                  : undefined
              }
              onClick={() => togglePlayer(key)}
            >
              {player.player_name || player.label} ({Math.round(player.confidence * 100)}%)
              {player.team_label ? ` · ${player.team_label}` : ""}
            </button>
          );
        })}
      </div>

      <div className="snapshot-actions">
        <button
          type="button"
          className="confirm-button"
          onClick={handleConfirm}
          disabled={isSaving || selectedPlayers.length < 2}
        >
          {isSaving
            ? "Tracking players…"
            : selectedPlayers.length < 2
              ? "Select at least 2 players for coach insights"
              : `Confirm ${selectedPlayers.length} player(s) to track`}
        </button>
      </div>

      {saveMessage && <p className="snapshot-save-message">{saveMessage}</p>}
    </div>
  );
}

export default PlayerSnapshotSelector;
