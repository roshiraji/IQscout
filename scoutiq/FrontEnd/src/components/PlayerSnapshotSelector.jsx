import { useMemo, useState } from "react";

function PlayerSnapshotSelector({ snapshot, imageUrl, onConfirm, isSaving }) {
  const [selectedLabels, setSelectedLabels] = useState(() => new Set());
  const [saveMessage, setSaveMessage] = useState("");

  const frameWidth = snapshot?.frame_width || 1;
  const frameHeight = snapshot?.frame_height || 1;
  const players = snapshot?.players || [];

  const selectedPlayers = useMemo(
    () => players.filter((player) => selectedLabels.has(player.label)),
    [players, selectedLabels]
  );

  function togglePlayer(label) {
    setSaveMessage("");
    setSelectedLabels((current) => {
      const next = new Set(current);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  }

  async function handleConfirm() {
    if (selectedPlayers.length === 0) {
      setSaveMessage("Select at least one player box to track.");
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
        Click a player box on the snapshot (or use the list below). Frame{" "}
        {snapshot.snapshot_frame_number} at {snapshot.timestamp_seconds}s —{" "}
        {players.length} players detected.
      </p>

      <div className="snapshot-stage">
        <img
          src={imageUrl}
          alt="Player snapshot for tracking selection"
          className="snapshot-image"
        />
        <div className="snapshot-overlay">
          {players.map((player) => {
            const isSelected = selectedLabels.has(player.label);
            const left = (player.x1 / frameWidth) * 100;
            const top = (player.y1 / frameHeight) * 100;
            const width = ((player.x2 - player.x1) / frameWidth) * 100;
            const height = ((player.y2 - player.y1) / frameHeight) * 100;

            return (
              <button
                key={player.label}
                type="button"
                className={`player-box ${isSelected ? "player-box-selected" : ""}`}
                style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                onClick={() => togglePlayer(player.label)}
                title={`${player.label} (${player.confidence})`}
                aria-pressed={isSelected}
              >
                <span className="player-box-label">{player.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="player-chip-row">
        {players.map((player) => {
          const isSelected = selectedLabels.has(player.label);
          return (
            <button
              key={player.label}
              type="button"
              className={`player-chip ${isSelected ? "player-chip-selected" : ""}`}
              onClick={() => togglePlayer(player.label)}
            >
              {player.label} ({Math.round(player.confidence * 100)}%)
            </button>
          );
        })}
      </div>

      <div className="snapshot-actions">
        <button
          type="button"
          className="confirm-button"
          onClick={handleConfirm}
          disabled={isSaving || selectedPlayers.length === 0}
        >
          {isSaving ? "Tracking players…" : `Confirm ${selectedPlayers.length} player(s) to track`}
        </button>
      </div>

      {saveMessage && <p className="snapshot-save-message">{saveMessage}</p>}
    </div>
  );
}

export default PlayerSnapshotSelector;
