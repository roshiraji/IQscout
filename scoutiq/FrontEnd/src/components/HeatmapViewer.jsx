function HeatmapViewer({ imageUrl }) {
  if (!imageUrl) {
    return (
      <div className="heatmap-panel">
        <h2>Heatmap</h2>
        <p className="status-text">
          Run Analyze on an approved clip to generate a player position heatmap.
        </p>
      </div>
    );
  }

  return (
    <div className="heatmap-panel">
      <h2>Player Position Heatmap</h2>
      <p className="status-text">Density of detected player locations across all frames.</p>
      <img src={imageUrl} alt="Player position heatmap" className="heatmap-image" />
    </div>
  );
}

export default HeatmapViewer;
