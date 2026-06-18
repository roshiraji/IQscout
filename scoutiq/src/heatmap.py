"""Generate player position heatmaps from YOLO detection CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def generate_player_heatmap(
    detections_csv: Path,
    output_path: Path,
    frame_width: int,
    frame_height: int,
    bins: int = 48,
) -> Path:
    """
    Build a 2D density heatmap from player center points in a labels CSV.

    Returns the written PNG path.
    """
    centers_x: list[float] = []
    centers_y: list[float] = []

    with detections_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            centers_x.append(float(row["center_x"]))
            centers_y.append(float(row["center_y"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.625), facecolor="#0f1117")
    ax.set_facecolor("#1a1d26")

    if not centers_x:
        ax.text(
            0.5,
            0.5,
            "No player detections to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#e5e7eb",
            fontsize=14,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        width = max(frame_width, 1)
        height = max(frame_height, 1)
        heatmap, _, _ = np.histogram2d(
            centers_x,
            centers_y,
            bins=bins,
            range=[[0, width], [0, height]],
        )

        ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[0, width, 0, height],
            cmap="inferno",
            aspect="auto",
            interpolation="bilinear",
        )
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("Court X (px)", color="#cbd5e1")
        ax.set_ylabel("Court Y (px)", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.set_title(
            f"Player position heatmap ({len(centers_x)} detections)",
            color="#f8fafc",
            pad=12,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
