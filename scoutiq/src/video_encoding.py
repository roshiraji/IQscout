"""Browser-compatible MP4 encoding helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def encode_for_browser(video_path: Path, *, quiet: bool = False) -> Path:
    """
    Re-encode an OpenCV-written MP4 (typically mp4v) to H.264 for browser playback.

    Uses libx264 with yuv420p and faststart so Chrome, Edge, and Safari can play
    the file in <video> tags and Streamlit st.video().
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import imageio_ffmpeg
    except ImportError:
        if not quiet:
            print("Note: install imageio-ffmpeg for in-browser video playback.")
        return video_path

    temp_path = video_path.with_suffix(".browser.mp4")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(temp_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and temp_path.is_file() and temp_path.stat().st_size > 0:
        video_path.unlink()
        temp_path.rename(video_path)
        if not quiet:
            print(f"Encoded for browser playback (H.264): {video_path.name}")
    else:
        temp_path.unlink(missing_ok=True)
        if not quiet:
            stderr = (result.stderr or "").strip()
            hint = f" {stderr[-300:]}" if stderr else ""
            print(f"Warning: Browser video encoding failed; playback may show a black screen.{hint}")

    return video_path
