"""Browser-compatible MP4 encoding helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# scoutiq/src/video_encoding.py -> scoutiq/
_SCOUTIQ_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_FFMPEG_CANDIDATES = (
    _SCOUTIQ_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "imageio_ffmpeg"
    / "binaries"
    / "ffmpeg-win-x86_64-v7.1.exe",
    _SCOUTIQ_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "imageio_ffmpeg"
    / "binaries"
    / "ffmpeg.exe",
)


def _resolve_ffmpeg_exe() -> str | None:
    """Find an ffmpeg binary. Never raises."""
    env = (os.getenv("IMAGEIO_FFMPEG_EXE") or "").strip()
    if env:
        env_path = Path(env)
        if env_path.is_file():
            return str(env_path.resolve())

    for candidate in _BUNDLED_FFMPEG_CANDIDATES:
        if candidate.is_file():
            return str(candidate.resolve())

    # Scan imageio_ffmpeg binaries without calling get_ffmpeg_exe() (that can raise).
    try:
        import imageio_ffmpeg

        pkg = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
        if pkg.is_dir():
            for exe in sorted(pkg.glob("ffmpeg*.exe")) + sorted(pkg.glob("ffmpeg")):
                if exe.is_file():
                    return str(exe.resolve())
    except Exception as exc:  # noqa: BLE001
        logger.debug("imageio_ffmpeg package scan failed: %s", exc)

    return shutil.which("ffmpeg")


def encode_for_browser(video_path: Path, *, quiet: bool = False) -> Path:
    """
    Re-encode OpenCV mp4v → H.264 for browser playback.

    Soft-fails to the original file if ffmpeg is missing or encoding fails.
    Never raises for encoding problems.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        ffmpeg_exe = _resolve_ffmpeg_exe()
        if not ffmpeg_exe:
            if not quiet:
                print(
                    "Note: ffmpeg not found — keeping OpenCV mp4v output "
                    "(install imageio-ffmpeg or set IMAGEIO_FFMPEG_EXE)."
                )
            return video_path

        # Help any transitive imageio calls in this process.
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe

        temp_path = video_path.with_suffix(".browser.mp4")
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
            video_path.unlink(missing_ok=True)
            temp_path.rename(video_path)
            if not quiet:
                print(f"Encoded for browser playback (H.264): {video_path.name}")
        else:
            temp_path.unlink(missing_ok=True)
            if not quiet:
                stderr = (result.stderr or "").strip()
                hint = f" {stderr[-300:]}" if stderr else ""
                print(
                    "Warning: Browser video encoding failed; "
                    f"playback may show a black screen.{hint}"
                )
    except Exception as exc:  # noqa: BLE001 — never fail the tracking job for encode
        logger.warning("encode_for_browser soft-failed: %s", exc)
        if not quiet:
            print(f"Warning: Browser video encoding skipped ({exc}).")

    return video_path
