"""
ScoutIQ Video Intake Filter — Streamlit upload UI.

Drag-and-drop MP4 upload, validation gate, and intake report export.
Does NOT run YOLO, tracking, or process_video.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow imports from src/ when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_intake_filter import save_intake_report, validate_video  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "data" / "input_videos"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "intake_reports"


def _ensure_directories() -> None:
    """Create upload and report folders if they do not exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_uploaded_file(uploaded_file) -> Path:
    """Write the uploaded bytes to data/input_videos/."""
    destination = INPUT_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _render_metadata(metadata: dict) -> None:
    """Display extracted video metadata in a clean layout."""
    st.subheader("Video Metadata")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filename", metadata.get("filename", "—"))
    col2.metric("File Size (MB)", metadata.get("file_size_mb", 0))
    col3.metric("Duration (s)", metadata.get("duration_seconds", 0))

    col4, col5, col6 = st.columns(3)
    col4.metric("FPS", metadata.get("fps", 0))
    col5.metric("Frame Count", metadata.get("frame_count", 0))
    col6.metric("Resolution", f"{metadata.get('width', 0)} x {metadata.get('height', 0)}")

    col7, col8, col9 = st.columns(3)
    col7.metric("Aspect Ratio", metadata.get("aspect_ratio", "—"))
    col8.metric("Resolution Label", metadata.get("resolution_label", "—"))
    col9.metric("MIME Type", metadata.get("mime_type") or "unknown")


def _render_decision(result: dict, report_path: Path) -> None:
    """Show approved / rejected status, warnings, errors, and next-step state."""
    st.divider()
    st.subheader("Intake Decision")

    if result["usable"]:
        st.success(f"Approved — {result['message']}")
    else:
        st.error(f"Rejected — {result['message']}")

    if result["warnings"]:
        st.warning("Warnings")
        for warning in result["warnings"]:
            st.markdown(f"- {warning}")

    if result["errors"]:
        st.error("Errors")
        for error in result["errors"]:
            st.markdown(f"- {error}")

    st.caption(f"Intake report saved to: `{report_path}`")

    # Next-step gate — only enabled when the video passes validation
    st.divider()
    if result["usable"]:
        st.info("Ready for Processing — this video can move forward to the ScoutIQ vision pipeline.")
        st.button("Proceed to Vision Pipeline", type="primary", disabled=False)
    else:
        st.info("Not Ready for Processing — fix the issues above before running any AI models.")
        st.button("Proceed to Vision Pipeline", type="primary", disabled=True)


def main() -> None:
    st.set_page_config(page_title="ScoutIQ Video Intake Filter", page_icon="🏀", layout="wide")

    _ensure_directories()

    st.title("ScoutIQ Video Intake Filter")
    st.markdown(
        "Upload a WNBA/NBA basketball **MP4** clip. "
        "The file will be checked for format, size, duration, resolution, and readability "
        "**before** any computer vision models (YOLO, tracking, etc.) are run."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your MP4 video here",
        type=["mp4"],
        help="Only .mp4 files are accepted in this MVP intake stage.",
    )

    if uploaded_file is None:
        st.info("Upload an MP4 video to begin the intake check.")
        return

    # Run validation inside try/except so a bad upload never crashes the UI
    try:
        saved_path = _save_uploaded_file(uploaded_file)
        st.caption(f"Saved upload to: `{saved_path}`")

        with st.spinner("Running intake validation..."):
            result = validate_video(saved_path)
            report_path = save_intake_report(
                validation_result=result,
                original_filename=uploaded_file.name,
                reports_dir=REPORTS_DIR,
            )

        _render_metadata(result["metadata"])
        _render_decision(result, report_path)

    except Exception as exc:
        st.error("An unexpected error occurred while validating the upload.")
        st.exception(exc)


if __name__ == "__main__":
    main()
