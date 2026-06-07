"""
ScoutIQ Video Intake Filter — Streamlit upload UI.

Drag-and-drop MP4 upload, validation gate, and intake report export.
Does NOT run YOLO, tracking, or process_video.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

# Allow imports from src/ when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from player_labeling import describe_device, resolve_device, run_player_labeling  # noqa: E402
from video_intake_filter import save_intake_report, validate_video  # noqa: E402
from yolo_config import (  # noqa: E402
    AVAILABLE_IMGSZ,
    AVAILABLE_YOLO_FAMILIES,
    DEFAULT_YOLO_FAMILY,
    DEFAULT_YOLO_MODEL,
    QUALITY_MODE,
    get_default_model_for_family,
    get_model_option,
    get_models_for_family,
    model_display_name,
    model_select_label,
    model_summary,
)

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


def _render_labeling_results(labeling_result: dict) -> None:
    """Show YOLO player labeling outputs after the vision pipeline runs."""
    st.divider()
    st.subheader("Vision Pipeline Results")

    st.success("Player labeling complete.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames processed", labeling_result["frames_processed"])
    col2.metric("Primary detections", labeling_result["total_detections"])
    col3.metric("Review candidates", labeling_result.get("review_candidates", 0))

    mode_label = "Quality" if labeling_result.get("quality_mode", True) else "Speed"
    st.markdown(f"**Model used:** {labeling_result.get('model_label', 'unknown')}")
    st.markdown(f"**Device used:** {labeling_result.get('device_label', 'unknown')}")
    st.markdown(
        f"**Inference mode:** {mode_label} | imgsz={labeling_result.get('imgsz', '—')} | "
        f"primary conf={labeling_result.get('primary_conf', '—')}"
    )

    timing_col1, timing_col2, timing_col3 = st.columns(3)
    timing_col1.metric("Avg inference (ms/frame)", labeling_result.get("avg_inference_ms_per_frame", 0))
    timing_col2.metric("Pipeline FPS", labeling_result.get("pipeline_fps", 0))
    timing_col3.metric("Total runtime (s)", labeling_result.get("total_runtime_seconds", 0))

    st.markdown(f"**Annotated video:** `{labeling_result['annotated_video']}`")
    st.markdown(f"**Player labels CSV:** `{labeling_result['detections_csv']}`")
    st.markdown(f"**ByteTrack CSV:** `{labeling_result.get('bytetrack_csv', '—')}`")
    st.markdown(f"**Frame statistics CSV:** `{labeling_result.get('frame_statistics_csv', '—')}`")
    st.markdown(
        f"**Low-confidence CSV:** `{labeling_result.get('low_confidence_csv', '—')}`"
    )

    annotated_path = Path(str(labeling_result["annotated_video"]))
    csv_path = Path(str(labeling_result["detections_csv"]))
    bytetrack_path = Path(str(labeling_result.get("bytetrack_csv", "")))
    frame_stats_path = Path(str(labeling_result.get("frame_statistics_csv", "")))
    low_conf_path = Path(str(labeling_result.get("low_confidence_csv", "")))

    st.markdown("### Watch annotated clip")
    if annotated_path.is_file() and annotated_path.stat().st_size > 0:
        video_bytes = annotated_path.read_bytes()
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            label="Download annotated MP4",
            data=video_bytes,
            file_name=annotated_path.name,
            mime="video/mp4",
        )
    else:
        st.error("Annotated video was not found. Try running the pipeline again.")

    st.markdown("### Download detection outputs")
    if csv_path.is_file():
        st.download_button(
            label="Download player labels CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            key="download_player_labels",
        )
    if bytetrack_path.is_file():
        st.download_button(
            label="Download ByteTrack detections CSV",
            data=bytetrack_path.read_bytes(),
            file_name=bytetrack_path.name,
            mime="text/csv",
            key="download_bytetrack",
        )
    if frame_stats_path.is_file():
        st.download_button(
            label="Download frame statistics CSV",
            data=frame_stats_path.read_bytes(),
            file_name=frame_stats_path.name,
            mime="text/csv",
            key="download_frame_stats",
        )
    if low_conf_path.is_file():
        st.download_button(
            label="Download low-confidence candidates CSV",
            data=low_conf_path.read_bytes(),
            file_name=low_conf_path.name,
            mime="text/csv",
            key="download_low_conf",
        )


def _render_decision(
    result: dict,
    report_path: Path,
    saved_path: Path,
    selected_model: str,
    quality_mode: bool,
    imgsz: int,
) -> None:
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

    # Next-step gate — runs YOLO player labeling when the user proceeds
    st.divider()
    if result["usable"]:
        st.info("Ready for Processing — this video can move forward to the ScoutIQ vision pipeline.")
        device_label = describe_device(resolve_device())
        model_label = model_display_name(selected_model)
        mode_label = "Quality" if quality_mode else "Speed"
        st.caption(
            f"YOLO will run on: {device_label} | Model: {model_label} | "
            f"Mode: {mode_label} | imgsz: {imgsz}"
        )
        if st.button("Proceed to Vision Pipeline", type="primary"):
            with st.spinner(f"Running {model_label} ({mode_label}, imgsz={imgsz}) on {device_label}..."):
                try:
                    labeling_result = run_player_labeling(
                        saved_path,
                        model_name=selected_model,
                        quality_mode=quality_mode,
                        imgsz=imgsz if quality_mode else None,
                    )
                    st.session_state["labeling_result"] = labeling_result
                    st.rerun()
                except Exception as exc:
                    st.error(f"Player labeling failed: {exc}")
                    st.exception(exc)
    else:
        st.info("Not Ready for Processing — fix the issues above before running any AI models.")
        st.button("Proceed to Vision Pipeline", type="primary", disabled=True)


def _render_device_banner() -> None:
    """Warn when Streamlit is running with CPU-only PyTorch (wrong venv)."""
    device_label = describe_device(resolve_device())
    if torch.cuda.is_available():
        st.caption(f"Compute device: {device_label}")
        return

    st.warning(
        "YOLO is running on **CPU** because this Streamlit session is not using "
        "`scoutiq/.venv` with CUDA PyTorch. Stop Streamlit and restart with:\n\n"
        "`cd scoutiq` then `.\\run_streamlit.ps1`"
    )


def main() -> None:
    st.set_page_config(page_title="ScoutIQ Video Intake Filter", page_icon="🏀", layout="wide")

    _ensure_directories()

    # YOLO model picker and quality settings — shared across the vision pipeline
    with st.sidebar:
        st.header("Vision Model")
        selected_family = st.selectbox(
            "YOLO family",
            AVAILABLE_YOLO_FAMILIES,
            index=AVAILABLE_YOLO_FAMILIES.index(DEFAULT_YOLO_FAMILY),
            help="YOLO26 is the default edge-first model. YOLO11 is the prior Ultralytics generation.",
        )
        family_models = get_models_for_family(selected_family)
        family_weights = [option.weights for option in family_models]
        default_family_model = get_default_model_for_family(selected_family)
        if st.session_state.get("model_family") != selected_family:
            st.session_state["model_family"] = selected_family
            st.session_state["model_weights"] = default_family_model
        saved_weights = st.session_state.get("model_weights", default_family_model)
        if saved_weights not in family_weights:
            saved_weights = default_family_model
        selected_model = st.selectbox(
            f"{selected_family} model size",
            family_weights,
            index=family_weights.index(saved_weights),
            format_func=model_select_label,
            help="Nano → Extra Large. Larger models are slower but more accurate.",
        )
        st.session_state["model_weights"] = selected_model
        model_option = get_model_option(selected_model)
        st.caption(model_display_name(selected_model))
        st.caption(model_summary(selected_model))
        speed_col, accuracy_col = st.columns(2)
        speed_col.metric("Speed", model_option.speed_tier)
        accuracy_col.metric("Accuracy", model_option.accuracy_tier)

        st.header("Detection Quality")
        quality_mode = st.checkbox(
            "Quality mode (recommended)",
            value=QUALITY_MODE,
            help="Higher resolution, lower confidence threshold, and test-time augmentation. "
            "Slower but better for tracking and jersey-ID prep.",
        )
        imgsz_options = [1280, 1920]
        selected_imgsz = st.selectbox(
            "Inference resolution (imgsz)",
            imgsz_options,
            index=0,
            disabled=not quality_mode,
            help="1280 is the default for RTX 3070 Ti. Try 1920 if you have VRAM headroom.",
        )
        if not quality_mode:
            st.caption("Speed mode uses imgsz=640 and conf=0.35.")

    st.title("ScoutIQ Video Intake Filter")
    _render_device_banner()
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

    # Clear prior labeling results when a different file is uploaded
    if st.session_state.get("last_upload") != uploaded_file.name:
        st.session_state["last_upload"] = uploaded_file.name
        st.session_state.pop("labeling_result", None)

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
        _render_decision(
            result,
            report_path,
            saved_path,
            selected_model,
            quality_mode=quality_mode,
            imgsz=selected_imgsz,
        )

        if st.session_state.get("labeling_result"):
            _render_labeling_results(st.session_state["labeling_result"])

    except Exception as exc:
        st.error("An unexpected error occurred while validating the upload.")
        st.exception(exc)


if __name__ == "__main__":
    main()
