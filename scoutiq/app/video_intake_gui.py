"""
ScoutIQ Video Intake Filter — Tkinter desktop GUI.

Lets the user pick an MP4 through the file explorer and validates it
using src/video_intake_filter.py. Does NOT run YOLO or tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Tkinter is bundled with most Python installs but missing/broken on some Windows setups.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    print(
        "ERROR: Tkinter is not available in this Python environment.\n"
        "On Windows, run the GUI with the project launcher instead:\n"
        "  .\\.venv-gui\\Scripts\\python.exe app\\video_intake_gui.py\n"
        "Or from scoutiq/:  .\\run_gui.bat"
    )
    raise SystemExit(1) from exc

# Resolve project paths so imports work from the scoutiq/ project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "intake_reports"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_intake_filter import save_intake_report, validate_video  # noqa: E402

# Simple color palette for status text
COLOR_APPROVED = "#1B7F3B"
COLOR_REJECTED = "#C62828"
COLOR_WARNING = "#E65100"
COLOR_MUTED = "#555555"


class VideoIntakeGUI:
    """Small desktop window for manual video intake validation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ScoutIQ Video Intake Filter")
        self.root.geometry("640x520")
        self.root.minsize(520, 420)

        self.selected_path: Path | None = None
        self._build_layout()

    def _build_layout(self) -> None:
        """Create labels, buttons, and result areas."""
        padding = {"padx": 12, "pady": 6}

        # 1. Instructions
        ttk.Label(
            self.root,
            text=(
                "Select a WNBA/NBA basketball MP4 clip. "
                "The file will be checked before any computer vision models run."
            ),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", **padding)

        # 2. File picker button
        ttk.Button(
            self.root,
            text="Select Video File",
            command=self._on_select_file,
        ).pack(anchor="w", **padding)

        # 3. Selected path display
        self.path_label = ttk.Label(
            self.root,
            text="No file selected.",
            wraplength=600,
            foreground=COLOR_MUTED,
        )
        self.path_label.pack(anchor="w", **padding)

        # 4. Approval / rejection status
        self.status_label = ttk.Label(
            self.root,
            text="Status: waiting for a file...",
            font=("Segoe UI", 12, "bold"),
        )
        self.status_label.pack(anchor="w", **padding)

        self.message_label = ttk.Label(self.root, text="", wraplength=600)
        self.message_label.pack(anchor="w", padx=12)

        # 5. Metadata and detail area
        ttk.Label(self.root, text="Validation Details", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        self.details_text = tk.Text(self.root, height=16, wrap="word", font=("Consolas", 10))
        self.details_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.details_text.configure(state="disabled")

        # Color tags for warnings and errors inside the text widget
        self.details_text.tag_configure("warning", foreground=COLOR_WARNING)
        self.details_text.tag_configure("error", foreground=COLOR_REJECTED)
        self.details_text.tag_configure("heading", font=("Consolas", 10, "bold"))

        # 6. Report path footer
        self.report_label = ttk.Label(
            self.root,
            text="",
            wraplength=600,
            foreground=COLOR_MUTED,
        )
        self.report_label.pack(anchor="w", padx=12, pady=(0, 12))

    def _on_select_file(self) -> None:
        """Open the native file picker and validate the chosen video."""
        file_path = filedialog.askopenfilename(
            title="Select MP4 Video",
            initialdir=str(PROJECT_ROOT / "data" / "input_videos"),
            filetypes=[
                ("MP4 videos", "*.mp4"),
                ("Video files", "*.mp4;*.mov;*.avi;*.mkv"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self.selected_path = Path(file_path)
        self.path_label.configure(text=f"Selected: {self.selected_path.resolve()}")

        try:
            result = validate_video(self.selected_path)
            report_path = save_intake_report(
                validation_result=result,
                original_filename=self.selected_path.name,
                reports_dir=REPORTS_DIR,
                selected_file_path=self.selected_path,
            )
            self._render_result(result, report_path)
        except Exception as exc:
            # Never crash the GUI on a bad file or unexpected error
            self.status_label.configure(text="Status: ERROR", foreground=COLOR_REJECTED)
            self.message_label.configure(text=str(exc))
            messagebox.showerror("Validation Error", f"An error occurred:\n{exc}")

    def _render_result(self, result: dict, report_path: Path) -> None:
        """Show approval status, metadata, warnings, and errors in the window."""
        metadata = result.get("metadata", {})

        if result.get("usable"):
            self.status_label.configure(text="APPROVED", foreground=COLOR_APPROVED)
        else:
            self.status_label.configure(text="REJECTED", foreground=COLOR_REJECTED)

        self.message_label.configure(text=result.get("message", ""))

        # Build readable detail text
        lines = [
            "Metadata",
            f"  filename:         {metadata.get('filename', '—')}",
            f"  file_size_mb:     {metadata.get('file_size_mb', 0)}",
            f"  duration_seconds: {metadata.get('duration_seconds', 0)}",
            f"  fps:              {metadata.get('fps', 0)}",
            f"  frame_count:      {metadata.get('frame_count', 0)}",
            f"  width:            {metadata.get('width', 0)}",
            f"  height:           {metadata.get('height', 0)}",
            f"  aspect_ratio:     {metadata.get('aspect_ratio', '—')}",
            f"  resolution_label: {metadata.get('resolution_label', '—')}",
            "",
        ]

        warnings = result.get("warnings", [])
        errors = result.get("errors", [])

        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", tk.END)

        self.details_text.insert(tk.END, "\n".join(lines))

        if warnings:
            self.details_text.insert(tk.END, "Warnings\n", "heading")
            for warning in warnings:
                self.details_text.insert(tk.END, f"  - {warning}\n", "warning")

        if errors:
            self.details_text.insert(tk.END, "Errors\n", "heading")
            for error in errors:
                self.details_text.insert(tk.END, f"  - {error}\n", "error")

        if not warnings and not errors:
            self.details_text.insert(tk.END, "No warnings or errors.\n")

        self.details_text.configure(state="disabled")
        self.report_label.configure(text=f"Intake report saved to: {report_path.resolve()}")


def main() -> None:
    """Launch the ScoutIQ video intake desktop GUI."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            "ERROR: Tkinter/Tcl is broken in this Python install.\n"
            "Use the ScoutIQ GUI environment instead:\n"
            "  scoutiq\\.venv-gui\\Scripts\\python.exe app\\video_intake_gui.py\n"
            f"Details: {exc}"
        )
        raise SystemExit(1) from exc

    root.title("ScoutIQ Video Intake Filter")
    # Bring the window to the front on launch (helps when it opens behind the IDE)
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))

    # Use themed widgets when available (Windows/macOS)
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass

    VideoIntakeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
