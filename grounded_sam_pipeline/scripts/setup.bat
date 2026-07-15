@echo off
REM Setup from ScoutIQ repo root (uses shared .venv)
cd /d "%~dp0..\.."
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r grounded_sam_pipeline\requirements.txt
pip install -e GroundingDINO
pip install -e .
pip install pytest
echo.
echo Setup complete. Run from grounded_sam_pipeline:
echo   cd grounded_sam_pipeline
echo   run.bat --validate-only
