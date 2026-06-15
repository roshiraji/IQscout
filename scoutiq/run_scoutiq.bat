@echo off
REM ============================================================================
REM ScoutIQ / VisionPlay AI — unified local development launcher
REM
REM Double-click this file (or run from cmd) to:
REM   1. Verify the .venv environment, Node.js, and required packages
REM   2. Start FastAPI (port 8000) and the React dashboard (port 5173)
REM   3. Open the React app in your default browser
REM   4. Keep one command window open with service status
REM
REM Optional: enable Streamlit in scripts\runtime_config.json (port 8501).
REM
REM Press Ctrl+C or close this window to stop all launcher-managed services.
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo ScoutIQ launcher starting...
echo Project folder: %CD%
echo.

REM --- Step 1: Environment check ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\check_environment.ps1"
if errorlevel 1 (
    echo.
    echo Launcher stopped due to environment check failure.
    pause
    exit /b 1
)

REM --- Step 2: Start services (blocks until Ctrl+C; cleanup runs in start_services.ps1 finally) ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_services.ps1"
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% NEQ 0 (
    echo ScoutIQ launcher exited with errors.
) else (
    echo ScoutIQ launcher stopped.
)
echo.
pause
exit /b %EXIT_CODE%
