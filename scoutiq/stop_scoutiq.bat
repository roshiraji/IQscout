@echo off
REM ============================================================================
REM ScoutIQ / VisionPlay AI — stop all launcher-managed services
REM
REM Double-click this file to terminate Streamlit, FastAPI/Uvicorn, and any
REM Python processes bound to ports 8501 or 8000 that were started by ScoutIQ.
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo Stopping ScoutIQ services...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_services.ps1"

echo.
pause
exit /b 0
