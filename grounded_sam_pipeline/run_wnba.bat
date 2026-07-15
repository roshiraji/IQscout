@echo off
REM Annotate WNBA video frames with Grounded SAM (calls scoutiq bridge).
cd /d "%~dp0"
set PY=%~dp0..\.venv\Scripts\python.exe
set SCOUTIQ=%~dp0..\scoutiq

if not exist "%PY%" (
    echo ERROR: Virtual env not found at ..\.venv
    exit /b 1
)

cd /d "%SCOUTIQ%"
"%PY%" -m src.ingest.wnba_grounded_sam %*
