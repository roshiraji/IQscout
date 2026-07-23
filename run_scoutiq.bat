@echo off
REM ============================================================================
REM ScoutIQ
REM
REM Double-click to start the whole app.
REM Close the ScoutIQ window (X or Quit) to stop servers and exit.
REM This is the only launcher you need.
REM ============================================================================

setlocal

REM First click: reopen ourselves minimized so no big console stays visible.
if /I not "%~1"=="_hidden" (
    start "ScoutIQ" /min "%~f0" _hidden
    exit /b 0
)

cd /d "%~dp0scoutiq"

if not exist "scripts\launch_app.ps1" (
    echo Could not find ScoutIQ launcher:
    echo   %CD%\scripts\launch_app.ps1
    pause
    exit /b 1
)

REM Blocks until the app window is closed, then stops API + dashboard and exits.
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%CD%\scripts\launch_app.ps1"
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
    echo ScoutIQ exited with an error.
    echo Check MessageBox details or logs in scoutiq\.scoutiq_runtime\
    pause
)

exit /b %EXIT_CODE%
