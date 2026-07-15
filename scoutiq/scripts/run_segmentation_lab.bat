@echo off
REM Run SAM segmentation lab on the fine-tune clip (isolated from main pipeline).
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo ScoutIQ venv not found. Create it first: python -m venv .venv
  exit /b 1
)

set EXTRA=%*
if "%EXTRA%"=="" set EXTRA=--open-report

echo.
echo Segmentation lab — testing SAM on fine-tune clip...
echo.

call .venv\Scripts\python.exe -m src.segmentation_lab.run %EXTRA%
