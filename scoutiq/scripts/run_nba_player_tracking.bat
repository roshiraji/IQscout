@echo off
REM Track players on 6-28.mp4 using trained best_player_detector.pt
cd /d "%~dp0.."
set PY=..\.venv\Scripts\python.exe
if not exist "%PY%" set PY=.venv\Scripts\python.exe

"%PY%" -m src.track.run --config configs/nba_player_tracking.yaml
