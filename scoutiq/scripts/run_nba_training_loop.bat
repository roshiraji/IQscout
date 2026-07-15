@echo off
REM NBA multi-source training loop (ingest -> train -> eval -> track -> repeat)
cd /d "%~dp0.."
set PY=%~dp0..\..\.venv\Scripts\python.exe
if not exist "%PY%" set PY=%~dp0..\.venv\Scripts\python.exe

echo === Ingest only (merge all NBA data sources) ===
"%PY%" -m src.training.nba_player_loop --config configs/nba_player_loop.yaml --ingest-only
if errorlevel 1 exit /b 1

echo.
echo === Training loop (1 round test — remove --max-rounds 1 for full loop) ===
"%PY%" -m src.training.nba_player_loop --config configs/nba_player_loop.yaml --max-rounds 1 --epochs 20
