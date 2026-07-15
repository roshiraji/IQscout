@echo off
REM Fusion pipeline: trained detector + LocateAnything + SAM2
cd /d "%~dp0.."
set PY=..\.venv\Scripts\python.exe
if not exist "%PY%" set PY=.venv\Scripts\python.exe

"%PY%" -m src.fusion.run_pipeline --config configs/nba_fusion_pipeline.yaml %*
