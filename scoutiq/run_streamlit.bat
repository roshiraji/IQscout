@echo off
REM ScoutIQ Streamlit launcher — uses scoutiq/.venv (CUDA PyTorch for GPU YOLO)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo scoutiq/.venv not found. Run setup from the scoutiq folder first.
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app\video_upload_filter.py
