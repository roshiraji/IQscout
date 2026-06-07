@echo off
REM ScoutIQ Video Intake GUI launcher (uses Python 3.12 + Tkinter)
cd /d "%~dp0"
if not exist ".venv-gui\Scripts\python.exe" (
  echo Creating GUI environment...
  py -3.12 -m venv .venv-gui
  .venv-gui\Scripts\python.exe -m pip install opencv-python
)
.venv-gui\Scripts\python.exe app\video_intake_gui.py
