# ScoutIQ Video Intake GUI launcher (uses Python 3.12 + Tkinter)
Set-Location $PSScriptRoot

$guiPython = Join-Path $PSScriptRoot ".venv-gui\Scripts\python.exe"
if (-not (Test-Path $guiPython)) {
    Write-Host "Creating GUI environment..."
    py -3.12 -m venv .venv-gui
    & (Join-Path $PSScriptRoot ".venv-gui\Scripts\python.exe") -m pip install opencv-python
}

& $guiPython app\video_intake_gui.py
