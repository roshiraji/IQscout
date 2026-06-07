# ScoutIQ Streamlit launcher — uses scoutiq/.venv (CUDA PyTorch for GPU YOLO)
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "scoutiq/.venv not found. Run setup from the scoutiq folder first."
    exit 1
}

& $python -m streamlit run app\video_upload_filter.py
