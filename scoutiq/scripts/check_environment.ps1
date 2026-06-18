# ScoutIQ environment check — verifies venv, folders, and required Python packages.
# Called by run_scoutiq.bat before any services start.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigFile = Join-Path $PSScriptRoot "runtime_config.json"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"

$RequiredFolders = @(
    "data\input_videos",
    "outputs\intake_reports",
    "outputs\annotated_videos",
    "outputs\detections",
    "outputs\player_snapshots",
    "outputs\tracking",
    "outputs\tracking_videos",
    "outputs\tracking_selections"
)

$RequiredImports = @(
    "cv2",
    "fastapi",
    "uvicorn",
    "ultralytics"
)

function Write-CheckFail {
    param([string]$Message)
    Write-Host ""
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix: activate the virtual environment and install dependencies:" -ForegroundColor Yellow
    Write-Host "  cd `"$ProjectRoot`"" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "ScoutIQ environment check"
Write-Host "Project root: $ProjectRoot"
Write-Host ""

if (-not (Test-Path (Join-Path $ProjectRoot ".venv"))) {
    Write-CheckFail "Virtual environment not found at .venv"
}

if (-not (Test-Path $PythonExe)) {
    Write-CheckFail "Python interpreter not found at .venv\Scripts\python.exe"
}

if (-not (Test-Path $RequirementsFile)) {
    Write-CheckFail "requirements.txt not found in the project root."
}

foreach ($folder in $RequiredFolders) {
    $fullPath = Join-Path $ProjectRoot $folder
    if (-not (Test-Path $fullPath)) {
        Write-Host "Creating missing folder: $folder" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

$importScript = "import " + ($RequiredImports -join ", ")
$importCheck = & $PythonExe -c $importScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $importCheck
    Write-CheckFail "One or more required packages are missing ($($RequiredImports -join ', '))."
}

$config = $null
if (Test-Path $ConfigFile) {
    $config = Get-Content -Path $ConfigFile -Raw | ConvertFrom-Json
}

if ($null -eq $config) {
    Write-CheckFail "Missing or invalid scripts/runtime_config.json"
}

if ($config.react.enabled) {
    $frontEndDir = Join-Path $ProjectRoot $config.react.directory
    $packageJson = Join-Path $frontEndDir "package.json"

    if (-not (Test-Path $packageJson)) {
        Write-CheckFail "React frontend not found at $($config.react.directory). Check out branch f-Anne_ReactComponent or restore FrontEnd/."
    }

    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $nodeCmd -or -not $npmCmd) {
        Write-CheckFail "Node.js and npm are required for the React dashboard. Install from https://nodejs.org/"
    }

    if (-not (Test-Path (Join-Path $frontEndDir "node_modules"))) {
        Write-Host "Installing React dependencies (first run)..." -ForegroundColor Yellow
        Push-Location $frontEndDir
        try {
            & npm.cmd install
            if ($LASTEXITCODE -ne 0) {
                Write-CheckFail "npm install failed in $($config.react.directory)."
            }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "[OK] React frontend found at $($config.react.directory)." -ForegroundColor Green
    Write-Host "[OK] Node: $($nodeCmd.Source)" -ForegroundColor Green
}

if ($config.streamlit.enabled) {
    $streamlitCheck = & $PythonExe -c "import streamlit" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host $streamlitCheck
        Write-CheckFail "Streamlit is enabled in runtime_config.json but not installed."
    }
    Write-Host "[OK] Streamlit available (legacy UI enabled)." -ForegroundColor Green
}

Write-Host "[OK] Virtual environment found." -ForegroundColor Green
Write-Host "[OK] Python: $PythonExe" -ForegroundColor Green
Write-Host "[OK] Required folders exist." -ForegroundColor Green
Write-Host "[OK] Required packages import successfully." -ForegroundColor Green
Write-Host ""

exit 0
