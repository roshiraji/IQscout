# ScoutIQ service stop — terminates launcher-managed processes and frees ports 5173/5174/8501/8000.

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot ".scoutiq_runtime"
$PidFile = Join-Path $RuntimeDir "pids.json"

function Stop-ProcessSafe {
    param(
        [int]$ProcessId,
        [string]$Label
    )

    if ($ProcessId -le 0) {
        return $false
    }

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $false
    }

    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        Write-Host "Stopped $Label [pid $ProcessId]" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Could not stop $Label [pid $ProcessId]: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

function Stop-PortListeners {
    param(
        [int]$Port,
        [string]$Label
    )

    $stopped = 0
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        $procId = $conn.OwningProcess
        if ($procId -le 0) {
            continue
        }

        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $proc) {
            continue
        }

        $name = $proc.ProcessName.ToLower()
        $isTarget = $name -match '^(python|pythonw|python3|streamlit|uvicorn|node)$'
        if (-not $isTarget) {
            Write-Host "Skipping PID $procId on port $Port ($name) - not a ScoutIQ Python service." -ForegroundColor DarkGray
            continue
        }

        if (Stop-ProcessSafe -ProcessId $procId -Label "$Label on port $Port") {
            $stopped++
        }
    }

    return $stopped
}

Write-Host ""
Write-Host "Stopping ScoutIQ services..."
Write-Host ""

$stoppedCount = 0

if (Test-Path $PidFile) {
    try {
        $records = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
        foreach ($record in $records) {
            if (Stop-ProcessSafe -ProcessId ([int]$record.pid) -Label $record.service) {
                $stoppedCount++
            }
        }
    }
    catch {
        Write-Host "Warning: could not read $PidFile - $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
else {
    Write-Host "No runtime PID file found. Checking known ports..." -ForegroundColor Yellow
}

$stoppedCount += Stop-PortListeners -Port 5173 -Label "React"
$stoppedCount += Stop-PortListeners -Port 5174 -Label "React (alt)"
$stoppedCount += Stop-PortListeners -Port 8501 -Label "Streamlit"
$stoppedCount += Stop-PortListeners -Port 8000 -Label "FastAPI"

if (Test-Path $PidFile) {
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Removed runtime PID file." -ForegroundColor DarkGray
}

Write-Host ""
if ($stoppedCount -gt 0) {
    Write-Host "ScoutIQ cleanup complete. Stopped $stoppedCount process(es)." -ForegroundColor Green
}
else {
    Write-Host "No ScoutIQ launcher processes were running." -ForegroundColor Yellow
}
Write-Host ""

exit 0
