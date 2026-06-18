# ScoutIQ service start — launches React dashboard, FastAPI, and optional Streamlit.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot ".scoutiq_runtime"
$PidFile = Join-Path $RuntimeDir "pids.json"
$ConfigFile = Join-Path $PSScriptRoot "runtime_config.json"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Resolve-FastApiModule {
    param([string]$ConfiguredModule)

    $candidates = @(
        $ConfiguredModule,
        "app.api:app",
        "src.api:app",
        "app.api_server:app"
    ) | Select-Object -Unique

    foreach ($module in $candidates) {
        $modulePath = ($module -split ":")[0]
        $relativePath = ($modulePath -replace "\.", "\") + ".py"
        $filePath = Join-Path $ProjectRoot $relativePath
        if (Test-Path $filePath) {
            return $module
        }
    }

    return $null
}

function Add-ServiceRecord {
    param(
        [System.Collections.Generic.List[object]]$Records,
        [string]$Service,
        [int]$ProcessId,
        [string]$Command,
        [int]$Port
    )

    $Records.Add([pscustomobject]@{
        service   = $Service
        pid       = $ProcessId
        command   = $Command
        port      = $Port
        timestamp = (Get-Date).ToString("o")
    }) | Out-Null
}

function Start-LoggedProcess {
    param(
        [string]$ServiceName,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [int]$Port,
        [string]$WorkingDirectory = $ProjectRoot
    )

    $logPath = Join-Path $RuntimeDir "$ServiceName.log"
    $errPath = Join-Path $RuntimeDir "$ServiceName.err.log"

    $quotedArgs = foreach ($arg in $ArgumentList) {
        if ($arg -match '\s') { "`"$arg`"" } else { $arg }
    }
    $argumentString = $quotedArgs -join " "
    $commandLine = "`"$FilePath`" $argumentString"

    # PowerShell requires separate files for stdout and stderr redirection.
    if (Test-Path $logPath) { Remove-Item $logPath -Force -ErrorAction SilentlyContinue }
    if (Test-Path $errPath) { Remove-Item $errPath -Force -ErrorAction SilentlyContinue }

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $argumentString `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -PassThru `
        -WindowStyle Hidden

    Start-Sleep -Seconds 4

    if ($process.HasExited) {
        $logHint = "See logs: $logPath and $errPath"
        if (Test-Path $errPath) {
            $stderrTail = Get-Content -Path $errPath -Tail 5 -ErrorAction SilentlyContinue
            if ($stderrTail) {
                $logHint += "`nLast stderr lines:`n$($stderrTail -join "`n")"
            }
        }
        throw "$ServiceName failed to start. $logHint"
    }

    return @{
        Process = $process
        Command = $commandLine
        LogPath = $logPath
        ErrPath = $errPath
    }
}

# Clean up any previous launcher-managed processes before starting fresh.
& (Join-Path $PSScriptRoot "stop_services.ps1") | Out-Null

if (-not (Test-Path $ConfigFile)) {
    throw "Missing runtime config: $ConfigFile"
}

$config = Get-Content -Path $ConfigFile -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

$records = New-Object System.Collections.Generic.List[object]
$startedServices = @()

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ScoutIQ / VisionPlay AI Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Python:  $PythonExe"
Write-Host ""

try {
    if ($config.fastapi.enabled) {
        $fastApiModule = Resolve-FastApiModule -ConfiguredModule $config.fastapi.module
        $fastApiPort = [int]$config.fastapi.port

        if ($null -eq $fastApiModule) {
            Write-Host "[SKIP] FastAPI backend not found. Skipping API startup." -ForegroundColor Yellow
        }
        else {
            $uvicornArgs = @(
                "-m", "uvicorn", $fastApiModule,
                "--host", "127.0.0.1",
                "--port", "$fastApiPort",
                "--reload"
            )

            $started = Start-LoggedProcess `
                -ServiceName "fastapi" `
                -FilePath $PythonExe `
                -ArgumentList $uvicornArgs `
                -Port $fastApiPort

            Add-ServiceRecord -Records $records -Service "fastapi" -ProcessId $started.Process.Id `
                -Command $started.Command -Port $fastApiPort

            $fastApiDocsUrl = "http://localhost:$fastApiPort/docs"
            Write-Host "[STARTED] FastAPI backend ($fastApiModule)" -ForegroundColor Green
            Write-Host "          API:  http://localhost:$fastApiPort/api/health"
            Write-Host "          Docs: $fastApiDocsUrl"
            Write-Host "          Log:  $($started.LogPath)"
            $startedServices += "FastAPI (http://localhost:$fastApiPort)"

            if ($config.browser.open_fastapi_docs) {
                Start-Process $fastApiDocsUrl | Out-Null
            }
        }
    }

    if ($config.react.enabled) {
        $frontEndDir = Join-Path $ProjectRoot $config.react.directory
        $reactPort = [int]$config.react.port
        $viteJs = Join-Path $frontEndDir "node_modules\vite\bin\vite.js"
        $nodeExe = (Get-Command node -ErrorAction Stop).Source

        if (-not (Test-Path $viteJs)) {
            throw "Vite not found in $($config.react.directory). Run: cd $($config.react.directory) && npm install"
        }

        $viteArgs = @($viteJs, "--host", "--port", "$reactPort")
        $started = Start-LoggedProcess `
            -ServiceName "react" `
            -FilePath $nodeExe `
            -ArgumentList $viteArgs `
            -Port $reactPort `
            -WorkingDirectory $frontEndDir

        Add-ServiceRecord -Records $records -Service "react" -ProcessId $started.Process.Id `
            -Command $started.Command -Port $reactPort

        $reactUrl = "http://localhost:$reactPort"
        Write-Host "[STARTED] React dashboard (WNBA UI)" -ForegroundColor Green
        Write-Host "          URL:  $reactUrl"
        Write-Host "          Log:  $($started.LogPath)"
        $startedServices += "React ($reactUrl)"

        if ($config.browser.open_react) {
            Start-Process $reactUrl | Out-Null
        }
    }

    if ($config.streamlit.enabled) {
        $streamlitScript = Join-Path $ProjectRoot ($config.streamlit.script)
        $streamlitPort = [int]$config.streamlit.port

        if (-not (Test-Path $streamlitScript)) {
            throw "Streamlit script not found: $($config.streamlit.script)"
        }

        # Always launch via python -m streamlit for reliable quoting on paths with spaces.
        $streamlitArgs = @(
            "-m", "streamlit", "run", $streamlitScript,
            "--server.port", "$streamlitPort",
            "--server.headless", "true"
        )

        $started = Start-LoggedProcess `
            -ServiceName "streamlit" `
            -FilePath $PythonExe `
            -ArgumentList $streamlitArgs `
            -Port $streamlitPort

        Add-ServiceRecord -Records $records -Service "streamlit" -ProcessId $started.Process.Id `
            -Command $started.Command -Port $streamlitPort

        $streamlitUrl = "http://localhost:$streamlitPort"
        Write-Host "[STARTED] Streamlit video upload filter" -ForegroundColor Green
        Write-Host "          URL:  $streamlitUrl"
        Write-Host "          Log:  $($started.LogPath)"
        $startedServices += "Streamlit ($streamlitUrl)"

        if ($config.browser.open_streamlit) {
            Start-Process $streamlitUrl | Out-Null
        }
    }

    if ($records.Count -eq 0) {
        throw "No services were started. Check scripts/runtime_config.json."
    }

    $records | ConvertTo-Json -Depth 4 | Set-Content -Path $PidFile -Encoding UTF8

    Write-Host ""
    Write-Host "Services running. Press Ctrl+C or close this window to stop all services." -ForegroundColor Cyan
    Write-Host ""

    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($record in $records) {
            $alive = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
            if (-not $alive) {
                throw "$($record.service) exited unexpectedly [pid $($record.pid)]. Check .scoutiq_runtime\$($record.service).log"
            }
        }
    }
}
catch {
    Write-Host ""
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Write-Host ""
    Write-Host "Shutting down ScoutIQ services..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "stop_services.ps1")
}
