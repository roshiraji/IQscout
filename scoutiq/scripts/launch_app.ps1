# ScoutIQ desktop-style app launcher.
# Starts API + React hidden, shows a splash while booting, opens an app window,
# and stops everything when that window is closed (or Quit is pressed).

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot ".scoutiq_runtime"
$PidFile = Join-Path $RuntimeDir "pids.json"
$QuitFlag = Join-Path $RuntimeDir "quit.requested"
$LockFile = Join-Path $RuntimeDir "launcher.lock"
$ConfigFile = Join-Path $PSScriptRoot "runtime_config.json"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StopScript = Join-Path $PSScriptRoot "stop_services.ps1"

function Show-FatalError {
    param([string]$Message)
    [System.Windows.Forms.MessageBox]::Show(
        $Message,
        "ScoutIQ",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
}

function Test-LauncherAlreadyRunning {
    if (-not (Test-Path $LockFile)) {
        return $false
    }
    try {
        $payload = Get-Content -Path $LockFile -Raw | ConvertFrom-Json
        $pidValue = [int]$payload.pid
        $alive = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        return [bool]$alive
    }
    catch {
        return $false
    }
}

function Enter-LauncherLock {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    if (Test-LauncherAlreadyRunning) {
        throw "ScoutIQ is already running.`nClose the existing ScoutIQ window first."
    }
    @{
        pid = $PID
        started = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Path $LockFile -Encoding UTF8
}

function Exit-LauncherLock {
    if (Test-Path $LockFile) {
        try {
            $payload = Get-Content -Path $LockFile -Raw | ConvertFrom-Json
            if ([int]$payload.pid -eq $PID) {
                Remove-Item -Path $LockFile -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            Remove-Item -Path $LockFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-FastApiModule {
    param([string]$ConfiguredModule)
    $candidates = @(
        $ConfiguredModule,
        "app.api_server:app",
        "app.api:app",
        "src.api:app"
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

function Find-BrowserExe {
    $candidates = @(
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:LOCALAPPDATA\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
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
    if (Test-Path $logPath) { Remove-Item $logPath -Force -ErrorAction SilentlyContinue }
    if (Test-Path $errPath) { Remove-Item $errPath -Force -ErrorAction SilentlyContinue }

    $quotedArgs = foreach ($arg in $ArgumentList) {
        if ($arg -match '\s') { "`"$arg`"" } else { $arg }
    }
    $argumentString = $quotedArgs -join " "
    $commandLine = "`"$FilePath`" $argumentString"

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $argumentString `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -PassThru `
        -WindowStyle Hidden

    Start-Sleep -Milliseconds 1500
    if ($process.HasExited) {
        $tail = ""
        if (Test-Path $errPath) {
            $tail = (Get-Content -Path $errPath -Tail 8 -ErrorAction SilentlyContinue) -join "`n"
        }
        throw "$ServiceName failed to start. See $errPath`n$tail"
    }

    return @{
        Process = $process
        Command = $commandLine
        LogPath = $logPath
    }
}

function Test-HttpOk {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    }
    catch {
        return $false
    }
}

function Get-ScoutStoryBeats {
    return @(
        @{ Title = "Ingest the clip"; Detail = "Validate WNBA film for tracking." },
        @{ Title = "See every player"; Detail = "Segment athletes across the floor." },
        @{ Title = "Keep identities stable"; Detail = "Stitch tracklets into person groups." },
        @{ Title = "Pick the matchup"; Detail = "Select the players that matter." },
        @{ Title = "Clear the noise"; Detail = "Isolate so footwork stays readable." },
        @{ Title = "Read possession"; Detail = "Ball + roles mark offense and defense." },
        @{ Title = "Coach the film"; Detail = "Matchup Outcomes turn film into insight." }
    )
}

function New-SplashForm {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "ScoutIQ"
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.ControlBox = $true
    $form.TopMost = $true
    $form.Width = 520
    $form.Height = 360
    $form.BackColor = [System.Drawing.Color]::FromArgb(10, 10, 10)
    $form.ForeColor = [System.Drawing.Color]::White

    $eyebrow = New-Object System.Windows.Forms.Label
    $eyebrow.Text = "WNBA VISION ANALYTICS"
    $eyebrow.Font = New-Object System.Drawing.Font("Segoe UI", 8, [System.Drawing.FontStyle]::Bold)
    $eyebrow.ForeColor = [System.Drawing.Color]::FromArgb(140, 140, 140)
    $eyebrow.AutoSize = $false
    $eyebrow.TextAlign = "MiddleCenter"
    $eyebrow.Width = 480
    $eyebrow.Height = 20
    $eyebrow.Left = 20
    $eyebrow.Top = 18
    $form.Controls.Add($eyebrow)

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "ScoutIQ"
    $title.Font = New-Object System.Drawing.Font("Segoe UI", 28, [System.Drawing.FontStyle]::Bold)
    $title.ForeColor = [System.Drawing.Color]::White
    $title.AutoSize = $false
    $title.TextAlign = "MiddleCenter"
    $title.Width = 480
    $title.Height = 48
    $title.Left = 20
    $title.Top = 38
    $form.Controls.Add($title)

    $subtitle = New-Object System.Windows.Forms.Label
    $subtitle.Text = "Film in. Matchup clarity out."
    $subtitle.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $subtitle.ForeColor = [System.Drawing.Color]::FromArgb(179, 179, 179)
    $subtitle.AutoSize = $false
    $subtitle.TextAlign = "MiddleCenter"
    $subtitle.Width = 480
    $subtitle.Height = 24
    $subtitle.Left = 20
    $subtitle.Top = 88
    $form.Controls.Add($subtitle)

    $storyTitle = New-Object System.Windows.Forms.Label
    $storyTitle.Name = "StoryTitle"
    $storyTitle.Text = "How ScoutIQ works"
    $storyTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
    $storyTitle.ForeColor = [System.Drawing.Color]::White
    $storyTitle.AutoSize = $false
    $storyTitle.TextAlign = "MiddleCenter"
    $storyTitle.Width = 480
    $storyTitle.Height = 28
    $storyTitle.Left = 20
    $storyTitle.Top = 130
    $form.Controls.Add($storyTitle)

    $storyDetail = New-Object System.Windows.Forms.Label
    $storyDetail.Name = "StoryDetail"
    $storyDetail.Text = "Turning clips into coachable insight..."
    $storyDetail.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $storyDetail.ForeColor = [System.Drawing.Color]::FromArgb(190, 190, 190)
    $storyDetail.AutoSize = $false
    $storyDetail.TextAlign = "MiddleCenter"
    $storyDetail.Width = 480
    $storyDetail.Height = 40
    $storyDetail.Left = 20
    $storyDetail.Top = 158
    $form.Controls.Add($storyDetail)

    $status = New-Object System.Windows.Forms.Label
    $status.Name = "StatusLabel"
    $status.Text = "Starting local engine..."
    $status.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $status.ForeColor = [System.Drawing.Color]::FromArgb(210, 210, 210)
    $status.AutoSize = $false
    $status.TextAlign = "MiddleCenter"
    $status.Width = 480
    $status.Height = 24
    $status.Left = 20
    $status.Top = 220
    $form.Controls.Add($status)

    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.Name = "ProgressBar"
    $bar.Style = "Marquee"
    $bar.MarqueeAnimationSpeed = 28
    $bar.Width = 400
    $bar.Height = 16
    $bar.Left = 60
    $bar.Top = 258
    $form.Controls.Add($bar)

    return $form
}

function Set-SplashStatus {
    param($Form, [string]$Text)
    $label = $Form.Controls["StatusLabel"]
    if ($label) { $label.Text = $Text }
    [System.Windows.Forms.Application]::DoEvents()
}

function Set-SplashStory {
    param($Form, [hashtable]$Beat)
    $title = $Form.Controls["StoryTitle"]
    $detail = $Form.Controls["StoryDetail"]
    if ($title) { $title.Text = [string]$Beat.Title }
    if ($detail) { $detail.Text = [string]$Beat.Detail }
    [System.Windows.Forms.Application]::DoEvents()
}

function Start-SplashStoryTimer {
    param($Form)
    $beats = @(Get-ScoutStoryBeats)
    $Form.Tag = @{ Beats = $beats; Index = 0 }
    Set-SplashStory -Form $Form -Beat $beats[0]
    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 2600
    $timer.add_Tick({
        param($sender, $eventArgs)
        $hostForm = $sender.Tag
        if (-not $hostForm -or -not $hostForm.Tag) { return }
        $meta = $hostForm.Tag
        $meta.Index = ($meta.Index + 1) % $meta.Beats.Count
        $hostForm.Tag = $meta
        Set-SplashStory -Form $hostForm -Beat $meta.Beats[$meta.Index]
    })
    $timer.Tag = $Form
    $timer.Start()
    return $timer
}

# --- Main -------------------------------------------------------------------

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
if (Test-Path $QuitFlag) { Remove-Item $QuitFlag -Force -ErrorAction SilentlyContinue }

try {
    Enter-LauncherLock
}
catch {
    Show-FatalError -Message $_.Exception.Message
    exit 1
}

$splash = New-SplashForm
$splash.Show()
$storyTimer = $null
$storyTimer = Start-SplashStoryTimer -Form $splash
Set-SplashStatus -Form $splash -Text "Starting local engine..."
[System.Windows.Forms.Application]::DoEvents()

try {
    # Quiet environment check (show MessageBox on failure instead of console).
    if (-not (Test-Path $PythonExe)) {
        throw "Virtual environment not found.`nExpected: $PythonExe"
    }
    if (-not (Test-Path $ConfigFile)) {
        throw "Missing runtime config:`n$ConfigFile"
    }

    $config = Get-Content -Path $ConfigFile -Raw | ConvertFrom-Json
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($config.react.enabled -and -not $nodeCmd) {
        throw "Node.js is required for the ScoutIQ dashboard.`nInstall Node.js, then try again."
    }

    Set-SplashStatus -Form $splash -Text "Clearing any previous session..."
    & $StopScript | Out-Null

    $records = New-Object System.Collections.Generic.List[object]
    $reactPort = [int]$config.react.port
    $fastApiPort = [int]$config.fastapi.port
    $reactUrl = "http://127.0.0.1:$reactPort"
    $healthUrl = "http://127.0.0.1:$fastApiPort/api/health"

    if ($config.fastapi.enabled) {
        Set-SplashStatus -Form $splash -Text "Loading vision backend..."
        $fastApiModule = Resolve-FastApiModule -ConfiguredModule $config.fastapi.module
        if ($null -eq $fastApiModule) {
            throw "FastAPI module not found. Check scripts/runtime_config.json."
        }
        $uvicornArgs = @(
            "-m", "uvicorn", $fastApiModule,
            "--host", "127.0.0.1",
            "--port", "$fastApiPort"
        )
        # No --reload in app mode (cleaner process tree on quit).
        $started = Start-LoggedProcess `
            -ServiceName "fastapi" `
            -FilePath $PythonExe `
            -ArgumentList $uvicornArgs `
            -Port $fastApiPort
        Add-ServiceRecord -Records $records -Service "fastapi" `
            -ProcessId $started.Process.Id -Command $started.Command -Port $fastApiPort
    }

    if ($config.react.enabled) {
        Set-SplashStatus -Form $splash -Text "Opening coaching workspace..."
        $frontEndDir = Join-Path $ProjectRoot $config.react.directory
        $viteJs = Join-Path $frontEndDir "node_modules\vite\bin\vite.js"
        if (-not (Test-Path $viteJs)) {
            throw "Frontend dependencies missing.`nRun: cd FrontEnd && npm install"
        }
        $nodeExe = $nodeCmd.Source
        $viteArgs = @($viteJs, "--host", "127.0.0.1", "--port", "$reactPort")
        $started = Start-LoggedProcess `
            -ServiceName "react" `
            -FilePath $nodeExe `
            -ArgumentList $viteArgs `
            -Port $reactPort `
            -WorkingDirectory $frontEndDir
        Add-ServiceRecord -Records $records -Service "react" `
            -ProcessId $started.Process.Id -Command $started.Command -Port $reactPort
    }

    if ($records.Count -eq 0) {
        throw "No services were enabled in runtime_config.json."
    }
    $records | ConvertTo-Json -Depth 4 | Set-Content -Path $PidFile -Encoding UTF8

    Set-SplashStatus -Form $splash -Text "Warming models and dashboard..."
    $deadline = (Get-Date).AddSeconds(90)
    $apiReady = -not $config.fastapi.enabled
    $uiReady = -not $config.react.enabled
    while ((Get-Date) -lt $deadline) {
        if (-not $apiReady) { $apiReady = Test-HttpOk -Url $healthUrl }
        if (-not $uiReady) { $uiReady = Test-HttpOk -Url $reactUrl }
        if ($apiReady -and $uiReady) { break }
        Set-SplashStatus -Form $splash -Text $(
            if (-not $apiReady) { "Connecting vision engine..." }
            elseif (-not $uiReady) { "Preparing coaching UI..." }
            else { "Almost ready..." }
        )
        Start-Sleep -Milliseconds 400
        [System.Windows.Forms.Application]::DoEvents()
        if (-not $splash.Visible) {
            throw "Startup cancelled."
        }
    }
    if (-not ($apiReady -and $uiReady)) {
        throw "Timed out waiting for ScoutIQ services.`nCheck logs in:`n$RuntimeDir"
    }

    Set-SplashStatus -Form $splash -Text "Launching ScoutIQ..."
    $browserExe = Find-BrowserExe
    $browserProc = $null
    $profileDir = Join-Path $RuntimeDir "browser_profile"

    if ($browserExe) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
        $browserProc = Start-Process -FilePath $browserExe -ArgumentList @(
            "--app=$reactUrl",
            "--user-data-dir=$profileDir",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions"
        ) -PassThru
    }
    else {
        # Fallback: default browser (close detection uses Quit button / quit flag).
        Start-Process $reactUrl | Out-Null
    }

    $splash.Hide()
    $splash.Close()

    function Test-AppBrowserAlive {
        param([string]$ProfilePath)
        $names = @("msedge.exe", "chrome.exe")
        foreach ($name in $names) {
            $procs = Get-CimInstance Win32_Process -Filter "Name = '$name'" -ErrorAction SilentlyContinue
            foreach ($proc in $procs) {
                if ($proc.CommandLine -and ($proc.CommandLine -like "*$ProfilePath*")) {
                    return $true
                }
            }
        }
        return $false
    }

    if ($browserProc) {
        # Chromium may re-parent; wait until our app-profile browser is gone, or Quit.
        Start-Sleep -Seconds 2
        while ($true) {
            if (Test-Path $QuitFlag) { break }
            $parentAlive = $false
            try {
                $browserProc.Refresh()
                $parentAlive = -not $browserProc.HasExited
            }
            catch {
                $parentAlive = $false
            }
            $profileAlive = Test-AppBrowserAlive -ProfilePath $profileDir
            if (-not $parentAlive -and -not $profileAlive) { break }
            Start-Sleep -Milliseconds 800
        }
    }
    else {
        while (-not (Test-Path $QuitFlag)) {
            Start-Sleep -Seconds 1
        }
    }
}
catch {
    if ($splash -and $splash.Visible) {
        $splash.Hide()
        $splash.Close()
    }
    Show-FatalError -Message $_.Exception.Message
    exit 1
}
finally {
    if ($storyTimer) {
        try { $storyTimer.Stop(); $storyTimer.Dispose() } catch { }
    }
    if ($splash -and -not $splash.IsDisposed) {
        try { $splash.Close() } catch { }
    }
    if (Test-Path $QuitFlag) {
        Remove-Item $QuitFlag -Force -ErrorAction SilentlyContinue
    }
    & $StopScript | Out-Null
    Exit-LauncherLock
}

exit 0

