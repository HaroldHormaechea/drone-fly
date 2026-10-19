<#
.SYNOPSIS
  Fly a trained model live in Liftoff, recorded: Liftoff replay + screen video (with the overlay)
  per attempt, then a flight report in Liftoff metres.

.DESCRIPTION
  Per attempt: start an ffmpeg screen capture (fragmented MP4, so stopping it never corrupts the
  file) -> `harness run` with --recordable (overlay visible in the video) and --save-recording ->
  stop the capture -> `harness flight-report --recording` (heights / distances from telemetry.csv,
  ground-truth gate count from the replay) -> reset the level for the next attempt.

  The level flown is Liftoff's LAST SESSION (Start-LiftoffSession.ps1 loads it when Liftoff is not
  running yet). Refuses to start while League / Riot / Vanguard is running (the harness would
  refuse anyway): close them first, if the user allows it.

  --extra-dim comes from the checkpoint's model_readout.contract.json sidecar when there is one;
  pass -ExtraDim for older checkpoints without it (e.g. model 11 = 70).

.EXAMPLE
  .\scripts\liftoff\Invoke-ModelFlight.ps1 -Checkpoint training/13-deploy-robust/model_readout.pt -Config artifacts/harness/live-m13-w512.yaml

.EXAMPLE
  .\scripts\liftoff\Invoke-ModelFlight.ps1 -Checkpoint training/11-count-curriculum/model_readout.pt -ExtraDim 70 -Config artifacts/harness/live-eyes-reacq-w512.yaml -Attempts 3 -Quit
#>
param(
    [Parameter(Mandatory)][string]$Checkpoint,
    [string]$Config = 'artifacts/harness/live-eyes-reacq-w512.yaml',
    [ValidateSet(0, 12, 30, 70, 98)][int]$ExtraDim = 0,
    [string]$Preset = 'liftoff',
    [string]$K1Path = '',
    [int]$Attempts = 1,
    [double]$MaxSeconds = 90,
    [string]$MapName = 'infinity-loop',
    [string]$ModelName = '',
    [string]$TrackdataRoot = 'artifacts/harness/trackdata',
    [string]$OutputRoot = 'artifacts/harness/liftoff/runs',
    [string]$ScreenDir = 'artifacts/harness/liftoff/screen',
    [switch]$NoScreen,
    [switch]$NoOverlay,
    [switch]$Quit
)
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')

function Resolve-RepoPath([string]$p) {
    if ([IO.Path]::IsPathRooted($p)) { return $p }
    return Join-Path $RepoRoot $p
}

# --- checks -------------------------------------------------------------------------------------
$ckpt = Resolve-RepoPath $Checkpoint
if (-not (Test-Path $ckpt)) { throw "checkpoint not found: $ckpt" }
if ($ExtraDim -eq 0) {
    $sidecar = [IO.Path]::ChangeExtension($ckpt, '.contract.json')
    if (-not (Test-Path $sidecar)) { throw "no ${sidecar}: pass -ExtraDim (12/30/70/98)" }
    $ExtraDim = [int](Get-Content $sidecar -Raw | ConvertFrom-Json).extra_dim
}
if (-not (Test-Path (Resolve-RepoPath $Config))) { throw "config not found: $Config" }
if (-not $K1Path) {
    $K1Path = @($env:DRONE_FLY_K1_PATH, 'artifacts/harness/liftoff/training-data/pruned/k1',
        'artifacts/pruned/k1') | Where-Object { $_ -and (Test-Path (Resolve-RepoPath $_)) } |
        Select-Object -First 1
    if (-not $K1Path) { throw 'K1 connectome not found: pass -K1Path or set DRONE_FLY_K1_PATH' }
}
if (-not $ModelName) { $ModelName = Split-Path (Split-Path $ckpt -Parent) -Leaf }
$antiCheat = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(vgc|vgtray|RiotClient.*|League of Legends.*)$' }
if ($antiCheat) {
    throw "anti-cheat running: $(($antiCheat.Name | Sort-Object -Unique) -join ', ') (close it first, if the user allows)"
}
$ffmpeg = $null
if (-not $NoScreen) {
    $ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
    if (-not $ffmpeg) {
        $ffmpeg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe" `
            -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $ffmpeg) { throw 'ffmpeg not found (winget install Gyan.FFmpeg), or pass -NoScreen' }
    $screenRoot = Resolve-RepoPath $ScreenDir
    New-Item -ItemType Directory -Force $screenRoot | Out-Null
}

if (-not (Get-Process Liftoff -ErrorAction SilentlyContinue)) {
    & (Join-Path $PSScriptRoot 'Start-LiftoffSession.ps1')
}

# --- attempts -----------------------------------------------------------------------------------
$stamp = (Get-Date).ToString('yyyyMMdd-HHmm')
$results = @()
for ($n = 1; $n -le $Attempts; $n++) {
    $name = "$MapName-model-$ModelName-$stamp-attempt-$n"
    $video = $null
    $capture = $null
    if ($ffmpeg) {
        $video = Join-Path $screenRoot "$name.mp4"
        # Fragmented MP4: playable even though the capture is stopped with Stop-Process.
        $capture = Start-Process $ffmpeg -PassThru -WindowStyle Hidden `
            -RedirectStandardError (Join-Path $screenRoot "$name.ffmpeg.log") -ArgumentList @(
            '-y', '-f', 'gdigrab', '-framerate', '30', '-draw_mouse', '0', '-i', 'desktop',
            '-t', "$([int]$MaxSeconds + 60)", '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            "`"$video`"")
        Start-Sleep -Seconds 2
    }
    $runArgs = @(
        'harness', 'run', '--harness', 'liftoff', '--preset', $Preset, '--config', $Config,
        '--extra-dim', "$ExtraDim", '--checkpoint', $Checkpoint, '--k1-path', $K1Path,
        '--output-root', $OutputRoot, '--duration', "$MaxSeconds", '--save-recording', $name
    )
    if (-not $NoOverlay) { $runArgs += '--recordable' }
    Write-Host "=== attempt $n / $Attempts ($name)"
    $before = @(Get-ChildItem -Directory (Resolve-RepoPath $OutputRoot) -ErrorAction SilentlyContinue |
        ForEach-Object Name)
    Push-Location $RepoRoot
    try {
        $ErrorActionPreference = 'Continue'
        # Piped stdin = non-interactive: the run closes its overlay and releases the pad instead of
        # waiting for Ctrl+C after the flight.
        '' | & $DroneFly @runArgs 2>&1 | ForEach-Object { "$_" } |
            Where-Object { $_ -notmatch 'Warning|warnings\.warn' } | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
    } finally {
        Pop-Location
    }
    if ($capture) {
        Start-Sleep -Seconds 3
        Stop-Process -Id $capture.Id -ErrorAction SilentlyContinue
    }
    if ($code -ne 0 -and $code -ne 5) { throw "attempt $n failed to run (exit $code)" }
    $run = Get-ChildItem -Directory (Resolve-RepoPath $OutputRoot) |
        Where-Object { $before -notcontains $_.Name } | Sort-Object Name | Select-Object -Last 1
    if ($run) {
        Invoke-DroneFly harness flight-report $run.FullName --recording $name `
            --trackdata-root $TrackdataRoot | Out-Null
    }
    $results += [pscustomobject]@{
        Attempt = $n; Exit = $code; Run = $run.FullName; Recording = $name; Video = $video
    }
    if ($n -lt $Attempts) { Invoke-LiftoffUi reset }
}
$results | Format-List | Out-String | Write-Host
if ($Quit) {
    Invoke-LiftoffUi quit
    Stop-PadServer
}
