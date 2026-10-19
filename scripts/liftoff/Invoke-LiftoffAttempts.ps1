<#
.SYNOPSIS
  Fly N AI attempts on the loaded Liftoff level: run -> (save recording) -> reset -> run ...

.DESCRIPTION
  Each attempt is one `drone-fly harness run` (it arms, flies until a fail-safe stop, disarms).
  Every -RecordEvery attempts, the run saves Liftoff's own flight recording of that attempt right
  after the stop (before Liftoff can auto-restart the course), named
  "<MapName>-model-<model>-attempt-<n>" under %USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings.
  Between attempts the level is reset (Esc -> Reset). Run folders go to artifacts/harness/liftoff/runs.
  Load the level first (Start-LiftoffSession.ps1); close it afterwards (Stop-LiftoffSession.ps1)
  or pass -Quit.

  UC69 (-Teacher): flies a UC68 teacher with ORACLE perception (exact gate poses from the
  catalogue track -Track + telemetry). Defaults then: -Attempts 10, -MaxSeconds 300, and the
  attempts are recorded into one session folder
  (artifacts/eval/<drone>/<run>/oracle/<track>/sessions/<UTC>-<sha8>, or -OutputRoot). There is
  no level loading: -Track must be the loaded Last Session, otherwise the start check fails and
  the attempt exits 6 (this script then stops). Exit 2 also covers "the teacher's run is still
  training" (-AllowConcurrentTraining for a labelled smoke run). After the last attempt it runs
  `drone-fly harness oracle-report` on the session.

.EXAMPLE
  .\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Attempts 100 -RecordEvery 50

.EXAMPLE
  .\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Teacher artifacts\training\liftoff-gremlin-micro\<run> -Track liftoff-arena-04 -Attempts 10 -RecordEvery 10 -Quit
#>
param(
    [int]$Attempts = 2,
    [int]$RecordEvery = 50,
    [string]$Checkpoint = 'training/07-proc-course-flight/model_readout.pt',
    [ValidateSet(12, 30, 70, 98)][int]$ExtraDim = 30,
    [string]$Preset = 'liftoff',
    [string]$Config = 'artifacts/harness/live.yaml',
    [string]$MapName = 'infinity-loop',
    [string]$ModelName = '',
    [double]$MaxSeconds = 60,
    [string]$OutputRoot = '',
    [string]$Teacher = '',
    [string]$Track = 'liftoff-arena-04',
    [string]$TrackdataRoot = '',
    [string]$ScenesRoot = '',
    [switch]$AllowConcurrentTraining,
    [switch]$Quit
)
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')
$teacherMode = [bool]$Teacher
if ($teacherMode) {
    if (-not $PSBoundParameters.ContainsKey('Attempts')) { $Attempts = 10 }
    if (-not $PSBoundParameters.ContainsKey('MaxSeconds')) { $MaxSeconds = 300 }
    if (-not $ModelName) { $ModelName = "teacher-oracle-$Track" }
    # One UTC stamp per session: every attempt lands in the same session folder.
    $sessionStamp = [DateTime]::UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'")
} else {
    if (-not $OutputRoot) { $OutputRoot = 'artifacts/harness/liftoff/runs' }
    if (-not $ModelName) { $ModelName = Split-Path (Split-Path $Checkpoint -Parent) -Leaf }
}

$results = @()
for ($n = 1; $n -le $Attempts; $n++) {
    if ($teacherMode) {
        $runArgs = @(
            'harness', 'run', '--harness', 'liftoff', '--preset', $Preset,
            '--teacher', $Teacher, '--track', $Track, '--perception', 'oracle',
            '--duration', "$MaxSeconds"
        )
        # live.yaml is a K1-stage overlay; only pass a config the caller asked for.
        if ($PSBoundParameters.ContainsKey('Config')) { $runArgs += @('--config', $Config) }
        if ($OutputRoot) { $runArgs += @('--session', $OutputRoot) }
        else { $runArgs += @('--session-stamp', $sessionStamp) }
        if ($TrackdataRoot) { $runArgs += @('--trackdata-root', $TrackdataRoot) }
        if ($ScenesRoot) { $runArgs += @('--scenes-root', $ScenesRoot) }
        if ($AllowConcurrentTraining) { $runArgs += '--allow-concurrent-training' }
    } else {
        $runArgs = @(
            'harness', 'run', '--harness', 'liftoff', '--preset', $Preset, '--config', $Config,
            '--extra-dim', "$ExtraDim", '--checkpoint', $Checkpoint, '--k1-path', 'artifacts/pruned/k1',
            '--output-root', $OutputRoot, '--duration', "$MaxSeconds"
        )
    }
    $recording = ''
    if ($RecordEvery -gt 0 -and ($n % $RecordEvery) -eq 0) {
        $recording = "$MapName-model-$ModelName-attempt-$n"
        $runArgs += @('--save-recording', $recording)
    }
    Write-Host "=== attempt $n / $Attempts$(if ($recording) { " (recording: $recording)" })"
    $code = Invoke-DroneFly @runArgs
    # 0 = completed, 5 = fail-safe / episode end (the normal end of an attempt); 6 = the drone is
    # not at the track's start (UC69 oracle: wrong level loaded); anything else is an error.
    if ($code -eq 6) {
        throw "attempt ${n}: the drone is not at ${Track}'s start (exit 6): load it as the Last Session and reset"
    }
    if ($code -ne 0 -and $code -ne 5) { throw "attempt $n failed to run (exit $code)" }
    $results += [pscustomobject]@{ Attempt = $n; Exit = $code; Recording = $recording }
    if ($n -lt $Attempts) {
        # A teacher run exits 0 when it finished the course: Liftoff then shows the post-race
        # screen, where Esc -> Reset does not respawn; "Retry" restarts the race at the spawn.
        if ($teacherMode -and $code -eq 0) { Invoke-LiftoffUi retry } else { Invoke-LiftoffUi reset }
    }
}
$results | Format-Table -AutoSize | Out-String | Write-Host
if ($Quit) {
    Invoke-LiftoffUi quit
    Stop-PadServer
}
if ($teacherMode) {
    if ($OutputRoot) {
        $sessions = @((Resolve-Path $OutputRoot).Path)
    } else {
        # A checkpoint that changed mid-session (new sha) started a second session folder.
        $pattern = Join-Path $RepoRoot "artifacts\eval\*\*\oracle\$Track\sessions\$sessionStamp-*"
        $sessions = @(Get-ChildItem -Directory $pattern -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
    }
    if (-not $sessions) { throw "no session folder found for stamp $sessionStamp" }
    foreach ($session in $sessions) {
        Write-Host "=== oracle report: $session"
        $code = Invoke-DroneFly harness oracle-report --session $session
        if ($code -ne 0) { throw "oracle-report failed for $session (exit $code)" }
    }
}
