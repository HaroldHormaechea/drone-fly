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

.EXAMPLE
  .\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Attempts 100 -RecordEvery 50
#>
param(
    [int]$Attempts = 2,
    [int]$RecordEvery = 50,
    [string]$Checkpoint = 'training/07-proc-course-flight/model_readout.pt',
    [ValidateSet(12, 30, 70)][int]$ExtraDim = 30,
    [string]$Preset = 'liftoff',
    [string]$Config = 'artifacts/harness/live.yaml',
    [string]$MapName = 'infinity-loop',
    [string]$ModelName = '',
    [double]$MaxSeconds = 60,
    [string]$OutputRoot = 'artifacts/harness/liftoff/runs',
    [switch]$Quit
)
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')
if (-not $ModelName) { $ModelName = Split-Path (Split-Path $Checkpoint -Parent) -Leaf }

$results = @()
for ($n = 1; $n -le $Attempts; $n++) {
    $runArgs = @(
        'harness', 'run', '--harness', 'liftoff', '--preset', $Preset, '--config', $Config,
        '--extra-dim', "$ExtraDim", '--checkpoint', $Checkpoint, '--k1-path', 'artifacts/pruned/k1',
        '--output-root', $OutputRoot, '--duration', "$MaxSeconds"
    )
    $recording = ''
    if ($RecordEvery -gt 0 -and ($n % $RecordEvery) -eq 0) {
        $recording = "$MapName-model-$ModelName-attempt-$n"
        $runArgs += @('--save-recording', $recording)
    }
    Write-Host "=== attempt $n / $Attempts$(if ($recording) { " (recording: $recording)" })"
    $code = Invoke-DroneFly @runArgs
    # 0 = completed, 5 = fail-safe stop (the normal end of an attempt); anything else is an error.
    if ($code -ne 0 -and $code -ne 5) { throw "attempt $n failed to run (exit $code)" }
    $results += [pscustomobject]@{ Attempt = $n; Exit = $code; Recording = $recording }
    if ($n -lt $Attempts) { Invoke-LiftoffUi reset }
}
$results | Format-Table -AutoSize | Out-String | Write-Host
if ($Quit) {
    Invoke-LiftoffUi quit
    Stop-PadServer
}
