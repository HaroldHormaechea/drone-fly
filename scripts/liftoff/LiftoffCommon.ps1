# Shared helpers for the Liftoff episode scripts (dot-sourced, not run directly).
$ErrorActionPreference = 'Stop'
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:DroneFly = Join-Path $RepoRoot '.venv-cuda\Scripts\drone-fly.exe'
if (-not (Test-Path $DroneFly)) {
    throw "drone-fly not found at $DroneFly (create .venv-cuda with the harness extra first)"
}

function Invoke-DroneFly {
    # Runs drone-fly from the repo root, echoes its output (minus Python warnings) to the host and
    # returns only the exit code.
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Push-Location $RepoRoot
    try {
        $ErrorActionPreference = 'Continue'  # stderr lines are log output, not failures
        & $DroneFly @Arguments 2>&1 |
            ForEach-Object { "$_" } |
            Where-Object { $_ -notmatch 'Warning|warnings\.warn' } |
            ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Invoke-LiftoffUi {
    # One menu step (launch | last-session | reset | quit); throws when it fails.
    param([Parameter(Mandatory)][string]$Action)
    $code = Invoke-DroneFly harness ui $Action
    if ($code -ne 0) { throw "drone-fly harness ui $Action failed (exit $code)" }
}

function Stop-PadServer {
    # Disconnects the long-lived virtual pad. Liftoff re-arms when the pad disconnects, so call this
    # only after Liftoff has closed (or the drone was reset and paused).
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'harness pad --port' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
