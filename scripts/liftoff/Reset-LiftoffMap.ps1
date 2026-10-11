<#
.SYNOPSIS
  Restart the current Liftoff level (Esc -> Reset): drone back at the start, full battery.
#>
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')
Invoke-LiftoffUi reset
Write-Host 'Liftoff: level reset'
