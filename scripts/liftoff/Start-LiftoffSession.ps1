<#
.SYNOPSIS
  Launch Liftoff (if needed) and load the last session (Single Player -> Quick Play -> Last Session).
.EXAMPLE
  .\scripts\liftoff\Start-LiftoffSession.ps1
#>
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')
Invoke-LiftoffUi launch
Invoke-LiftoffUi last-session
Write-Host 'Liftoff: last session loaded'
