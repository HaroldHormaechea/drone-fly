<#
.SYNOPSIS
  Quit Liftoff completely (Esc -> Quit -> Confirm -> Quit -> Yes), then disconnect the virtual pad.
#>
. (Join-Path $PSScriptRoot 'LiftoffCommon.ps1')
Invoke-LiftoffUi quit
Stop-PadServer
Write-Host 'Liftoff closed, virtual pad disconnected'
