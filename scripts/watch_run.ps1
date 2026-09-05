<#
.SYNOPSIS
    Live-view the current TENURE training run's progress bar from PowerShell.

.DESCRIPTION
    Training (tenure.train) prints a file-friendly progress bar (one line per
    --log-every updates: J_H / MAE / entropy / ETA). When launched as a Claude
    background task, that stream is captured to a *.output file on disk. This
    script auto-discovers the newest such file that contains a progress bar and
    tails it, so you never need to know the task id.

.EXAMPLE
    .\scripts\watch_run.ps1            # live-follow the active run (Ctrl-C to stop)
    .\scripts\watch_run.ps1 -Once      # print the current tail once and exit
    .\scripts\watch_run.ps1 -Tail 40   # show more history
#>
param(
    [int]$Tail = 25,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$slug = 'c--Users-xiele-Documents-Rex-Technologies-leroiai-offSim'
$taskGlob = Join-Path $env:TEMP "claude\$slug\*\tasks\*.output"

$live = Get-ChildItem $taskGlob -ErrorAction SilentlyContinue |
    Where-Object { Select-String -Path $_.FullName -Pattern 'J_H~' -Quiet } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $live) {
    Write-Host "No active training log found (no *.output contains a progress bar)." -ForegroundColor Yellow
    Write-Host "The run may not have printed its first bar line yet, or it has finished." -ForegroundColor Yellow
    exit 1
}

$age = [int]((Get-Date) - $live.LastWriteTime).TotalSeconds
Write-Host "Watching: $($live.Name)  (last write ${age}s ago)" -ForegroundColor Cyan
if ($age -gt 180) {
    Write-Host "NOTE: no new output in ${age}s -- this run may be finished or stalled." -ForegroundColor Yellow
}
Write-Host ("-" * 72) -ForegroundColor DarkGray

if ($Once) {
    Get-Content $live.FullName -Tail $Tail
} else {
    Write-Host "(live -- press Ctrl-C to stop)" -ForegroundColor DarkGray
    Get-Content $live.FullName -Wait -Tail $Tail
}
