# Flag-only watchdog loop for the managing session. Runs watchdog.ps1 every
# $EveryMin minutes and EXITS (printing the watchdog output) only when something
# needs the manager: an unexplained flag, or the fast-tier lock going away.
# Flags that a live, progressing fast tier explains are ignored. While
# -HoldUntil is in the future, "has not committed" and "may have stopped" are ignored.
param([int]$EveryMin = 10, [string]$HoldUntil = '')

$repo = if ($env:NUZLINE_REPO) { $env:NUZLINE_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$scratch = if ($env:NUZLINE_SCRATCH) { $env:NUZLINE_SCRATCH } else { Join-Path $repo '..\_scratch' }
$dog = Join-Path $PSScriptRoot 'watchdog.ps1'
$lock = Join-Path $scratch 'fast_tier.lock'
$logDirs = (Join-Path $scratch 'e1\gard'), (Join-Path $scratch 'e2')
$hold = if ($HoldUntil) { [datetime]$HoldUntil } else { [datetime]::MinValue }
$hadLock = Test-Path $lock

while ($true) {
    $out = & $dog -FixLock
    $flags = @($out | Where-Object { $_ -match '^ - ' })
    $hasLock = Test-Path $lock

    # A fast tier is "progressing" when the lock exists and some ft_*.log was written in the last 25 min.
    $newest = Get-ChildItem $logDirs -Filter 'ft_*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $progressing = $hasLock -and $newest -and (((Get-Date) - $newest.LastWriteTime).TotalMinutes -lt 150)

    $real = @()
    foreach ($f in $flags) {
        if ($progressing -and ($f -match 'has run for|lock held for|process \d+ is at')) { continue }
        if ((Get-Date) -lt $hold -and ($f -match 'has not committed|may have stopped')) { continue }
        $real += $f
    }
    $reason = $null
    if ($real.Count) { $reason = 'FLAG' }
    elseif ($hadLock -and -not $hasLock) { $reason = 'FAST TIER ENDED (lock gone)' }
    if ($reason) {
        "WATCH EXIT: $reason"
        if ($newest) { "newest tier log: $($newest.FullName)"; Get-Content $newest.FullName -Tail 3 }
        $out
        exit 0
    }
    $hadLock = $hasLock
    Start-Sleep -Seconds ($EveryMin * 60)
}
