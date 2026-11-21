# Flag-only watchdog loop for the managing session. Runs watchdog.ps1 every
# $EveryMin minutes and EXITS (printing the watchdog output) only when something
# needs the manager: an unexplained flag, or the fast-tier lock going away.
# Flags that a live, progressing fast tier explains are ignored. While
# -HoldUntil is in the future, "has not committed" and "may have stopped" are ignored.
# -Streams: the stream list (default <scratch>\manager\streams.json; format in docs/process.md
# section 6). -Lock and -MinFreeGB pass through to watchdog.ps1 (defaults there).
param([int]$EveryMin = 10, [string]$HoldUntil = '', [string]$Streams = '', [string]$Lock = '', [double]$MinFreeGB = 1)

$repo = if ($env:NUZLINE_REPO) { $env:NUZLINE_REPO } else { Split-Path -Parent (git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir) }
$scratch = if ($env:NUZLINE_SCRATCH) { $env:NUZLINE_SCRATCH } else { Join-Path $repo '_scratch' }
if (-not $Streams) { $Streams = Join-Path $scratch 'manager\streams.json' }
$dog = Join-Path $PSScriptRoot 'watchdog.ps1'
$lock = if ($Lock) { $Lock } else { Join-Path $scratch 'fast_tier.lock' }
$hold = if ($HoldUntil) { [datetime]$HoldUntil } else { [datetime]::MinValue }
$hadLock = Test-Path $lock

while ($true) {
    $out = & $dog -FixLock -Streams $Streams -Lock $lock -MinFreeGB $MinFreeGB
    $flags = @($out | Where-Object { $_ -match '^ - ' })
    $hasLock = Test-Path $lock

    # Fast-tier log folders: each stream's optional "logs" entry (relative to the repo), re-read every round.
    $logDirs = @()
    if (Test-Path $Streams) {
        $logDirs = @(Get-Content $Streams -Raw | ConvertFrom-Json | ForEach-Object { $_ } | Where-Object { $_.logs } |
            ForEach-Object { Join-Path $repo $_.logs } | Where-Object { Test-Path $_ })
    }
    # A fast tier is "progressing" when the lock exists and some ft_*.log was written in the last 150 min.
    $newest = $null
    if ($logDirs.Count) {
        $newest = Get-ChildItem $logDirs -Filter 'ft_*.log' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
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
