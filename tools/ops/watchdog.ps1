# Managing-session watchdog for the Nuzline engine streams. Read-only except for
# removing a fast-tier lock whose owning process is dead (pass -FixLock).
param([switch]$FixLock)

$repo = if ($env:NUZLINE_REPO) { $env:NUZLINE_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$scratch = if ($env:NUZLINE_SCRATCH) { $env:NUZLINE_SCRATCH } else { Join-Path $repo '..\_scratch' }
$lock = Join-Path $scratch 'fast_tier.lock'
$streams = @(
    @{ name = 'E1'; branch = 'claude/e1-batch-c'; wt = '..\_scratch\e1\cwt' },
    @{ name = 'E2'; branch = 'claude/engine-e2-mechanics-e737dd'; wt = '.claude\worktrees\engine-e2-mechanics-e737dd' }
)
$flags = @()
$ages = @()
$now = Get-Date
"== watchdog $($now.ToString('ddd MM-dd HH:mm'))"

# 1. Commits per stream
foreach ($s in $streams) {
    $ts = git -C $repo log -1 --format=%ct $s.branch
    $subj = git -C $repo log -1 --format='%h %s' $s.branch
    $age = [int](($now.ToUniversalTime() - [DateTimeOffset]::FromUnixTimeSeconds([int64]$ts).UtcDateTime).TotalMinutes)
    $dirty = (git -C (Join-Path $repo $s.wt) status --short | Measure-Object).Count
    $line = "{0}: last commit {1} min ago, {2} uncommitted files | {3}" -f $s.name, $age, $dirty, $subj
    if ($line.Length -gt 150) { $line = $line.Substring(0, 150) }
    $line
    $ages += $age
    if ($age -gt 120) { $flags += "$($s.name) has not committed for $age min" }
}

# 2. Python test processes
$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.WorkingSetSize -gt 200MB }
$pytestRunning = $false
foreach ($p in $procs) {
    $gb = [math]::Round($p.WorkingSetSize / 1GB, 1)
    $mins = [int]($now - $p.CreationDate).TotalMinutes
    $cmd = $p.CommandLine
    if ($cmd -match 'pytest') { $pytestRunning = $true }
    if ($cmd.Length -gt 110) { $cmd = $cmd.Substring($cmd.Length - 110) }
    "proc {0}: {1} GB, {2} min | {3}" -f $p.ProcessId, $gb, $mins, $cmd
    if ($gb -gt 12) { $flags += "process $($p.ProcessId) is at $gb GB" }
    if ($mins -gt 150) { $flags += "process $($p.ProcessId) has run for $mins min" }
}

# 2b. Everything stopped? (usage limit or sign-in expiry stops every session at once)
if (-not $procs -and ($ages | Where-Object { $_ -gt 40 }).Count -eq $streams.Count) {
    $flags += "no test or tool process is running and no stream has committed for 40+ min: streams may have stopped (usage limit or sign-in); check each session's last event and send a resume message"
}

# 3. Memory
$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
"free RAM: $free GB"
if ($free -lt 3) { $flags += "only $free GB RAM free" }

# 4. Fast-tier lock
if (Test-Path $lock) {
    $text = (Get-Content $lock -Raw).Trim()
    $age = [int]($now - (Get-Item $lock).LastWriteTime).TotalMinutes
    "lock: held $age min | $($text -replace '\s+', ' ')"
    $lockPid = $null
    if ($text -match 'pid\s*[=:]\s*(\d+)') { $lockPid = [int]$Matches[1] }
    $alive = $false
    if ($lockPid) { $alive = [bool](Get-Process -Id $lockPid -ErrorAction SilentlyContinue) }
    if ($lockPid -and -not $alive) {
        $flags += "lock owner pid $lockPid is dead"
        if ($FixLock) { Remove-Item $lock -Confirm:$false; "lock removed (dead owner)" }
    } elseif (-not $lockPid -and -not $pytestRunning) {
        $flags += "lock held with no pid recorded and no pytest process running"
    } elseif ($age -gt 150) {
        $flags += "lock held for $age min"
    }
} else {
    "lock: free"
}

if ($flags.Count) { "FLAGS:"; $flags | ForEach-Object { " - $_" } } else { "FLAGS: none" }
