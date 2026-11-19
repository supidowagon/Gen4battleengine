# Managing-session watchdog for the Nuzline engine streams. Read-only except for
# removing a fast-tier lock whose owning process is dead (pass -FixLock).
# Streams come from a JSON file (default $NUZLINE_SCRATCH\manager\streams.json; format in README.md).
param([switch]$FixLock, [string]$Streams = '')

$repo = if ($env:NUZLINE_REPO) { $env:NUZLINE_REPO } else { Split-Path -Parent (git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir) }
$scratch = if ($env:NUZLINE_SCRATCH) { $env:NUZLINE_SCRATCH } else { Join-Path $repo '_scratch' }
if (-not $Streams) { $Streams = Join-Path $scratch 'manager\streams.json' }
$lock = Join-Path $scratch 'fast_tier.lock'
$flags = @()
$ages = @()
$now = Get-Date
"== watchdog $($now.ToString('ddd MM-dd HH:mm'))"

$list = @()
if (Test-Path $Streams) {
    $list = @(Get-Content $Streams -Raw | ConvertFrom-Json | ForEach-Object { $_ })
} else {
    $flags += "no stream list at $Streams"
}

# 1. Commits per stream
foreach ($s in $list) {
    $ts = git -C $repo log -1 --format=%ct $s.branch 2>$null
    if (-not $ts) { "{0}: branch {1} not found" -f $s.name, $s.branch; $flags += "$($s.name): branch $($s.branch) not found"; continue }
    $subj = git -C $repo log -1 --format='%h %s' $s.branch
    $age = [int](($now.ToUniversalTime() - [DateTimeOffset]::FromUnixTimeSeconds([int64]$ts).UtcDateTime).TotalMinutes)
    $wt = Join-Path $repo $s.worktree
    $dirty = if (Test-Path $wt) { "{0} uncommitted files" -f (git -C $wt status --short | Measure-Object).Count } else { "worktree $($s.worktree) missing" }
    $line = "{0}: last commit {1} min ago, {2} | {3}" -f $s.name, $age, $dirty, $subj
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
if ($list.Count -and -not $procs -and @($ages | Where-Object { $_ -gt 90 }).Count -eq $list.Count) {
    $flags += "no test or tool process is running and no stream has committed for 90+ min: streams may have stopped (usage limit or sign-in); check each session's last event and send a resume message"
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
