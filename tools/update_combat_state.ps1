param(
    [ValidateCount(1, 4)][int[]]$Ports = @(9003, 9004),
    [string]$DllPath = (Join-Path $PSScriptRoot 'HollowRL.dll'),
    [string]$ExpectedSha256 = '',
    [switch]$CheckOnly
)

# Run on the game host after stopping its controller. Only the Mod is updated;
# no save, damage, charm, ability or model files are read or changed.
$ErrorActionPreference = 'Stop'
if (($Ports | Sort-Object -Unique).Count -ne $Ports.Count) { throw 'Ports must be unique.' }
if (@($Ports | Where-Object { $_ -lt 1024 -or $_ -gt 65535 }).Count) { throw 'Ports must be in 1024..65535.' }
$taskSource = (Resolve-Path -LiteralPath $DllPath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) { throw 'The source DLL is not a file.' }
$taskAssembly = [Reflection.AssemblyName]::GetAssemblyName($taskSource)
if ($taskAssembly.Name -ne 'HollowRL') { throw 'The source is not a HollowRL assembly.' }
$taskSourceHash = (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
$taskManifestPath = Join-Path $PSScriptRoot 'manifest.json'
if (Test-Path -LiteralPath $taskManifestPath -PathType Leaf) {
    $taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskManifest.name -ne 'combat_state_b_machine' -or -not $taskManifest.files.'HollowRL.dll') {
        throw 'The package manifest is missing its HollowRL DLL checksum.'
    }
    if ($taskSourceHash -ne $taskManifest.files.'HollowRL.dll') { throw 'The source DLL does not match the package manifest.' }
}
if ($ExpectedSha256 -and $taskSourceHash -ne $ExpectedSha256) { throw 'The source DLL does not match -ExpectedSha256.' }

function Get-LocalGames {
    @(Get-CimInstance Win32_Process -Filter "Name = 'hollow_knight.exe'")
}

function Get-GamePort($Game) {
    if ($Game.CommandLine -match '--hollowrl-port=(\d+)(?:\s|$)') { return [int]$Matches[1] }
    return -1
}

function Assert-NoReparsePoint([string]$Path, [string]$Boundary) {
    $taskInspect = Get-Item -LiteralPath $Path -ErrorAction Stop
    while ($null -ne $taskInspect) {
        if ($taskInspect.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing a reparse-point installation path: $($taskInspect.FullName)"
        }
        if ($taskInspect.FullName.TrimEnd('\') -eq $Boundary.TrimEnd('\')) { return }
        $taskInspect = if ($taskInspect -is [IO.DirectoryInfo]) { $taskInspect.Parent } else { $taskInspect.Directory }
    }
    throw "Installation path is outside the verified game directory: $Path"
}

function Assert-OriginalProcess($Game) {
    $taskCurrent = @(Get-CimInstance Win32_Process -Filter "ProcessId = $($Game.pid)")
    if ($taskCurrent.Count -ne 1 -or $taskCurrent[0].Name -ne 'hollow_knight.exe' -or
        $taskCurrent[0].ExecutablePath -ne $Game.executable -or
        $taskCurrent[0].CommandLine -ne $Game.command_line -or
        $taskCurrent[0].CreationDate.ToUniversalTime().ToString('o') -ne $Game.created_at) {
        throw "Port $($Game.port) changed processes. No further game will be closed."
    }
}

$taskAllGames = @(Get-LocalGames)
$taskGames = @()
foreach ($taskPort in $Ports) {
    $taskMatches = @($taskAllGames | Where-Object { (Get-GamePort $_) -eq $taskPort })
    if ($taskMatches.Count -ne 1) { throw "Expected exactly one running original game on port $taskPort. No games were changed." }
    $taskGame = $taskMatches[0]
    if (-not $taskGame.ExecutablePath -or -not $taskGame.CommandLine -or -not $taskGame.CreationDate) {
        throw "Cannot read original process metadata for port $taskPort."
    }
    $taskExecutable = (Resolve-Path -LiteralPath $taskGame.ExecutablePath).Path
    if ([IO.Path]::GetFileName($taskExecutable) -ne 'hollow_knight.exe') { throw 'Unexpected game executable.' }
    $taskPrefix = '"' + $taskExecutable + '"'
    if (-not $taskGame.CommandLine.StartsWith($taskPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unrecognized original command line for port $taskPort. No games were changed."
    }
    $taskGameDirectory = Split-Path -Parent $taskExecutable
    $taskPlugins = [IO.Path]::GetFullPath((Join-Path $taskGameDirectory 'BepInEx\plugins'))
    if (-not (Test-Path -LiteralPath $taskPlugins -PathType Container)) { throw "No BepInEx plugins directory for port $taskPort." }
    Assert-NoReparsePoint $taskPlugins $taskGameDirectory
    $taskInstalled = @(Get-ChildItem -LiteralPath $taskPlugins -Filter 'HollowRL.dll' -File -Recurse)
    if ($taskInstalled.Count -ne 1) { throw "Expected exactly one installed HollowRL.dll under $taskPlugins; found $($taskInstalled.Count)." }
    $taskTarget = (Resolve-Path -LiteralPath $taskInstalled[0].FullName).Path
    if (-not $taskTarget.StartsWith($taskPlugins + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Plugin target escaped its verified directory.' }
    Assert-NoReparsePoint $taskTarget $taskGameDirectory
    if ($taskTarget -eq $taskSource) { throw 'The package source must be separate from the installed DLL.' }
    $taskGames += [pscustomobject]@{
        port = $taskPort; pid = [int]$taskGame.ProcessId
        executable = $taskExecutable; game_directory = $taskGameDirectory
        command_line = $taskGame.CommandLine
        arguments = $taskGame.CommandLine.Substring($taskPrefix.Length).Trim()
        created_at = $taskGame.CreationDate.ToUniversalTime().ToString('o')
        target_dll = $taskTarget
        original_sha256 = (Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash
    }
}

# A third game in the same installation would keep the replaced DLL loaded.
# Other installations are left alone and never become stop/restart targets.
foreach ($taskOther in @($taskAllGames | Where-Object { $_.ProcessId -notin $taskGames.pid })) {
    if (-not $taskOther.ExecutablePath) { throw 'Cannot establish whether another game shares the target installation.' }
    $taskOtherDirectory = Split-Path -Parent $taskOther.ExecutablePath
    if ($taskOtherDirectory -in $taskGames.game_directory) {
        throw 'An additional game shares a target installation. Close that window before updating.'
    }
}

$taskPlan = [ordered]@{
    check_only = [bool]$CheckOnly; source_dll = $taskSource; source_sha256 = $taskSourceHash
    ports = $Ports; games_to_restart = @($taskGames); save_changed = $false
}
if ($CheckOnly) { $taskPlan | ConvertTo-Json -Depth 6; return }

$taskBackupBase = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'combat_state_backups'))
$taskBackup = [IO.Path]::GetFullPath((Join-Path $taskBackupBase ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8))))
if (-not $taskBackup.StartsWith($taskBackupBase + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid backup path.' }
New-Item -ItemType Directory -Path $taskBackup -Force | Out-Null
$taskPlan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskBackup 'installation_plan.json') -Encoding UTF8
$taskBackups = @{}
foreach ($taskGame in $taskGames) {
    if (-not $taskBackups.ContainsKey($taskGame.target_dll)) {
        $taskOriginal = Join-Path $taskBackup ('port_' + $taskGame.port + '.original.HollowRL.dll')
        Copy-Item -LiteralPath $taskGame.target_dll -Destination $taskOriginal -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $taskOriginal -Algorithm SHA256).Hash -ne $taskGame.original_sha256) { throw 'Original DLL backup checksum mismatch.' }
        $taskBackups[$taskGame.target_dll] = $taskOriginal
    }
}

$taskClosedGames = @()
$taskInstallTargets = @()
$taskAllowRestart = $true
$taskFailure = $null

function Restart-ClosedGames {
    foreach ($taskRestart in $taskClosedGames) {
        if (@(Get-LocalGames | Where-Object { (Get-GamePort $_) -eq $taskRestart.port }).Count) {
            Write-Warning "Port $($taskRestart.port) already has a running game; no duplicate was started."
            continue
        }
        $taskInstalledHash = (Get-FileHash -LiteralPath $taskRestart.target_dll -Algorithm SHA256).Hash
        $taskExpectedInstalledHash = if ($null -eq $taskFailure) { $taskSourceHash } else { $taskRestart.original_sha256 }
        if ($taskInstalledHash -ne $taskExpectedInstalledHash) { throw "Refusing to restart port $($taskRestart.port) with an unverified DLL." }
        Start-Process -FilePath $taskRestart.executable -ArgumentList $taskRestart.arguments `
            -WorkingDirectory $taskRestart.game_directory -WindowStyle Hidden | Out-Null
        Write-Host "Restarted port $($taskRestart.port) with its original executable and arguments."
    }
}

try {
    foreach ($taskGame in $taskGames) {
        Assert-OriginalProcess $taskGame
        $taskProcess = Get-Process -Id $taskGame.pid -ErrorAction Stop
        [void]$taskProcess.CloseMainWindow()
        if (-not $taskProcess.WaitForExit(15000)) {
            throw "Port $($taskGame.port) did not close. Close it manually and rerun; no DLL was replaced."
        }
        $taskClosedGames += $taskGame
    }
    $taskRunning = @(Get-LocalGames)
    foreach ($taskGame in $taskGames) {
        if (@($taskRunning | Where-Object { (Get-GamePort $_) -eq $taskGame.port -or
                    ($_.ExecutablePath -and (Split-Path -Parent $_.ExecutablePath) -eq $taskGame.game_directory) }).Count) {
            throw 'A target installation reopened while updating; refusing to replace its loaded DLL.'
        }
        if ((Get-FileHash -LiteralPath $taskGame.target_dll -Algorithm SHA256).Hash -ne $taskGame.original_sha256) {
            throw 'An installed DLL changed after backup; refusing to overwrite it.'
        }
    }
    if ((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash -ne $taskSourceHash) { throw 'Package DLL changed after validation.' }
    foreach ($taskTarget in @($taskGames.target_dll | Sort-Object -Unique)) {
        $taskInstallTargets += $taskTarget
        Copy-Item -LiteralPath $taskSource -Destination $taskTarget -Force -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne $taskSourceHash) { throw 'Installed DLL checksum mismatch.' }
    }
    [ordered]@{success = $true; installed_sha256 = $taskSourceHash; ports = $Ports; save_changed = $false} |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskBackup 'result.json') -Encoding UTF8
}
catch {
    $taskFailure = $_
    $taskRollbackErrors = @()
    foreach ($taskTarget in $taskInstallTargets) {
        try {
            $taskOriginal = $taskBackups[$taskTarget]
            Copy-Item -LiteralPath $taskOriginal -Destination $taskTarget -Force -ErrorAction Stop
            if ((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $taskOriginal -Algorithm SHA256).Hash) { throw 'Rollback checksum mismatch.' }
        }
        catch { $taskRollbackErrors += "$taskTarget : $_" }
    }
    [ordered]@{success = $false; error = "$taskFailure"; rollback_errors = $taskRollbackErrors; save_changed = $false} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBackup 'result.json') -Encoding UTF8
    if ($taskRollbackErrors.Count) {
        $taskAllowRestart = $false
        throw "Rollback failed; games will remain closed. Restore DLLs from $taskBackup. $($taskRollbackErrors -join ' | ')"
    }
    throw $taskFailure
}
finally {
    if ($taskAllowRestart) { Restart-ClosedGames }
}
Write-Host "SUCCESS: combat-state Mod installed and verified. Backups: $taskBackup"
Write-Host 'Load save slot 1 in each window, then stand at the Traitor Lord statue. Do not enter the fight or choose difficulty.'
Write-Host 'No save files, damage values, charms, abilities or model weights were changed.'
