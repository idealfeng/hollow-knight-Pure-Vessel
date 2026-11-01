param(
    [ValidateCount(1, 4)][int[]]$Ports = @(9003, 9004),
    [string]$SavePath = (Join-Path $env:USERPROFILE 'AppData\LocalLow\Team Cherry\Hollow Knight\user1.dat'),
    [switch]$NoRestart,
    [switch]$CheckOnly
)

# Run on the game host. The converter changes only nail damage and equipped
# charms. Existing game paths/arguments, abilities and save progress are kept.
$ErrorActionPreference = 'Stop'
$taskConverter = Join-Path $PSScriptRoot 'configure_training_loadout.ps1'
if (-not (Test-Path -LiteralPath $taskConverter -PathType Leaf)) { throw "Missing converter: $taskConverter" }
$taskSave = (Resolve-Path -LiteralPath $SavePath -ErrorAction Stop).Path
if (($Ports | Sort-Object -Unique).Count -ne $Ports.Count) { throw 'Ports must be unique.' }
if (@($Ports | Where-Object { $_ -lt 1024 -or $_ -gt 65535 }).Count) { throw 'Ports must be in 1024..65535.' }

function Get-LocalGames {
    @(Get-CimInstance Win32_Process -Filter "Name = 'hollow_knight.exe'")
}

function Get-GamePort($Game) {
    if ($Game.CommandLine -match '--hollowrl-port=(\d+)(?:\s|$)') { return [int]$Matches[1] }
    return -1
}

$taskAllGames = @(Get-LocalGames)
$taskGames = @()
if ($NoRestart) {
    if ($taskAllGames.Count) { throw 'Close all local Hollow Knight windows before using -NoRestart.' }
}
else {
    foreach ($taskPort in $Ports) {
        $taskMatches = @($taskAllGames | Where-Object { (Get-GamePort $_) -eq $taskPort })
        if ($taskMatches.Count -ne 1) {
            throw "Expected one running game on port $taskPort. Leave the original windows running, or close all games and use -NoRestart. No save changes made."
        }
        $taskGame = $taskMatches[0]
        if (-not $taskGame.ExecutablePath) { throw "Cannot read executable path for port $taskPort." }
        $taskPrefix = '"' + $taskGame.ExecutablePath + '"'
        if (-not $taskGame.CommandLine.StartsWith($taskPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unrecognized game command line for port $taskPort; no save changes made."
        }
        $taskGames += [pscustomobject]@{
            port = $taskPort; pid = [int]$taskGame.ProcessId
            executable = $taskGame.ExecutablePath
            arguments = $taskGame.CommandLine.Substring($taskPrefix.Length).Trim()
        }
    }
    $taskOtherGames = @($taskAllGames | Where-Object { $_.ProcessId -notin $taskGames.pid })
    if ($taskOtherGames.Count) { throw 'Another local game shares the save directory. Close it before updating.' }
}

if ($CheckOnly) {
    [pscustomobject]@{
        check_only = $true; save = $taskSave; ports = $Ports
        games_to_restart = @($taskGames); nail_damage = 21; combat_charms = @()
        mod_changed = $false
    } | ConvertTo-Json -Depth 5
    return
}

$taskSaveDirectory = Split-Path -Parent $taskSave
$taskBackupBase = [IO.Path]::GetFullPath((Join-Path $taskSaveDirectory 'HollowRL_loadout_backups'))
$taskBackup = [IO.Path]::GetFullPath((Join-Path $taskBackupBase (Get-Date -Format 'yyyyMMdd_HHmmss_fff')))
if (-not $taskBackup.StartsWith($taskBackupBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Backup path is outside the intended directory.'
}
New-Item -ItemType Directory -Path $taskBackup -Force | Out-Null
$taskOriginal = Join-Path $taskBackup 'user1.before.dat'
$taskPrepared = Join-Path $taskBackup 'user1.21.dat'
$taskGames | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskBackup 'game_arguments.json') -Encoding UTF8
$taskClosedGames = @()
$taskInstallStarted = $false
$taskOriginalHash = $null
$taskAllowRestart = $true

function Restart-ClosedGames {
    foreach ($taskRestart in $taskClosedGames) {
        $taskStillRunning = Get-Process -Id $taskRestart.pid -ErrorAction SilentlyContinue
        if ($taskStillRunning) { continue }
        Start-Process -FilePath $taskRestart.executable -ArgumentList $taskRestart.arguments `
            -WorkingDirectory (Split-Path -Parent $taskRestart.executable) -WindowStyle Normal | Out-Null
        Write-Host "Restarted port $($taskRestart.port) with original arguments."
    }
}

try {
    foreach ($taskGame in $taskGames) {
        $taskProcess = Get-Process -Id $taskGame.pid -ErrorAction Stop
        [void]$taskProcess.CloseMainWindow()
        if (-not $taskProcess.WaitForExit(15000)) {
            throw "Port $($taskGame.port) did not close. Close it manually, then rerun; no save was changed."
        }
        $taskClosedGames += $taskGame
    }
    if (@(Get-LocalGames).Count) { throw 'A game is still running; refusing to replace the shared save.' }
    Copy-Item -LiteralPath $taskSave -Destination $taskOriginal -ErrorAction Stop
    $taskOriginalHash = (Get-FileHash -LiteralPath $taskOriginal -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $taskSave -Algorithm SHA256).Hash -ne $taskOriginalHash) {
        throw 'Save changed while being backed up; no replacement made.'
    }
    $taskConversionText = & powershell -NoProfile -ExecutionPolicy Bypass -File $taskConverter `
        -SavePath $taskOriginal -OutputPath $taskPrepared
    if ($LASTEXITCODE -ne 0) { throw 'Loadout conversion failed; original save retained.' }
    $taskConversion = ($taskConversionText -join "`n") | ConvertFrom-Json
    if ($taskConversion.nail_damage -ne 21 -or @($taskConversion.combat_charms).Count -ne 0) {
        throw 'Converted loadout did not pass validation.'
    }
    if ((Get-FileHash -LiteralPath $taskSave -Algorithm SHA256).Hash -ne $taskOriginalHash) {
        throw 'The live save changed after backup; refusing to overwrite it.'
    }
    $taskInstallStarted = $true
    Copy-Item -LiteralPath $taskPrepared -Destination $taskSave -Force -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $taskSave -Algorithm SHA256).Hash -ne $taskConversion.sha256) {
        throw 'Installed save checksum mismatch.'
    }
    $taskConversion | Add-Member -NotePropertyName installed_save -NotePropertyValue $taskSave
    $taskConversion | Add-Member -NotePropertyName original_sha256 -NotePropertyValue $taskOriginalHash
    $taskConversion | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBackup 'verification.json') -Encoding UTF8
    Write-Host "SUCCESS: nail damage 21, no combat charms. Backup: $taskBackup"
}
catch {
    $taskFailure = $_
    if ($taskInstallStarted) {
        try {
            Copy-Item -LiteralPath $taskOriginal -Destination $taskSave -Force -ErrorAction Stop
            if ((Get-FileHash -LiteralPath $taskSave -Algorithm SHA256).Hash -ne $taskOriginalHash) {
                throw 'Rollback checksum mismatch.'
            }
            Write-Host 'Original save restored and verified.'
        }
        catch {
            $taskAllowRestart = $false
            throw "Rollback failed. Games will remain closed. Restore $taskOriginal manually. Details: $_"
        }
    }
    throw $taskFailure
}
finally {
    if ($taskAllowRestart -and -not $NoRestart) { Restart-ClosedGames }
}
Write-Host 'Load save slot 1 in each window, then stand at the Traitor Lord statue. Do not enter the fight manually.'
