param(
    [string]$GameDir = 'D:\Program Files\Steam\steamapps\common\Hollow Knight',
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001
)

$ErrorActionPreference = 'Stop'
$gameRoot = (Resolve-Path -LiteralPath $GameDir).Path
$gameExe = Join-Path $gameRoot 'hollow_knight.exe'
$builtDll = Join-Path $PSScriptRoot 'unity_mod\HollowRL\bin\Release\HollowRL.dll'
$targetDll = Join-Path $gameRoot 'BepInEx\plugins\HollowRL\HollowRL.dll'
if (-not (Test-Path -LiteralPath $builtDll)) { throw "Build the Mod first: $builtDll" }
if (-not (Test-Path -LiteralPath $gameExe)) { throw "Game executable missing: $gameExe" }
$activeTrainer = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'main_multi\.py' })
if ($activeTrainer.Count -gt 0) {
    throw 'A PPO trainer is still running. Stop it normally and keep its checkpoint before preparing human recording.'
}

# Only restart the selected installation. Other instances remain available.
$targetProcesses = @(Get-Process hollow_knight -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $gameExe })
$backupRoot = Join-Path $PSScriptRoot 'mod_backups'
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
if (Test-Path -LiteralPath $targetDll) {
    $backupFile = Join-Path $backupRoot ('HollowRL_before_human_' + (Get-Date -Format 'yyyyMMdd_HHmmssfff') + '.dll')
    Copy-Item -LiteralPath $targetDll -Destination $backupFile
    Write-Host "Previous Mod backed up to $backupFile"
}
foreach ($gameProcess in $targetProcesses) {
    # The user has requested a Mod update and recording setup. Verify the
    # executable path again immediately before stopping this game process.
    $currentProcess = Get-Process -Id $gameProcess.Id -ErrorAction Stop
    if ($currentProcess.Path -ne $gameExe) { throw 'Game process identity changed.' }
    Stop-Process -Id $currentProcess.Id
    if (-not $currentProcess.WaitForExit(10000)) { throw 'Game did not exit.' }
}
Copy-Item -LiteralPath $builtDll -Destination $targetDll -Force
if ((Get-FileHash -LiteralPath $builtDll).Hash -ne (Get-FileHash -LiteralPath $targetDll).Hash) {
    throw 'Installed Mod hash does not match the build.'
}
& (Join-Path $PSScriptRoot 'launch_multi_games.ps1') -GameDirs @($gameRoot) -Ports @($Port) `
    -Fps 60 -GameSpeed 1 -Width 1280 -Height 720
if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw 'Game launcher failed.' }
Write-Host "Human recording game is ready on port $Port. Load training save slot 1."
