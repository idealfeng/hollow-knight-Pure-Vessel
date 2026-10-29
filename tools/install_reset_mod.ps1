param(
    [Parameter(Mandatory = $true)]
    [ValidateCount(1, 4)]
    [string[]]$GameDirs
)

$ErrorActionPreference = 'Stop'
$sourceDll = Join-Path $PSScriptRoot 'HollowRL.dll'
$checksumFile = Join-Path $PSScriptRoot 'SHA256.txt'
if (-not (Test-Path -LiteralPath $sourceDll) -or -not (Test-Path -LiteralPath $checksumFile)) {
    throw 'Run this script from the extracted fast_reset_mod package, beside HollowRL.dll and SHA256.txt.'
}
$expectedHash = (Get-Content -LiteralPath $checksumFile -Raw).Trim()
if ((Get-FileHash -LiteralPath $sourceDll -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Package DLL checksum mismatch. Copy the package again.'
}
$targets = @()
foreach ($gameDir in $GameDirs) {
    $root = (Resolve-Path -LiteralPath $gameDir).Path
    $exe = Join-Path $root 'hollow_knight.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw "Game executable missing: $exe" }
    $running = @(Get-Process hollow_knight -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $exe })
    if ($running.Count) { throw "Close this game before replacing its loaded DLL: $exe" }
    $targets += Join-Path $root 'BepInEx\plugins\HollowRL\HollowRL.dll'
}
foreach ($target in $targets) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    if (Test-Path -LiteralPath $target) {
        $backup = $target + '.before_fast_reset_' + (Get-Date -Format 'yyyyMMdd_HHmmssfff') + '.bak'
        Copy-Item -LiteralPath $target -Destination $backup -ErrorAction Stop
        Write-Host "Backup: $backup"
    }
    Copy-Item -LiteralPath $sourceDll -Destination $target -Force
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $expectedHash) {
        throw "Installed DLL hash mismatch: $target"
    }
    Write-Host "Installed: $target"
}
Write-Host 'Restart games with their original host/port/boss arguments, then load the training save once.'
