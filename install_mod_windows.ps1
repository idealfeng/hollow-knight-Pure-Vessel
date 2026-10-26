param(
    [string]$GameDir = "D:\Program Files\Steam\steamapps\common\Hollow Knight"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$solution = Join-Path $projectRoot "unity_mod\HollowRL.slnx"
$builtDll = Join-Path $projectRoot "unity_mod\HollowRL\bin\Release\HollowRL.dll"
$pluginDir = Join-Path $GameDir "BepInEx\plugins\HollowRL"
$targetDll = Join-Path $pluginDir "HollowRL.dll"

if (Get-Process hollow_knight -ErrorAction SilentlyContinue) {
    throw "Hollow Knight is running. Close it before replacing the loaded mod DLL."
}
if (-not (Test-Path -LiteralPath $GameDir)) {
    throw "Hollow Knight directory not found: $GameDir"
}

& dotnet build $solution --configuration Release
if ($LASTEXITCODE -ne 0) {
    throw "HollowRL mod build failed."
}

New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null
Copy-Item -LiteralPath $builtDll -Destination $targetDll -Force
Write-Host "Installed: $targetDll"
