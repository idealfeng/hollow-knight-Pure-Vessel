param(
    [string]$DllPath = '',
    [string]$OutputPath = ''
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $DllPath) { $DllPath = Join-Path $taskRoot 'unity_mod\HollowRL\bin\Release\HollowRL.dll' }
if (-not $OutputPath) { $OutputPath = Join-Path $taskRoot 'artifacts\combat_state_b_machine.zip' }
$taskSource = (Resolve-Path -LiteralPath $DllPath -ErrorAction Stop).Path
if ([Reflection.AssemblyName]::GetAssemblyName($taskSource).Name -ne 'HollowRL') { throw 'Not a HollowRL assembly.' }
$taskSourceHash = (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
$taskOutput = [IO.Path]::GetFullPath($OutputPath)
$taskOutputParent = Split-Path -Parent $taskOutput
New-Item -ItemType Directory -Path $taskOutputParent -Force | Out-Null
$taskStaging = Join-Path ([IO.Path]::GetTempPath()) ('HollowRL-combat-state-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskStaging | Out-Null
Copy-Item -LiteralPath $taskSource -Destination (Join-Path $taskStaging 'HollowRL.dll')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'update_combat_state.ps1') -Destination (Join-Path $taskStaging 'update_combat_state.ps1')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'combat_state_README.txt') -Destination (Join-Path $taskStaging 'README.txt')
if ((Get-FileHash -LiteralPath (Join-Path $taskStaging 'HollowRL.dll') -Algorithm SHA256).Hash -ne $taskSourceHash) {
    throw 'Staged DLL checksum mismatch.'
}
$taskManifest = [ordered]@{
    schema = 1; name = 'combat_state_b_machine'; state_section = 'dash_v1'
    mod_changed = $true; save_changed = $false; source_dll_sha256 = $taskSourceHash
    files = [ordered]@{}
}
Get-ChildItem -LiteralPath $taskStaging -File | ForEach-Object {
    $taskManifest.files[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
}
$taskManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskStaging 'manifest.json') -Encoding UTF8
$taskTemporaryZip = Join-Path $taskOutputParent ([IO.Path]::GetFileNameWithoutExtension($taskOutput) + '.building.zip')
Compress-Archive -Path (Join-Path $taskStaging '*') -DestinationPath $taskTemporaryZip -Force
Move-Item -LiteralPath $taskTemporaryZip -Destination $taskOutput -Force
Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256
Write-Output "Package: $taskOutput"
Write-Output "DLL SHA256: $taskSourceHash"
# Keep the small staging directory for inspection; no game or save is changed.
