param(
    [string]$OutputPath = ''
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) { $OutputPath = Join-Path $taskRoot 'artifacts\loadout21_b_machine.zip' }
$taskOutput = [IO.Path]::GetFullPath($OutputPath)
$taskOutputParent = Split-Path -Parent $taskOutput
New-Item -ItemType Directory -Path $taskOutputParent -Force | Out-Null
$taskStaging = Join-Path ([IO.Path]::GetTempPath()) ('HollowRL-loadout-package-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskStaging | Out-Null
foreach ($taskName in @('update_training_loadout.ps1', 'configure_training_loadout.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $taskName) -Destination (Join-Path $taskStaging $taskName)
}
$taskReadme = @'
HollowRL: 21-damage, no-combat-charms save update

1. Keep B's original 9003/9004 game windows running. Stop any controller/trainer.
2. Extract this ZIP to a normal folder (do not run from inside the ZIP).
3. Open PowerShell in that folder and run:
   powershell -ExecutionPolicy Bypass -File .\update_training_loadout.ps1
4. The script closes those games, backs up save slot 1, sets nail damage to 21,
   removes combat charms, verifies the result, and restarts the original games.
5. Load save slot 1 in both windows; stand at the Traitor Lord statue.
   Do not select difficulty or enter the fight manually.
6. Tell the A-machine operator that B's two windows are ready.

If all games are already closed and you will restart them yourself:
   powershell -ExecutionPolicy Bypass -File .\update_training_loadout.ps1 -NoRestart

The backup is in the save directory under HollowRL_loadout_backups/<timestamp>.
This package needs only Windows PowerShell. It does not install Python, weights
or a DLL. Existing unlocked spells and movement abilities are preserved.
The converter requires the Pure Nail and all three upgraded spells to exist.
Void Heart is retained if present; no combat charms are equipped.
Central training and model updates run only on A, after all four windows reload.
'@
Set-Content -LiteralPath (Join-Path $taskStaging 'README.txt') -Value $taskReadme -Encoding UTF8
$taskManifest = [ordered]@{schema=1;name='loadout21_b_machine';nail_damage=21;combat_charms=@();mod_changed=$false;files=@{}}
Get-ChildItem -LiteralPath $taskStaging -File | ForEach-Object {
    $taskManifest.files[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
}
$taskManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskStaging 'manifest.json') -Encoding UTF8
$taskTemporaryZip = Join-Path $taskOutputParent ([IO.Path]::GetFileNameWithoutExtension($taskOutput) + '.building.zip')
Compress-Archive -Path (Join-Path $taskStaging '*') -DestinationPath $taskTemporaryZip -Force
Move-Item -LiteralPath $taskTemporaryZip -Destination $taskOutput -Force
Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256
Write-Output "Package: $taskOutput"
# Leave the small temporary staging directory available for inspection.
