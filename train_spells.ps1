param(
    [string[]]$Endpoints = @('127.0.0.1:9001'),
    [string]$InitCheckpoint = 'python/runs/spell_initial/initial_policy.pth',
    [string]$RunDir = 'python/runs/traitor_radiant_spells',
    [ValidateRange(1, 1000000)][int]$TotalUpdates = 10000,
    [switch]$NoMonitor
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot 'python\.venv\Scripts\python.exe'
$taskInit = if ([IO.Path]::IsPathRooted($InitCheckpoint)) { $InitCheckpoint } else { Join-Path $PSScriptRoot $InitCheckpoint }
$taskRun = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $PSScriptRoot $RunDir }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the Python environment first.' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRun 'ppo_checkpoint.pth')) -and -not (Test-Path -LiteralPath $taskInit)) {
    throw 'Create the spell initialization with python/migrate_spell_policy.py first.'
}
$taskArguments = @(
    '-u', (Join-Path $PSScriptRoot 'python\main_multi.py'),
    '--endpoints'
) + $Endpoints + @(
    '--run-dir', $taskRun, '--init-checkpoint', $taskInit,
    '--action-distribution', 'exclusive_directions_spells_v1',
    '--reward-profile', 'radiant_spells_v1', '--boss-level', '2', '--eval-objective', 'nohit',
    '--episode-time-limit', '120', '--observation-time-limit', '60',
    '--control-hz', '10', '--frame-stack', '4', '--game-speed', '1',
    '--lr', '0.0001', '--ent-coef', '0.005', '--freeze-obs-norm',
    '--steps-per-env', '512', '--total-updates', "$TotalUpdates",
    '--eval-every', '25', '--eval-episodes', '10'
)
if ($NoMonitor) { $taskArguments += '--no-monitor' }
Write-Host 'Verify every game loaded the 21-damage, no-combat-charms save before training.'
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Spell trainer exited with code $LASTEXITCODE" }
