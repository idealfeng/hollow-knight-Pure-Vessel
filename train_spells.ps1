param(
    [string[]]$Endpoints = @('127.0.0.1:9001'),
    [string]$InitCheckpoint = 'models/traitor_lord_radiant.pth',
    [string]$RunDir = 'python/runs/traitor_radiant_spells',
    [ValidateRange(1, 1000000)][int]$TotalUpdates = 10000,
    [ValidateRange(1, 16)][int]$FrameStack = 16,
    [ValidateSet('radiant_spells_v1', 'radiant_spells_dodge_v1')]
    [string]$RewardProfile = 'radiant_spells_dodge_v1',
    [ValidateRange(1e-7, 1.0)][double]$Lr = 0.0001,
    [ValidateRange(0.0, 1.0)][double]$EntCoef = 0.005,
    [ValidateRange(0.1, 3600)][double]$EpisodeTimeLimit = 150,
    [ValidateRange(0.1, 3600)][double]$ObservationTimeLimit = 60,
    [ValidateRange(0, 100000)][int]$EvalEvery = 25,
    [ValidateRange(1, 1000)][int]$EvalEpisodes = 10,
    [switch]$FreezeObsNorm = $true,
    [switch]$NoMonitor
)
$ErrorActionPreference = 'Stop'
$taskInit = if ([IO.Path]::IsPathRooted($InitCheckpoint)) { $InitCheckpoint } else { Join-Path $PSScriptRoot $InitCheckpoint }
$taskRun = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $PSScriptRoot $RunDir }
$taskResuming = Test-Path -LiteralPath (Join-Path $taskRun 'ppo_checkpoint.pth')
if (-not $taskResuming -and -not (Test-Path -LiteralPath $taskInit)) {
    throw "Initial checkpoint missing: $taskInit. Provide a complete 21-damage spell checkpoint."
}

$taskParameters = @{RunDir=$taskRun}
if (-not $taskResuming -or $PSBoundParameters.ContainsKey('Endpoints')) { $taskParameters.Endpoints = $Endpoints }
if (-not $taskResuming -or $PSBoundParameters.ContainsKey('InitCheckpoint')) { $taskParameters.InitCheckpoint = $taskInit }
foreach ($taskName in @('TotalUpdates', 'FrameStack', 'RewardProfile', 'Lr', 'EntCoef',
    'EpisodeTimeLimit', 'ObservationTimeLimit', 'EvalEvery', 'EvalEpisodes', 'FreezeObsNorm', 'NoMonitor')) {
    # Existing runs inherit saved settings through the general launcher.
    # Only explicit switches may replace them; preset defaults apply to a new run.
    if (-not $taskResuming -or $PSBoundParameters.ContainsKey($taskName)) {
        $taskParameters[$taskName] = (Get-Variable -Name $taskName -ValueOnly)
    }
}
if (-not $taskResuming) {
    $taskParameters.ActionDistribution = 'exclusive_directions_spells_v1'
    $taskParameters.BossLevel = 2
    $taskParameters.EvalObjective = 'nohit'
    $taskParameters.ControlHz = 10
    $taskParameters.GameSpeed = 1.0
    $taskParameters.StepsPerEnv = 512
}
if ($taskResuming) { Write-Host "Resuming saved run settings: $taskRun" }
else { Write-Host '21-damage spell policy | Native Radiant | 1x speed' }
Write-Host 'Load the 21-damage, no-combat-charms save in every selected game before training.'
& (Join-Path $PSScriptRoot 'run_multi_windows.ps1') @taskParameters
exit $LASTEXITCODE
