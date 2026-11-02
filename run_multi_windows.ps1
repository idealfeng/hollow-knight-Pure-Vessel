param(
    [ValidateCount(1, 4)]
    [int[]]$Ports = @(9001, 9002),
    [string[]]$Endpoints = @(),
    [ValidateRange(1, 16)]
    [int]$FrameStack = 4,
    [ValidateRange(5, 30)]
    [int]$ControlHz = 10,
    [ValidateRange(0.5, 4.0)]
    [double]$GameSpeed = 1.0,
    [ValidateRange(0, 8192)]
    [int]$StepsPerEnv = 0,
    [ValidateRange(1, 1000000)]
    [int]$TotalUpdates = 10000,
    [ValidateRange(0, 10000)]
    [double]$StopBelowHits = 0,
    [ValidateRange(1e-7, 1.0)]
    [double]$Lr = 3e-4,
    [ValidateRange(0.0, 1.0)]
    [double]$EntCoef = 0.01,
    [ValidateRange(0, 100000)]
    [int]$EvalEvery = 0,
    [ValidateRange(1, 1000)]
    [int]$EvalEpisodes = 1,
    [ValidateSet('', 'exclusive_directions_v1', 'exclusive_directions_spells_v1', 'discrete_macros_v1')]
    [string]$ActionDistribution = '',
    [ValidateSet('victory_v1', 'nohit_v1', 'radiant_spacing_v1', 'radiant_wave_clear_v1', 'radiant_spells_v1', 'radiant_spells_dodge_v1', 'radiant_potential_v1')]
    [string]$RewardProfile = 'radiant_spells_dodge_v1',
    [ValidateRange(0, 2)]
    [int]$BossLevel = 2,
    [ValidateRange(0.1, 3600)]
    [double]$EpisodeTimeLimit = 150,
    [ValidateRange(0.1, 3600)]
    [double]$ObservationTimeLimit = 60,
    [ValidateSet('victory', 'nohit')]
    [string]$EvalObjective = 'nohit',
    [switch]$FreezeObsNorm,
    [switch]$NoMonitor,
    [ValidateRange(3.0, 15.0)]
    [double]$DeathSettleWait = 7.0,
    [ValidateRange(0.5, 5.0)]
    [double]$PostLeftWait = 2.0,
    [string]$InitCheckpoint = "",
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot "python\.venv\Scripts\python.exe"
$trainerDir = Join-Path $projectRoot "python"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found at '$pythonExe'."
}
if (($Ports | Sort-Object -Unique).Count -ne $Ports.Count) {
    throw "Ports must be unique."
}

# Inherit existing run settings unless the caller explicitly overrides them.
# This keeps a launcher default from silently changing a resumed experiment.
if ($RunDir) {
    $resolvedRunDir = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $projectRoot $RunDir }
}
else {
    $runPorts = if ($Endpoints.Count -gt 0) {
        @($Endpoints | ForEach-Object { $_.Substring($_.LastIndexOf(':') + 1) })
    } else { @($Ports) }
    $resolvedRunDir = Join-Path $trainerDir ('runs\traitor_exclusive_v2_' + ($runPorts -join '_'))
}
$savedCheckpoint = Join-Path $resolvedRunDir 'ppo_checkpoint.pth'
$resuming = Test-Path -LiteralPath $savedCheckpoint
$savedSettings = $null
if ($resuming) {
    $configPath = Join-Path $resolvedRunDir 'run_config.json'
    if (Test-Path -LiteralPath $configPath) {
        $savedSettings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    }
    else {
        # Older runs may only have the complete checkpoint. The trainer also
        # loads this trusted input; inspect its stored configuration here.
        $configCode = "import json, sys, torch; print(json.dumps(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('training_config', {})))"
        $savedText = & $pythonExe -c $configCode $savedCheckpoint
        if ($LASTEXITCODE -ne 0) { throw 'Could not read the existing checkpoint configuration.' }
        $savedSettings = $savedText | ConvertFrom-Json
    }
    if (-not $PSBoundParameters.ContainsKey('Endpoints') -and -not $PSBoundParameters.ContainsKey('Ports')) {
        if ($savedSettings.endpoints) { $Endpoints = @($savedSettings.endpoints) }
        elseif ($savedSettings.ports) { $Ports = @($savedSettings.ports) }
    }
    foreach ($binding in @(
        @('FrameStack', 'frame_stack'), @('ControlHz', 'control_hz'),
        @('GameSpeed', 'game_speed'), @('StepsPerEnv', 'steps_per_env'),
        @('TotalUpdates', 'total_updates'), @('Lr', 'lr'), @('EntCoef', 'ent_coef'),
        @('EvalEvery', 'eval_every'), @('EvalEpisodes', 'eval_episodes'),
        @('EpisodeTimeLimit', 'episode_time_limit'), @('ObservationTimeLimit', 'observation_time_limit'),
        @('FreezeObsNorm', 'freeze_obs_norm'), @('NoMonitor', 'no_monitor'),
        @('StopBelowHits', 'stop_below_hits')
    )) {
        if (-not $PSBoundParameters.ContainsKey($binding[0])) {
            $property = $savedSettings.PSObject.Properties[$binding[1]]
            if ($property -and $null -ne $property.Value) {
                Set-Variable -Name $binding[0] -Value $property.Value
            }
        }
    }
}

if ($Endpoints.Count -gt 0) {
    $arguments = @("main_multi.py", "--endpoints")
    $arguments += @($Endpoints)
}
else {
    $arguments = @("main_multi.py", "--ports")
    $arguments += @($Ports | ForEach-Object { "$_" })
}
$arguments += @(
    "--frame-stack", "$FrameStack",
    "--control-hz", "$ControlHz",
    "--game-speed", "$GameSpeed",
    "--total-updates", "$TotalUpdates",
    "--lr", "$Lr",
    "--ent-coef", "$EntCoef",
    "--eval-every", "$EvalEvery",
    "--eval-episodes", "$EvalEpisodes",
    "--episode-time-limit", "$EpisodeTimeLimit"
)
# A null observation limit in an old run means "use its episode limit".
if (-not $resuming -or $PSBoundParameters.ContainsKey('ObservationTimeLimit') -or
    $null -ne $savedSettings.observation_time_limit) {
    $arguments += @('--observation-time-limit', "$ObservationTimeLimit")
}
foreach ($option in @(
    @('ActionDistribution', '--action-distribution', $ActionDistribution),
    @('RewardProfile', '--reward-profile', $RewardProfile),
    @('BossLevel', '--boss-level', "$BossLevel"),
    @('EvalObjective', '--eval-objective', $EvalObjective)
)) {
    if ((-not $resuming -or $PSBoundParameters.ContainsKey($option[0])) -and
        -not [string]::IsNullOrEmpty([string]$option[2])) {
        $arguments += @($option[1], [string]$option[2])
    }
}
if ($FreezeObsNorm) { $arguments += '--freeze-obs-norm' }
if ($StopBelowHits -gt 0) {
    $arguments += @("--stop-below-hits", "$StopBelowHits")
}
if ($NoMonitor) {
    $arguments += "--no-monitor"
}
if ($StepsPerEnv -gt 0) {
    $arguments += @("--steps-per-env", "$StepsPerEnv")
}
if ($InitCheckpoint) {
    if ([IO.Path]::IsPathRooted($InitCheckpoint)) {
        $resolvedInit = $InitCheckpoint
    }
    else {
        $resolvedInit = Join-Path $projectRoot $InitCheckpoint
    }
    $arguments += @("--init-checkpoint", $resolvedInit)
}
$arguments += @("--run-dir", $resolvedRunDir)

Push-Location $trainerDir
try {
    $env:HOLLOWRL_DEATH_SETTLE_WAIT = "$DeathSettleWait"
    $env:HOLLOWRL_POST_LEFT_WAIT = "$PostLeftWait"
    if ($Endpoints.Count -gt 0) {
        Write-Host "Central PPO trainer: endpoints=$($Endpoints -join ',')"
    }
    else {
        Write-Host "Central PPO trainer: ports=$($Ports -join ',')"
    }
    & $pythonExe -u @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
