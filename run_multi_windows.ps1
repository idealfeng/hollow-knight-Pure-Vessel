param(
    [ValidateCount(1, 4)]
    [int[]]$Ports = @(9001, 9002),
    [ValidateRange(1, 8)]
    [int]$FrameStack = 4,
    [ValidateRange(5, 30)]
    [int]$ControlHz = 10,
    [ValidateRange(0.5, 4.0)]
    [double]$GameSpeed = 2.0,
    [ValidateRange(0, 8192)]
    [int]$StepsPerEnv = 0,
    [ValidateRange(1, 1000000)]
    [int]$TotalUpdates = 10000,
    [switch]$NoMonitor,
    [ValidateRange(3.0, 15.0)]
    [double]$DeathSettleWait = 7.0,
    [ValidateRange(0.5, 5.0)]
    [double]$PostLeftWait = 2.0,
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

$arguments = @("main_multi.py", "--ports")
$arguments += @($Ports | ForEach-Object { "$_" })
$arguments += @(
    "--frame-stack", "$FrameStack",
    "--control-hz", "$ControlHz",
    "--game-speed", "$GameSpeed",
    "--total-updates", "$TotalUpdates"
)
if ($NoMonitor) {
    $arguments += "--no-monitor"
}
if ($StepsPerEnv -gt 0) {
    $arguments += @("--steps-per-env", "$StepsPerEnv")
}
if ($RunDir) {
    if ([IO.Path]::IsPathRooted($RunDir)) {
        $resolvedRunDir = $RunDir
    }
    else {
        $resolvedRunDir = Join-Path $projectRoot $RunDir
    }
    $arguments += @("--run-dir", $resolvedRunDir)
}

Push-Location $trainerDir
try {
    $env:HOLLOWRL_DEATH_SETTLE_WAIT = "$DeathSettleWait"
    $env:HOLLOWRL_POST_LEFT_WAIT = "$PostLeftWait"
    Write-Host "Central PPO trainer: ports=$($Ports -join ',')"
    & $pythonExe @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
