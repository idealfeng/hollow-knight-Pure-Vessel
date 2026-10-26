param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9000,
    [ValidateRange(1, 8)]
    [int]$FrameStack = 4,
    [ValidateRange(5, 30)]
    [int]$ControlHz = 10
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot "python\.venv\Scripts\python.exe"
$trainerDir = Join-Path $projectRoot "python"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found at '$pythonExe'. Run: python -m venv python\.venv; python\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location $trainerDir
try {
    $env:HOLLOWRL_PORT = "$Port"
    $env:HOLLOWRL_FRAME_STACK = "$FrameStack"
    $env:HOLLOWRL_CONTROL_HZ = "$ControlHz"
    Write-Host "HollowRL trainer: port=$Port, frame_stack=$FrameStack, control_hz=$ControlHz"
    & $pythonExe main.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
