param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001,
    [ValidateRange(1, 8)]
    [int]$FrameStack = 4,
    [ValidateRange(5, 30)]
    [int]$ControlHz = 10,
    [string]$RunDir = "python/runs/traitor_single"
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot "python\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found at '$pythonExe'. Run: python -m venv python\.venv; python\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& (Join-Path $projectRoot "run_multi_windows.ps1") -Ports @($Port) `
    -FrameStack $FrameStack -ControlHz $ControlHz -RunDir $RunDir
