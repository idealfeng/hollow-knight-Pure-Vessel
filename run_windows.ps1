$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot "python\.venv\Scripts\python.exe"
$trainerDir = Join-Path $projectRoot "python"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found at '$pythonExe'. Run: python -m venv python\.venv; python\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location $trainerDir
try {
    & $pythonExe main.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
