param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001,
    [ValidateRange(1, 10000)]
    [int]$Episodes = 100,
    [switch]$NoPrompt
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot 'python\.venv\Scripts\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'python\play.py'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the Python environment first: python -m venv python\.venv'
}
Write-Host 'Frozen policy | Native Radiant | 1x speed | Stop after a flawless win'
Write-Host 'Load save 1 and stand at the Traitor Lord statue. Start your screen recorder.'
if (-not $NoPrompt) { Read-Host 'Press Enter when recording is ready' | Out-Null }
& $taskPython $taskRunner --port $Port --episodes $Episodes
if ($LASTEXITCODE -ne 0) { throw "Policy player exited with code $LASTEXITCODE" }
