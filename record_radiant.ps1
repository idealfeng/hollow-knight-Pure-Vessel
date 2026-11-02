param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001,
    [ValidateRange(0, 10000)]
    [int]$Episodes = 100,
    [string]$Checkpoint = (Join-Path $PSScriptRoot 'models\traitor_lord_radiant.pth'),
    [ValidateSet(0, 1, 2)]
    [int]$BossLevel = 2,
    [ValidateSet('flawless', 'win', 'episodes')]
    [string]$StopAfter = 'flawless',
    [switch]$NoPrompt
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot 'python\.venv\Scripts\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'python\play.py'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the Python environment first: python -m venv python\.venv'
}
if ($StopAfter -eq 'episodes' -and $Episodes -eq 0) {
    throw '-StopAfter episodes requires a positive finite -Episodes budget.'
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Checkpoint not found: $Checkpoint"
}
$taskDifficulty = @('Attuned', 'Ascended', 'Radiant')[$BossLevel]
Write-Host "Frozen policy | Native $taskDifficulty | 1x speed | Stop after: $StopAfter"
Write-Host 'Load save 1 and stand at the Traitor Lord statue. Start your screen recorder.'
if (-not $NoPrompt) { Read-Host 'Press Enter when recording is ready' | Out-Null }
& $taskPython $taskRunner --checkpoint $Checkpoint --port $Port --episodes $Episodes `
    --boss-level $BossLevel --stop-after $StopAfter
if ($LASTEXITCODE -ne 0) { throw "Policy player exited with code $LASTEXITCODE" }
