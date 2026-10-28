param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001,
    [ValidateRange(1, 100)]
    [int]$Episodes = 5,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot 'python\.venv\Scripts\python.exe'
$recorder = Join-Path $PSScriptRoot 'python\record_human.py'
$recordArguments = @('-u', $recorder, '--port', "$Port", '--episodes', "$Episodes")
if ($Output) {
    $resolvedOutput = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $PSScriptRoot $Output }
    $recordArguments += @('--output', $resolvedOutput)
}
& $pythonExe @recordArguments
exit $LASTEXITCODE
