param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9001,
    [ValidateRange(1, 100)]
    [int]$Episodes = 5,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$activeTrainer = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'main_multi\.py' })
if ($activeTrainer.Count -gt 0) {
    throw 'A PPO trainer is still running. Stop it normally before recording human play.'
}
$pythonExe = Join-Path $PSScriptRoot 'python\.venv\Scripts\python.exe'
$recorder = Join-Path $PSScriptRoot 'python\record_human.py'
$recordArguments = @('-u', $recorder, '--port', "$Port", '--episodes', "$Episodes")
if ($Output) {
    $resolvedOutput = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $PSScriptRoot $Output }
    $recordArguments += @('--output', $resolvedOutput)
}
& $pythonExe @recordArguments
exit $LASTEXITCODE
