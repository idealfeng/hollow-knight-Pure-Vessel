param(
    [Parameter(Mandatory = $true)][string]$SavePath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

# Write a repaired copy; installing it requires closing/reloading the game.
# Preserve every JSON byte except these two capability booleans. Never use
# BinaryFormatter.Deserialize on a save file: its single string record has a
# small, fixed format that BinaryReader/BinaryWriter can handle directly.
$ErrorActionPreference = 'Stop'
$taskSource = [IO.Path]::GetFullPath($SavePath)
$taskOutput = [IO.Path]::GetFullPath($OutputPath)
if ($taskSource -eq $taskOutput) { throw 'Choose a separate output file; this tool does not overwrite the source save.' }
if (Test-Path -LiteralPath $taskOutput) { throw "Output already exists: $taskOutput" }

function Read-SaveText([byte[]]$Payload) {
    [byte[]]$taskExpectedHeader = 0,1,0,0,0,255,255,255,255,1,0,0,0,0,0,0,0,6,1,0,0,0
    if ($Payload.Length -lt 24) { throw 'Save file is too short.' }
    for ($taskIndex = 0; $taskIndex -lt $taskExpectedHeader.Length; $taskIndex++) {
        if ($Payload[$taskIndex] -ne $taskExpectedHeader[$taskIndex]) { throw 'Unsupported save string-record format.' }
    }
    $taskStream = [IO.MemoryStream]::new($Payload)
    $taskReader = [IO.BinaryReader]::new($taskStream)
    try {
        $taskStream.Position = 22
        $taskEncoded = $taskReader.ReadString()
        if ($taskStream.Position -ne $taskStream.Length - 1 -or $taskReader.ReadByte() -ne 11) {
            throw 'Unexpected trailing save records; refusing to modify this file.'
        }
        $taskCipher = [Convert]::FromBase64String($taskEncoded)
        $taskCrypto = New-SaveCrypto
        $taskTransform = $taskCrypto.CreateDecryptor()
        try {
            return [Text.Encoding]::UTF8.GetString($taskTransform.TransformFinalBlock($taskCipher, 0, $taskCipher.Length))
        } finally { $taskTransform.Dispose(); $taskCrypto.Dispose() }
    } finally { $taskReader.Dispose(); $taskStream.Dispose() }
}

function New-SaveCrypto {
    # The unmodified game's TeamCherry.SharedUtils.Encryption format.
    $taskCrypto = [Security.Cryptography.RijndaelManaged]::new()
    $taskCrypto.Key = [Text.Encoding]::UTF8.GetBytes('UKu52ePUBwetZ9wNX88o54dnfKRu0T1l')
    $taskCrypto.Mode = [Security.Cryptography.CipherMode]::ECB
    $taskCrypto.Padding = [Security.Cryptography.PaddingMode]::PKCS7
    return $taskCrypto
}

$taskOriginalBytes = [IO.File]::ReadAllBytes($taskSource)
$taskOriginalText = Read-SaveText $taskOriginalBytes
$taskOriginalData = $taskOriginalText | ConvertFrom-Json
$taskRepairedText = $taskOriginalText
foreach ($taskAbility in @('hasWalljump', 'hasSuperDash')) {
    $taskValue = $taskOriginalData.playerData.$taskAbility
    if ($taskValue -isnot [bool]) { throw "Missing boolean ability: $taskAbility" }
    $taskPattern = '("' + [Regex]::Escape($taskAbility) + '"\s*:\s*)(true|false)\b'
    if ([Regex]::Matches($taskRepairedText, $taskPattern).Count -ne 1) {
        throw "Ability field is missing or ambiguous: $taskAbility"
    }
    $taskRepairedText = [Regex]::Replace($taskRepairedText, $taskPattern, '${1}true')
}
$taskRepairedData = $taskRepairedText | ConvertFrom-Json
if (-not $taskRepairedData.playerData.hasWalljump -or -not $taskRepairedData.playerData.hasSuperDash) {
    throw 'Ability verification failed.'
}

$taskCrypto = New-SaveCrypto
$taskTransform = $taskCrypto.CreateEncryptor()
try {
    $taskPlain = [Text.Encoding]::UTF8.GetBytes($taskRepairedText)
    $taskEncoded = [Convert]::ToBase64String($taskTransform.TransformFinalBlock($taskPlain, 0, $taskPlain.Length))
} finally { $taskTransform.Dispose(); $taskCrypto.Dispose() }
$taskStream = [IO.MemoryStream]::new()
$taskWriter = [IO.BinaryWriter]::new($taskStream)
try {
    $taskWriter.Write($taskOriginalBytes, 0, 22)
    $taskWriter.Write($taskEncoded)
    $taskWriter.Write([byte]11)
    $taskWriter.Flush()
    [byte[]]$taskResult = $taskStream.ToArray()
} finally { $taskWriter.Dispose(); $taskStream.Dispose() }
if ((Read-SaveText $taskResult) -cne $taskRepairedText) { throw 'Repaired save did not survive encryption round-trip.' }
[IO.File]::WriteAllBytes($taskOutput, $taskResult)
[pscustomobject]@{
    source = $taskSource
    output = $taskOutput
    previous_hasWalljump = $taskOriginalData.playerData.hasWalljump
    previous_hasSuperDash = $taskOriginalData.playerData.hasSuperDash
    hasWalljump = $true
    hasSuperDash = $true
    sha256 = (Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash
    requires_save_reload = $true
} | ConvertTo-Json
