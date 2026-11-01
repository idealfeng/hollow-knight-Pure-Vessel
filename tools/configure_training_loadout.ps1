param(
    [Parameter(Mandatory = $true)][string]$SavePath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
# Produce a separate legal 21-damage, spell-enabled, no-combat-charms copy.
# Install only while the game is closed, then reload save slot 1.
$ErrorActionPreference = 'Stop'
$taskSource = [IO.Path]::GetFullPath($SavePath)
$taskOutput = [IO.Path]::GetFullPath($OutputPath)
if ($taskSource -eq $taskOutput) { throw 'Choose a separate output file.' }
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
$taskData = $taskOriginalText | ConvertFrom-Json
$taskPlayer = $taskData.playerData
if ($taskPlayer.nailSmithUpgrades -ne 4) { throw 'Expected a fully upgraded nail.' }
foreach ($taskSpell in @('fireballLevel', 'quakeLevel', 'screamLevel')) {
    if ($taskPlayer.$taskSpell -ne 2) { throw "Upgrade $taskSpell in game first; this tool does not grant abilities." }
}
$taskDesired = [ordered]@{nailDamage=21;charmSlotsFilled=0;overcharmed=$false}
for ($taskIndex=1; $taskIndex -le 40; $taskIndex++) {
    $taskDesired["equippedCharm_$taskIndex"] = ($taskIndex -eq 36 -and $taskPlayer.gotShadeCharm -and $taskPlayer.royalCharmState -eq 4)
}
$taskDesired['equippedCharms'] = $(if ($taskDesired['equippedCharm_36']) { '[36]' } else { '[]' })
$taskRepairedText = $taskOriginalText
foreach ($taskField in $taskDesired.Keys) {
    $taskValue = $taskDesired[$taskField]
    if ($taskField -eq 'equippedCharms') { $taskPatternValue = '\[[^\]]*\]'; $taskLiteral = $taskValue }
    elseif ($taskValue -is [bool]) { $taskPatternValue = 'true|false'; $taskLiteral = $taskValue.ToString().ToLowerInvariant() }
    else { $taskPatternValue = '-?\d+'; $taskLiteral = [string]$taskValue }
    $taskPattern = '("' + [Regex]::Escape($taskField) + '"\s*:\s*)(' + $taskPatternValue + ')'
    if ([Regex]::Matches($taskRepairedText, $taskPattern).Count -ne 1) { throw "Missing or ambiguous field: $taskField" }
    $taskRepairedText = [Regex]::Replace($taskRepairedText, $taskPattern, '${1}' + $taskLiteral)
}
$taskVerified = ($taskRepairedText | ConvertFrom-Json).playerData
if ($taskVerified.nailDamage -ne 21 -or $taskVerified.charmSlotsFilled -ne 0 -or $taskVerified.overcharmed) { throw 'Loadout verification failed.' }
foreach ($taskIndex in 1..40) {
    if ($taskIndex -ne 36 -and $taskVerified."equippedCharm_$taskIndex") { throw "Combat charm still equipped: $taskIndex" }
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
if ((Read-SaveText $taskResult) -cne $taskRepairedText) { throw 'Save round-trip failed.' }
[IO.File]::WriteAllBytes($taskOutput, $taskResult)
[pscustomobject]@{
    source=$taskSource;output=$taskOutput;previous_nail_damage=$taskPlayer.nailDamage
    nail_damage=21;combat_charms=@();void_heart_retained=[bool]$taskVerified.equippedCharm_36
    spells=@('shade_soul','abyss_shriek','descending_dark');requires_save_reload=$true
    sha256=(Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash
} | ConvertTo-Json
