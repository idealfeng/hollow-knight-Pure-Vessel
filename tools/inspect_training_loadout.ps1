param(
    [string]$SavePath = (Join-Path $PSScriptRoot '..\HK_save\user1.dat'),
    [switch]$RequireStandardLoadout
)

# Read-only inspection of a disk save. No game connection, save installation,
# BinaryFormatter deserialization, process control or file write is performed.
# A disk save does not establish the loadout currently loaded in a game window.
$ErrorActionPreference = 'Stop'

function Read-SaveText([byte[]]$Payload) {
    [byte[]]$taskExpectedHeader = 0,1,0,0,0,255,255,255,255,1,0,0,0,0,0,0,0,6,1,0,0,0
    if ($Payload.Length -lt 24) { throw 'Save file is too short.' }
    for ($taskIndex = 0; $taskIndex -lt $taskExpectedHeader.Length; $taskIndex++) {
        if ($Payload[$taskIndex] -ne $taskExpectedHeader[$taskIndex]) {
            throw 'Unsupported save string-record format.'
        }
    }
    $taskStream = [IO.MemoryStream]::new($Payload)
    $taskReader = [IO.BinaryReader]::new($taskStream)
    try {
        $taskStream.Position = 22
        $taskEncoded = $taskReader.ReadString()
        if ($taskStream.Position -ne $taskStream.Length - 1 -or $taskReader.ReadByte() -ne 11) {
            throw 'Unexpected trailing save records.'
        }
        $taskCipher = [Convert]::FromBase64String($taskEncoded)
        # The native game's TeamCherry.SharedUtils.Encryption format, also
        # used by configure_training_loadout.ps1. Only decryption is needed.
        $taskCrypto = [Security.Cryptography.RijndaelManaged]::new()
        $taskCrypto.Key = [Text.Encoding]::UTF8.GetBytes('UKu52ePUBwetZ9wNX88o54dnfKRu0T1l')
        $taskCrypto.Mode = [Security.Cryptography.CipherMode]::ECB
        $taskCrypto.Padding = [Security.Cryptography.PaddingMode]::PKCS7
        $taskTransform = $taskCrypto.CreateDecryptor()
        try {
            return [Text.Encoding]::UTF8.GetString(
                $taskTransform.TransformFinalBlock($taskCipher, 0, $taskCipher.Length))
        } finally { $taskTransform.Dispose(); $taskCrypto.Dispose() }
    } finally { $taskReader.Dispose(); $taskStream.Dispose() }
}

try {
    $taskSource = (Resolve-Path -LiteralPath $SavePath -ErrorAction Stop).ProviderPath
    if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) { throw 'SavePath must be a file.' }
    $taskPayload = [IO.File]::ReadAllBytes($taskSource)
    $taskHasher = [Security.Cryptography.SHA256]::Create()
    try { $taskHash = [BitConverter]::ToString($taskHasher.ComputeHash($taskPayload)).Replace('-', '').ToLowerInvariant() }
    finally { $taskHasher.Dispose() }
    $taskData = (Read-SaveText $taskPayload) | ConvertFrom-Json
    $taskPlayer = $taskData.playerData
    if ($null -eq $taskPlayer) { throw 'Save has no playerData object.' }

    $taskListedCharms = @($taskPlayer.equippedCharms | ForEach-Object { [int]$_ })
    $taskFlaggedCharms = @(1..40 | Where-Object { $taskPlayer."equippedCharm_$_" -eq $true })
    $taskEquippedCharms = @(@($taskListedCharms) + @($taskFlaggedCharms) | Sort-Object -Unique)
    $taskErrors = @()
    $taskRequiredFields = @('nailSmithUpgrades', 'nailDamage', 'health', 'healthBlue',
        'fireballLevel', 'quakeLevel', 'screamLevel', 'overcharmed', 'charmSlotsFilled', 'equippedCharms')
    $taskRequiredFields += @(1..40 | ForEach-Object { "equippedCharm_$_" })
    foreach ($taskField in $taskRequiredFields) {
        if ($taskPlayer.PSObject.Properties.Name -notcontains $taskField) {
            $taskErrors += "Missing required field: $taskField"
        }
    }
    $taskExpected = [ordered]@{
        nailSmithUpgrades = 4; nailDamage = 21; health = 9; healthBlue = 0
        fireballLevel = 2; quakeLevel = 2; screamLevel = 2; charmSlotsFilled = 0
    }
    foreach ($taskField in $taskExpected.Keys) {
        if ($taskPlayer.$taskField -ne $taskExpected[$taskField]) {
            $taskErrors += "Expected $taskField=$($taskExpected[$taskField]); observed $($taskPlayer.$taskField)"
        }
    }
    if ($taskPlayer.overcharmed -ne $false) { $taskErrors += 'Expected overcharmed=false.' }
    $taskCombatCharms = @($taskEquippedCharms | Where-Object { $_ -ne 36 })
    if ($taskCombatCharms.Count) { $taskErrors += "Combat charms equipped: $($taskCombatCharms -join ',')" }

    $taskResult = [ordered]@{
        evidence = 'disk_save_only'
        path = $taskSource
        sha256 = $taskHash
        standard_loadout_required = [bool]$RequireStandardLoadout
        standard_loadout_valid = ($taskErrors.Count -eq 0)
        validation_errors = @($taskErrors)
        nailSmithUpgrades = $taskPlayer.nailSmithUpgrades
        nailDamage = $taskPlayer.nailDamage
        health = $taskPlayer.health
        maxHealth = $taskPlayer.maxHealth
        maxHealthBase = $taskPlayer.maxHealthBase
        healthBlue = $taskPlayer.healthBlue
        MPCharge = $taskPlayer.MPCharge
        MPReserve = $taskPlayer.MPReserve
        MPReserveMax = $taskPlayer.MPReserveMax
        fireballLevel = $taskPlayer.fireballLevel
        quakeLevel = $taskPlayer.quakeLevel
        screamLevel = $taskPlayer.screamLevel
        charmSlotsFilled = $taskPlayer.charmSlotsFilled
        overcharmed = $taskPlayer.overcharmed
        equippedCharms = @($taskEquippedCharms)
        equippedCharms_list = @($taskListedCharms)
        equippedCharms_flags = @($taskFlaggedCharms)
        abilities = [ordered]@{
            hasDash = $taskPlayer.hasDash
            hasShadowDash = $taskPlayer.hasShadowDash
            hasWalljump = $taskPlayer.hasWalljump
            hasDoubleJump = $taskPlayer.hasDoubleJump
            hasSuperDash = $taskPlayer.hasSuperDash
        }
    }
    $taskResult | ConvertTo-Json -Depth 5
    if ($RequireStandardLoadout -and $taskErrors.Count) {
        [Console]::Error.WriteLine('Disk save does not match the standard 21-damage training loadout.')
        exit 1
    }
} catch {
    [ordered]@{
        evidence = 'disk_save_only'; path = $SavePath
        standard_loadout_required = [bool]$RequireStandardLoadout
        standard_loadout_valid = $false; error = $_.Exception.Message
    } | ConvertTo-Json -Depth 3
    [Console]::Error.WriteLine('Could not inspect the disk save: ' + $_.Exception.Message)
    exit 1
}
