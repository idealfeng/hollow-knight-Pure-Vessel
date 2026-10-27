param(
    [Parameter(Mandatory = $true)]
    [ValidateCount(1, 4)]
    [string[]]$GameDirs,
    [int[]]$Ports = @(9001, 9002),
    [ValidateRange(30, 120)]
    [int]$Fps = 30,
    [ValidateRange(0.5, 4.0)]
    [double]$GameSpeed = 2.0,
    [ValidateRange(320, 1920)]
    [int]$Width = 640,
    [ValidateRange(180, 1080)]
    [int]$Height = 360
)

$ErrorActionPreference = "Stop"
if ($GameDirs.Count -ne $Ports.Count) {
    throw "GameDirs and Ports must contain the same number of items."
}
if (($Ports | Sort-Object -Unique).Count -ne $Ports.Count) {
    throw "Ports must be unique."
}

$appIdSource = Join-Path $PSScriptRoot "steam_appid.txt"
if (-not (Test-Path -LiteralPath $appIdSource)) {
    throw "Missing local Steam App ID file: $appIdSource"
}

$launched = @()

for ($index = 0; $index -lt $GameDirs.Count; $index++) {
    $gameDir = (Resolve-Path -LiteralPath $GameDirs[$index]).Path
    $exe = Join-Path $gameDir "hollow_knight.exe"
    $plugin = Join-Path $gameDir "BepInEx\plugins\HollowRL\HollowRL.dll"
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "Game executable not found: $exe"
    }
    if (-not (Test-Path -LiteralPath $plugin)) {
        throw "HollowRL plugin not found: $plugin"
    }

    $port = $Ports[$index]
    $portPattern = "^\s*TCP\s+127\.0\.0\.1:$port\s+\S+\s+LISTENING\s+(\d+)\s*$"
    $listeningLine = netstat -ano -p tcp | Select-String -Pattern $portPattern | Select-Object -First 1
    if ($listeningLine) {
        $null = $listeningLine.Line -match $portPattern
        $existingPid = [int]$Matches[1]
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if (-not $existingProcess -or $existingProcess.Path -ne $exe) {
            throw "Port $port is already owned by PID $existingPid, not by $exe"
        }
        $launched += [pscustomobject]@{
            Index = $index
            Port = $port
            Process = $existingProcess
            Exe = $exe
            GameDir = $gameDir
        }
        Write-Host "Already running: environment $index on port $port (PID $existingPid)"
        continue
    }

    $appIdTarget = Join-Path $gameDir "steam_appid.txt"
    if (Test-Path -LiteralPath $appIdTarget) {
        $appId = (Get-Content -LiteralPath $appIdTarget -Raw).Trim()
        if ($appId -ne "367520") {
            throw "Unexpected Steam App ID in ${appIdTarget}: $appId"
        }
    }
    else {
        Copy-Item -LiteralPath $appIdSource -Destination $appIdTarget
    }

    $gameArguments = @(
        "--hollowrl-port=$($Ports[$index])",
        "--hollowrl-fps=$Fps",
        "--hollowrl-speed=$GameSpeed",
        "-screen-fullscreen", "0",
        "-screen-width", "$Width",
        "-screen-height", "$Height"
    )
    $process = Start-Process -FilePath $exe -WorkingDirectory $gameDir `
        -ArgumentList $gameArguments -WindowStyle Normal -PassThru
    $launched += [pscustomobject]@{
        Index = $index
        Port = $Ports[$index]
        Process = $process
        Exe = $exe
        GameDir = $gameDir
    }
    Write-Host "Launching environment $index on port $($Ports[$index]) (PID $($process.Id)): $exe"
}

foreach ($instance in $launched) {
    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        $instance.Process.Refresh()
        if ($instance.Process.HasExited) {
            $playerLog = Join-Path $env:USERPROFILE "AppData\LocalLow\Team Cherry\Hollow Knight\Player-prev.log"
            $hint = ""
            if (Test-Path -LiteralPath $playerLog) {
                $hint = (Get-Content -LiteralPath $playerLog -Tail 25) -join "`n"
            }
            throw "Environment $($instance.Index) exited before port $($instance.Port) was ready. See $playerLog`n$hint"
        }
        $portPattern = "^\s*TCP\s+127\.0\.0\.1:$($instance.Port)\s+\S+\s+LISTENING\s+$($instance.Process.Id)\s*$"
        $listener = netstat -ano -p tcp | Select-String -Pattern $portPattern
        if ($listener) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        throw "Environment $($instance.Index) did not listen on port $($instance.Port) within 30 seconds. Check $($instance.GameDir)\BepInEx\LogOutput.log"
    }
    Write-Host "Ready: environment $($instance.Index), PID $($instance.Process.Id), port $($instance.Port)"
}
