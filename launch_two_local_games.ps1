$primary = "D:\Program Files\Steam\steamapps\common\Hollow Knight"
$secondary = "D:\HollowRL-instances\env2"

& (Join-Path $PSScriptRoot "launch_multi_games.ps1") `
    -GameDirs $primary,$secondary `
    -Ports 9001,9002 `
    -Fps 60 `
    -GameSpeed 2.0 `
    -Width 640 `
    -Height 360
