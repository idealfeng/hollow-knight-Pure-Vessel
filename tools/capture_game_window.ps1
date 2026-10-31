param(
    [int]$Port = 9001,
    [string]$Output = "python/runs/screenshots/game_$Port.png"
)
$ErrorActionPreference = 'Stop'
$matches = @(Get-CimInstance Win32_Process -Filter "Name = 'hollow_knight.exe'" |
    Where-Object { $_.CommandLine -match "--hollowrl-port=$Port(?:\s|$)" })
if ($matches.Count -ne 1) { throw "Expected one game for port $Port" }
$gameProcess = Get-Process -Id $matches[0].ProcessId
$window = $gameProcess.MainWindowHandle
if ($window -eq [IntPtr]::Zero) { throw 'No game window' }
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GameCaptureNative {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
}
'@
[void][GameCaptureNative]::SetProcessDPIAware()
$rect = New-Object GameCaptureNative+Rect
if (-not [GameCaptureNative]::GetWindowRect($window, [ref]$rect)) { throw 'Cannot locate window' }
$bitmap = New-Object Drawing.Bitmap ($rect.Right - $rect.Left), ($rect.Bottom - $rect.Top)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
try {
    $deviceContext = $graphics.GetHdc()
    try {
        if (-not [GameCaptureNative]::PrintWindow($window, $deviceContext, 2)) { throw 'Window capture failed' }
    } finally { $graphics.ReleaseHdc($deviceContext) }
    $outputPath = [IO.Path]::GetFullPath($Output)
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath))
    $bitmap.Save($outputPath, [Drawing.Imaging.ImageFormat]::Png)
    Write-Output $outputPath
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
