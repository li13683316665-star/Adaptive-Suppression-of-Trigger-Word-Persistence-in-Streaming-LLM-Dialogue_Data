# Brings minimized or background PowerShell windows to the foreground (Windows).
# Run: powershell -ExecutionPolicy Bypass -File scripts/Bring-PowerShellToFront.ps1
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  public const int SW_RESTORE = 9;
}
"@
$procs = Get-Process powershell,pwsh -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero }
if (-not $procs) {
    Write-Host "No PowerShell window with a visible handle found (may be headless or already closed)."
    exit 0
}
Write-Host "Found $($procs.Count) window(s). Restoring and focusing the first:"
$procs | ForEach-Object { Write-Host "  PID $($_.Id)  Title: $($_.MainWindowTitle)" }
$p = $procs | Select-Object -First 1
[void][Win32]::ShowWindowAsync($p.MainWindowHandle, [Win32]::SW_RESTORE)
[void][Win32]::SetForegroundWindow($p.MainWindowHandle)
