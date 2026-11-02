$py = "c:\Users\51292\maplestory-auto-ds\.venv\Scripts\python.exe"
$script = "c:\Users\51292\maplestory-auto-ds\tools\test_mouse_admin.py"
$out = "c:\Users\51292\maplestory-auto-ds\_debug_out\mouse_admin_test.txt"
New-Item -ItemType Directory -Force -Path "c:\Users\51292\maplestory-auto-ds\_debug_out" | Out-Null
Remove-Item $out -ErrorAction SilentlyContinue
Start-Process -FilePath $py -ArgumentList $script -WorkingDirectory "c:\Users\51292\maplestory-auto-ds" -Verb RunAs -Wait
if (Test-Path $out) {
  Get-Content $out
} else {
  Write-Output "No output file. UAC may have been cancelled."
}
