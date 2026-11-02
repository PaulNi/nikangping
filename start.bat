@echo off
setlocal
cd /d "%~dp0"

rem Game runs as admin; bot must too or mouse-follow is blocked by UIPI.
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [MapleStoryGo] Requesting Administrator...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs"
  exit /b
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv\Scripts\python.exe
  echo Run from project folder after creating the venv.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -m src.main
echo.
pause
