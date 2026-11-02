@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo.
echo ========================================
echo   20秒后自动结束 — 请立刻切到游戏打键
echo ========================================
echo.
".venv\Scripts\python.exe" "tools\record_key_timing.py"
echo.
pause
