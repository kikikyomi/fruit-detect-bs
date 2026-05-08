@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start.ps1"

echo.
echo Startup command finished. You can close this window after the browser opens.
pause
