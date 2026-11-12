@echo off
setlocal
cd /d "%~dp0..\.." || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0provider_access.ps1" -Action Configure
set "SETUP_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %SETUP_EXIT%
