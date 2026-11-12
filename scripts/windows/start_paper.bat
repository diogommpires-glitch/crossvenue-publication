@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv-paper\Scripts\python.exe" (
 echo Run scripts\windows\install.bat first.
 exit /b 1
)
if not exist "config\cross_asset_paper.json" (
 echo Create the private config from config\cross_asset_paper.example.json first.
 exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0provider_access.ps1" -Action Run -Port 8877
exit /b %ERRORLEVEL%
