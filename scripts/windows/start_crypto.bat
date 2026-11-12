@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv-paper\Scripts\python.exe" (
 echo Run scripts\windows\install.bat first.
 exit /b 1
)
".venv-paper\Scripts\python.exe" launch.py --capital 10000 --port 8877
exit /b %ERRORLEVEL%
