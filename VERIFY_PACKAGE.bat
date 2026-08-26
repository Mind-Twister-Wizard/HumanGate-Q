@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setup is missing. Run START_HERE.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" verify_package.py
pause

