@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setup is missing. Run START_HERE.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" review_flagged_cases.py --limit 10
pause

