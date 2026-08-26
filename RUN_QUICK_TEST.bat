@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setup is missing. Run START_HERE.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_all.py --quick
if errorlevel 1 (
    echo Quick experiment failed. Read the error above.
    pause
    exit /b 1
)
echo SUCCESS: Quick results are in outputs\quick\
pause

