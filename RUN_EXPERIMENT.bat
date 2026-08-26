@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setup is missing. Run START_HERE.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_all.py
if errorlevel 1 (
    echo Experiment failed. Read the error above.
    pause
    exit /b 1
)
echo SUCCESS: Open outputs\latest\PAPER_RESULTS_SUMMARY.md
pause

