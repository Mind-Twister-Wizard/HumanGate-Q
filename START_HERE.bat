@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo HumanGate-Q: complete setup and experiment
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating Python environment...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 goto :failed
) else (
    echo [1/5] Existing Python environment found.
)

echo [2/5] Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [3/5] Downloading the Kaggle dataset...
".venv\Scripts\python.exe" download_dataset.py
if errorlevel 1 goto :dataset_failed

echo [4/5] Verifying package...
".venv\Scripts\python.exe" verify_package.py
if errorlevel 1 goto :failed

echo [5/5] Running the complete experiment...
".venv\Scripts\python.exe" run_all.py
if errorlevel 1 goto :failed

echo.
echo SUCCESS: Open outputs\latest\PAPER_RESULTS_SUMMARY.md
pause
exit /b 0

:dataset_failed
echo.
echo Automatic Kaggle download did not finish.
echo Download the dataset manually from the URL in QUICK_START_WINDOWS.md,
echo extract it into data\raw, and then run RUN_EXPERIMENT.bat.
pause
exit /b 2

:failed
echo.
echo HumanGate-Q stopped because a command failed.
echo Read the error shown above and QUICK_START_WINDOWS.md.
pause
exit /b 1

