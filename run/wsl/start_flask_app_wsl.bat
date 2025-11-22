@echo off
chcp 65001 > nul
REM ============================================================================
REM Batch Script for Running Flask App (WSL Execution)
REM ============================================================================
REM This script uses WSL (Windows Subsystem for Linux) to run the
REM Flask web application in the correct Conda environment to load
REM the GPU-trained model.

echo.
echo Starting Flask web app in WSL environment...
echo (App logs will be displayed in this window. Press Ctrl+C to exit)
echo.

REM Set the path to the shell script inside WSL
set "SCRIPT_PATH=%~dp0sh\start_flask.sh"

REM Execute the shell script using wsl.exe
REM Any arguments passed to this batch file (%*) will be forwarded to the shell script.
REM e.g., start_flask_app_wsl.bat --port 5001
wsl.exe --cd ~ -u root -- bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate rapids-25.10 && bash $(wslpath -u '%SCRIPT_PATH%') %*"

echo.
echo Flask app has been terminated.
echo.
pause
