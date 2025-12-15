@echo off
chcp 65001 > nul
REM ============================================================================
REM LightGBM Model Training Batch Script (WSL Execution)
REM ============================================================================
REM This script uses WSL (Windows Subsystem for Linux) to run
REM the LightGBM training pipeline.

REM --- WSL Execution ---
echo.
echo Starting LightGBM training via WSL... (Logs will be displayed in this window)
echo.

REM Set the path to the shell script inside WSL
set "SCRIPT_PATH=%~dp0sh\train_lgbm.sh"

REM Execute the shell script using wsl.exe
wsl.exe --cd ~ -u root -- bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate rapids-25.10 && bash $(wslpath -u '%SCRIPT_PATH%')"

echo.
echo The training process has completed.
echo.
pause
