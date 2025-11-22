@echo off
chcp 65001 > nul
REM ============================================================================
REM GPU Model Training Batch Script (WSL Execution)
REM ============================================================================
REM This script uses WSL (Windows Subsystem for Linux) to run
REM the GPU-accelerated training pipeline.

REM --- User Input ---
echo.
echo Starting GPU Model Training Pipeline.
echo.
set /p n_iter="Enter number of iterations (e.g., 100): "
set /p max_depth="Enter max_depth candidates (e.g., 10 20 30 40): "
echo.

REM --- WSL Execution ---
echo Starting GPU training via WSL... (Logs will be displayed in this window)
echo.

REM Set the path to the shell script inside WSL
set "SCRIPT_PATH=%~dp0sh\train_gpu.sh"

REM Execute the shell script using wsl.exe
wsl.exe --cd ~ -u root -- bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate rapids-25.10 && bash $(wslpath -u '%SCRIPT_PATH%') --n_iter %n_iter% --max_depth %max_depth%"

echo.
echo The training process has completed.
echo.
pause
