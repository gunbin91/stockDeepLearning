@echo off
chcp 65001 > nul
REM ============================================================================
REM Batch Script for Cleaning Up CatBoost Preprocessed Data (WSL Execution)
REM ============================================================================
REM This script cleans up only preprocessed data, NOT trained models.
REM Trained models (catboost_model.cbm, etc.) are NOT deleted.

echo.
echo Cleaning up CatBoost preprocessed data in WSL environment...
echo.
echo NOTE: This will only delete preprocessed data files.
echo       Trained models will NOT be deleted.
echo.

REM Set the path to the shell script inside WSL
set "SCRIPT_PATH=%~dp0sh\cleanup_catboost_data.sh"

REM Execute the shell script using wsl.exe
wsl.exe --cd ~ -u root -- bash -c "bash $(wslpath -u '%SCRIPT_PATH%')"

echo.
echo Cleanup process has completed.
echo.
pause
