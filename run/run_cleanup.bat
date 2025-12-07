@echo off
chcp 65001 > nul
REM ============================================================================
REM Batch Script for Cleaning Up Preprocessed Data (WSL Execution)
REM ============================================================================

echo.
echo Cleaning up preprocessed data in WSL environment...
echo.

REM Set the path to the shell script inside WSL
set "SCRIPT_PATH=%~dp0sh\cleanup_data.sh"

REM Execute the shell script using wsl.exe
wsl.exe --cd ~ -u root -- bash -c "bash $(wslpath -u '%SCRIPT_PATH%')"

echo.
echo Cleanup process has completed.
echo.
pause

