@echo off
chcp 65001 > nul

:: Navigate to the script's directory to prevent path issues.
cd /d "%~dp0"

:: Activate the virtual environment and run the script
.\venv\Scripts\python.exe weight_optimizer.py

echo.
echo Script execution complete. Press any key to close the window.
pause > nul