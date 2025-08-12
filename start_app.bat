@echo off
SET "SCRIPT_DIR=%~dp0"

REM Check if the virtual environment exists
IF NOT EXIST "%SCRIPT_DIR%venv" (
    echo "Creating virtual environment..."
    python -m venv "%SCRIPT_DIR%venv"
    IF %ERRORLEVEL% NEQ 0 (
        echo "Failed to create virtual environment. Please ensure Python is installed and in your PATH."
        pause
        exit /b
    )
)

REM Activate the virtual environment and install requirements
echo "Activating virtual environment and installing requirements..."
CALL "%SCRIPT_DIR%venv\Scripts\activate.bat"
pip install -r "%SCRIPT_DIR%requirements.txt"

REM Run the Streamlit application
echo "Starting the Streamlit application..."
streamlit run "%SCRIPT_DIR%app.py"

echo "Application finished."
pause
