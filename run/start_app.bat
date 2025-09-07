@echo off
REM Get the directory where the script is located and navigate to the project root
cd /d "%~dp0.."

REM Virtual environment folder name
set VENV_DIR=venv

REM Check if virtual environment exists, if not, create and install packages
if not exist %VENV_DIR%\ (
  echo Virtual environment not found. Creating a new one and installing packages.
  REM Create virtual environment with Python
  python -m venv %VENV_DIR%
  
  REM Activate virtual environment
  call %VENV_DIR%\Scripts\activate.bat
  
  REM Install basic requirements
  pip install -r requirements.txt
  pip install pandas-ta

  REM Install additional NLP related packages
  pip install "transformers[torch]" sentencepiece

  REM Pin numpy version to 1.26.4
  pip install --no-cache-dir --force-reinstall numpy==1.26.4

  REM Deactivate
  call %VENV_DIR%\Scripts\deactivate.bat
  echo Installation complete. Starting the app.
)

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Install/update latest packages
echo Installing/updating latest packages...
pip install -r requirements.txt
pip install pandas-ta
pip install "transformers[torch]" sentencepiece

REM Final pin numpy version to 1.26.4
echo Pinning Numpy version to compatible 1.26.4...
pip install --no-cache-dir --force-reinstall numpy==1.26.4

REM Run Streamlit app
echo Starting AI Stock Recommendation Platform...
echo Please wait until the app opens in your web browser.
streamlit run app.py
