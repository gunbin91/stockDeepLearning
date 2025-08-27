@echo off
chcp 65001 > nul
SET "ROOT_DIR=%~dp0.."

REM 가상 환경 확인
IF NOT EXIST "%ROOT_DIR%\venv" (
    echo "가상 환경을 생성합니다..."
    python -m venv "%ROOT_DIR%\venv"
    IF %ERRORLEVEL% NEQ 0 (
        echo "가상 환경 생성에 실패했습니다. Python이 설치되어 있고 PATH에 등록되었는지 확인하세요."
        pause
        exit /b
    )
)

REM 가상 환경 활성화 및 패키지 설치
echo "가상 환경을 활성화하고 패키지를 설치합니다..."
CALL "%ROOT_DIR%\venv\Scripts\activate.bat"
pip install -r "%ROOT_DIR%\requirements.txt"

REM Streamlit 앱 실행
echo "Streamlit 애플리케이션을 시작합니다..."
streamlit run "%ROOT_DIR%\app.py"

echo "애플리케이션이 종료되었습니다."
pause