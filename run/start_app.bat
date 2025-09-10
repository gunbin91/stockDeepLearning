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

REM 가상 환경 활성화
echo "가상 환경을 활성화합니다..."
CALL "%ROOT_DIR%\venv\Scripts\activate.bat"

REM ==========================================================
REM 수정된 부분: pip 및 빌드 도구 업그레이드
REM ==========================================================
echo "pip, setuptools, wheel을 최신 버전으로 업그레이드합니다..."
python -m pip install --upgrade pip setuptools wheel
REM ==========================================================

REM 패키지 설치
echo "requirements.txt 파일의 패키지를 설치합니다..."
pip install -r "%ROOT_DIR%\requirements.txt"

REM ==========================================================
REM pandas_ta posix module import fix for Windows
REM ==========================================================
echo "Fixing pandas_ta posix import issue..."
SET "VENV_SITE_PACKAGES=%ROOT_DIR%\venv\Lib\site-packages"
SET "ALLIGATOR_FILE=%VENV_SITE_PACKAGES%\pandas_ta\overlap\alligator.py"
IF EXIST "%ALLIGATOR_FILE%" (
    powershell -Command "(Get-Content \"%ALLIGATOR_FILE%\") | Where-Object {$_ -notmatch \"from posix import pread\"} | Set-Content \"%ALLIGATOR_FILE%\""
    IF %ERRORLEVEL% NEQ 0 (
        echo "Failed to fix pandas_ta posix import issue."
    ) ELSE (
        echo "pandas_ta posix import issue fixed."
    )
) ELSE (
    echo "alligator.py not found at %ALLIGATOR_FILE%. Skipping fix."
)
REM ==========================================================

REM Streamlit 앱 실행
echo "Streamlit 애플리케이션을 시작합니다..."
streamlit run "%ROOT_DIR%\app.py"

echo "애플리케이션이 종료되었습니다."
pause