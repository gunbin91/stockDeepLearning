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

REM ---------------------------------------------------------------------------
REM WSL에서 bash가 깨지지 않도록:
REM - BOM 제거
REM - CRLF -> LF 변환
REM ---------------------------------------------------------------------------
python -c "from pathlib import Path; p=Path(r'%SCRIPT_PATH%'); b=p.read_bytes(); b=b[3:] if b.startswith(b'\xef\xbb\xbf') else b; b=b.replace(b'\r\n', b'\n'); p.write_bytes(b)"

REM Execute the shell script using wsl.exe
REM Any arguments passed to this batch file (%*) will be forwarded to the shell script.
REM e.g., start_flask_app.bat --port 5000
REM
REM WSL 기본 사용자로 실행 (conda/환경 활성화는 start_flask.sh 내부에서 자동 처리)
wsl.exe --cd ~ -- bash -lc "bash \"$(wslpath -u '%SCRIPT_PATH%')\" %*"

echo.
echo Flask app has been terminated.
echo.
pause

