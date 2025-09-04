@echo off
chcp 65001 > nul

cd /d "%~dp0.."

echo ==================================================
echo    🏦 재무 데이터베이스 구축 스크립트 실행 🏦
echo ==================================================

.\venv\Scripts\python.exe .\scripts\build_financial_db.py

echo.
echo 재무 데이터베이스 구축이 완료되었습니다.
echo 아무 키나 눌러 창을 닫으세요.
pause > nul
