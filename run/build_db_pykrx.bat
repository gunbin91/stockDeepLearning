@echo off
chcp 65001 > nul

cd /d "%~dp0.."

echo ==================================================
echo     🐍 pykrx DB 구축 스크립트 실행 🐍
echo ==================================================

.\venv\Scripts\python.exe .\scripts\build_db_pykrx.py

echo.
echo DB 구축이 완료되었습니다.
pause > nul
