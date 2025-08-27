@echo off
chcp 65001 > nul

cd /d "%~dp0.."

set /p CAPITAL="Enter initial capital (default 1 billion, skip with Enter): "
if "%CAPITAL%"=="" set CAPITAL=1000000000

.\venv\Scripts\python.exe .\scripts\backtest.py --capital %CAPITAL%

pause > nul