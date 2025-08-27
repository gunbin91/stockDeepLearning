@echo off
chcp 65001 > nul

cd /d "%~dp0.."

set /p TOP_N="Enter the number of top stocks for simulation (default 5, skip with Enter): "
if "%TOP_N%"=="" set TOP_N=5

.\venv\Scripts\python.exe .\scripts\weight_optimizer.py --top_n %TOP_N%

pause > nul