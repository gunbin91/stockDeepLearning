@echo off
chcp 65001 > nul
REM ============================================================================
REM WSL 환경 패키지 설치 스크립트 실행 파일
REM ============================================================================
REM 이 배치 파일은 WSL2 환경에서 install_packages_wsl.sh 스크립트를 실행합니다.
REM requirements.txt에 명시된 모든 패키지를 rapids-25.10 Conda 환경에 설치합니다.

echo.
echo ============================================================================
echo WSL 환경 패키지 설치
echo ============================================================================
echo requirements.txt에 명시된 모든 패키지를 설치합니다.
echo (로그가 이 창에 표시됩니다)
echo.

REM 현재 스크립트의 디렉토리 경로 가져오기
set "SCRIPT_PATH=%~dp0sh\install_packages_wsl.sh"

REM WSL에서 bash 스크립트 실행 (다른 .bat 파일과 동일한 방식)
wsl.exe --cd ~ -u root -- bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate rapids-25.10 && bash $(wslpath -u '%SCRIPT_PATH%')"

REM 종료 코드 확인
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] 패키지 설치 중 오류가 발생했습니다.
    echo 종료 코드: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo 패키지 설치 완료
echo ============================================================================
pause

