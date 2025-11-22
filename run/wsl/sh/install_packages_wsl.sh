#!/bin/bash

# ============================================================================
# WSL 환경 패키지 설치 스크립트 (requirements.txt 기반)
# ============================================================================
# 이 스크립트는 'rapids-25.10' Conda 환경에
# requirements.txt 파일에 명시된 모든 Python 패키지를 설치합니다.

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

echo "🚀 Conda 환경(rapids-25.10)을 활성화합니다..."
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rapids-25.10

echo "🐍 현재 Python 인터프리터:"
which python

echo "📦 requirements.txt를 사용하여 패키지 설치를 시작합니다..."

# 프로젝트 루트로 이동
# 이 스크립트는 run/wsl/sh/ 에 있으므로, 프로젝트 루트는 세 단계 위
SCRIPT_DIR=$(dirname $(realpath "$0"))
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
cd $PROJECT_ROOT

# pip를 사용하여 requirements.txt 파일에 명시된 모든 패키지를 설치
pip install --upgrade --no-cache-dir -r requirements.txt

echo "✅ 모든 패키지 설치가 완료되었습니다."
echo "   'rapids-25.10' 환경이 requirements.txt와 동기화되었습니다."

