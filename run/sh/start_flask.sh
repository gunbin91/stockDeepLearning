#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

echo "🚀 Conda 환경(rapids-25.10)을 활성화합니다..."
# Conda 초기화 스크립트 경로를 찾아서 실행
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rapids-25.10

echo "🐍 Python 인터프리터 정보:"
which python

echo "🌐 Flask 웹 애플리케이션을 시작합니다..."
# 이 스크립트는 run/sh/ 에 있으므로, 프로젝트 루트는 두 단계 위
SCRIPT_DIR=$(dirname $(realpath "$0"))
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd $PROJECT_ROOT

# flask_app.py 실행
# $@는 이 셸 스크립트에 전달된 모든 인자(port, host 등)를 그대로 python 스크립트에 전달
python flask_app.py "$@"

echo "✅ Flask 앱이 종료되었습니다."

