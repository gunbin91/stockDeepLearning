#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# Conda 환경 활성화
# source /path/to/conda/bin/activate 대신 conda run 사용을 권장
# 이는 스크립트 실행 환경을 더 안정적으로 만들어 줌
echo "🚀 Conda 환경(rapids-25.10)을 활성화합니다..."
# Conda 초기화 스크립트 경로를 찾아서 실행
# 일반적으로 /root/miniconda3/etc/profile.d/conda.sh 에 위치
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rapids-25.10

echo "🐍 Python 인터프리터 정보:"
which python

echo "🧠 GPU 학습 스크립트를 실행합니다..."
# 스크립트의 실제 위치를 기준으로 경로 설정
# 이 스크립트는 run/sh/ 에 있으므로, 프로젝트 루트는 두 단계 위
SCRIPT_DIR=$(dirname $(realpath "$0"))
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd $PROJECT_ROOT

# python <스크립트 경로> <인자값>
# $@는 이 셸 스크립트에 전달된 모든 인자(n_iter, max_depth 등)를 그대로 python 스크립트에 전달
python scripts/train_gpu_main.py "$@"

echo "✅ 모든 작업이 완료되었습니다."

