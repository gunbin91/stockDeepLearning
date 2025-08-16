#!/bin/bash

# 스크립트가 있는 디렉토리로 이동하여 경로 문제를 방지합니다.
cd "$(dirname "$0")"

echo "=================================================="
echo "      🚀 주식 예측 모델 학습 스크립트 🚀"
echo "=================================================="
echo "RandomizedSearchCV를 위한 하이퍼파라미터 값을 설정합니다."
echo "입력하지 않고 엔터를 누르면 기본값이 사용됩니다."

# CPU 코어 수 입력받기
MAX_CORES=$(sysctl -n hw.ncpu)
echo "
[1/3] 사용할 CPU 코어 수를 입력하세요."
echo "      (사용 가능: 1 ~ $MAX_CORES, 전체 사용은 -1, 기본값: -1)"
read -p "입력: " N_JOBS
if [ -z "$N_JOBS" ]; then
    N_JOBS=-1
fi

# n_iter 입력받기
echo "
[2/3] 몇 개의 파라미터 조합을 테스트할지 횟수를 입력하세요."
echo "      (값이 클수록 오래 걸리지만 더 좋은 모델을 찾을 수 있습니다. 기본값: 10)"
read -p "입력: " N_ITER
if [ -z "$N_ITER" ]; then
    N_ITER=10
fi

# max_depth 입력받기
echo "
[3/3] max_depth 후보 리스트를 입력하세요."
echo "      (예: 10 20 30, 기본값: 10 20 30)"
read -p "입력: " MAX_DEPTH
if [ -z "$MAX_DEPTH" ]; then
    MAX_DEPTH="10 20 30"
fi

# 실행될 최종 명령어 구성
COMMAND="./venv/bin/python train_model.py --n_jobs $N_JOBS --n_iter $N_ITER --max_depth $MAX_DEPTH"

echo "
--------------------------------------------------"
echo "설정이 완료되었습니다. 아래 명령어로 모델 학습을 시작합니다."
echo "$COMMAND"
echo "--------------------------------------------------"

# 최종 명령어 실행
eval $COMMAND

echo "
학습이 완료되었습니다. 아무 키나 눌러 창을 닫으세요."
read -n 1 -s -r