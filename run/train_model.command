#!/bin/bash

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

echo "=================================================="
echo "      🚀 주식 예측 모델 학습 스크립트 🚀"
echo "=================================================="
echo "Optuna를 위한 하이퍼파라미터 값을 설정합니다."
echo "입력하지 않고 엔터를 누르면 기본값이 사용됩니다."

# CPU 코어 수 입력받기
MAX_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
echo "
[1/4] 사용할 CPU 코어 수를 입력하세요."
echo "      (사용 가능: $MAX_CORES, 전체 사용은 -1, 기본값: -1)"
read -p "입력: " N_JOBS
if [ -z "$N_JOBS" ]; then
    N_JOBS=-1
fi

# n_iter 입력받기
echo "
[2/4] 몇 개의 파라미터 조합을 테스트할지 횟수를 입력하세요."
echo "      (값이 클수록 오래 걸리지만 더 좋은 모델을 찾을 수 있습니다. 기본값: 10)"
read -p "입력: " N_ITER
if [ -z "$N_ITER" ]; then
    N_ITER=10
fi

# max_depth 입력받기
echo "
[3/4] max_depth 후보 리스트를 입력하세요."
echo "      (예: 10 20 30, 기본값: 10 20 30)"
read -p "입력: " MAX_DEPTH
if [ -z "$MAX_DEPTH" ]; then
    MAX_DEPTH="10 20 30"
fi

# years 입력받기 (학습 데이터 파일이 없을 때만 사용)
echo "
[4/4] 학습 데이터 파일이 없을 때, 최근 몇 년치 데이터로 학습할지 입력하세요."
echo "      (예: 5 = 최근 5년, 전체 데이터는 엔터, 기본값: 전체 데이터)"
read -p "입력: " YEARS
if [ -z "$YEARS" ]; then
    YEARS_ARG=""
else
    YEARS_ARG="--years $YEARS"
fi

# 실행될 최종 명령어 구성
COMMAND="./venv/bin/python ./scripts/train_model.py --n_jobs $N_JOBS --n_iter $N_ITER --max_depth $MAX_DEPTH $YEARS_ARG"

echo "
--------------------------------------------------"
echo "설정이 완료되었습니다. 아래 명령어로 모델 학습을 시작합니다."
echo "$COMMAND"
echo "--------------------------------------------------"

# 최종 명령어 실행
eval $COMMAND

echo ""
echo "학습이 완료되었습니다. 아무 키나 눌러 창을 닫으세요."
read -n 1 -s -r