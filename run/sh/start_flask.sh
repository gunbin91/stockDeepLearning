#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

echo "🚀 Conda 환경(rapids-25.10)을 활성화합니다..."

# PATH에 conda가 없어도 동작하도록 conda.sh 위치를 후보에서 탐색
FOUND_CONDA_SH=""
for CANDIDATE in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "/usr/local/miniconda3/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh"
do
  if [ -f "$CANDIDATE" ]; then
    FOUND_CONDA_SH="$CANDIDATE"
    break
  fi
done

if [ -n "$FOUND_CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$FOUND_CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "❌ conda를 찾을 수 없습니다."
  echo "   - WSL에서 conda 설치 위치($HOME/miniconda3 등)를 확인해주세요."
  echo "   - 또는 PATH에 conda가 잡히도록 설정해주세요."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "rapids-25.10"; then
  conda activate rapids-25.10
else
  echo "❌ conda 환경 'rapids-25.10'을 찾을 수 없습니다."
  echo "   현재 conda env 목록:"
  conda env list
  exit 1
fi

echo "🐍 Python 인터프리터 정보:"
which python

echo "🌐 Flask 웹 애플리케이션을 시작합니다..."
# 이 스크립트는 run/sh/ 에 있으므로, 프로젝트 루트는 두 단계 위
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd "$PROJECT_ROOT"

# flask_app.py 실행
python flask_app.py "$@"

echo "✅ Flask 앱이 종료되었습니다."

