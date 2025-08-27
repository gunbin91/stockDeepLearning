#!/bin/bash
# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

echo "=================================================="
echo "     ⚖️  가중치 최적화 스크립트 실행 ⚖️"
echo "=================================================="

# 상위 종목 수 입력받기
echo "시뮬레이션에 사용할 상위 종목 수를 입력하세요 (기본값 5, Enter로 스킵):"
read -p "입력: " TOP_N
if [ -z "$TOP_N" ]; then
    TOP_N=5
fi

# 가상환경 파이썬으로 스크립트 실행
./venv/bin/python ./scripts/weight_optimizer.py --top_n $TOP_N

echo ""
echo "가중치 최적화가 완료되었습니다."
echo "아무 키나 눌러 창을 닫으세요."
read -n 1 -s -r
