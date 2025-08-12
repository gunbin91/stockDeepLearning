<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:b54727daad5473b0bac5feae44d216af23500c6e0ab19d462bb0ae38a0682a73
size 960
=======
#!/bin/zsh

# 스크립트가 위치한 디렉토리로 이동
cd "$(dirname "$0")"

echo "AI 모델 학습을 시작합니다."
echo "이 과정은 약 5~10분 정도 소요될 수 있으며, 인터넷 속도에 따라 달라질 수 있습니다."
echo "학습이 완료되면 '모델을 ... 경로에 저장했습니다.' 메시지가 표시됩니다."
echo "----------------------------------------------------------------"

# 가상 환경의 Python을 사용하여 학습 스크립트 실행
# 스크립트 실행 전 가상환경이 존재하는지 확인
if [ ! -d "venv" ]; then
  echo "[오류] venv 가상환경을 찾을 수 없습니다."
  echo "먼저 ./start_app.command 를 실행하여 가상환경을 생성해주세요."
else
  ./venv/bin/python train_model.py
fi

echo "----------------------------------------------------------------"
echo "모델 학습이 완료되었습니다."
read -p "아무 키나 눌러 창을 닫아주세요..."
>>>>>>> origin/window
