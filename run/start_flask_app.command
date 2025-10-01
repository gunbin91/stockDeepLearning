#!/bin/zsh

# 스크립트가 위치한 디렉토리(run)의 상위 디렉토리(프로젝트 루트)로 이동
cd "$(dirname "$0")/.."

# 가상 환경 폴더 이름
VENV_DIR="venv"

# 가상 환경이 없는 경우 생성 및 패키지 설치
if [ ! -d "$VENV_DIR" ]; then
  echo "가상 환경을 찾을 수 없습니다. 새로 생성하고 패키지를 설치합니다."
  # Python3로 가상 환경 생성
  /opt/homebrew/bin/python3.12 -m venv "$VENV_DIR"
  
  # 가상 환경 활성화
  source "$VENV_DIR/bin/activate"
  
  pip install --upgrade pip
  # 기본 요구사항 설치
  pip install -r requirements.txt

  # NLP 관련 패키지 추가 설치
  pip install 'transformers[torch]' sentencepiece

  # 비활성화
  deactivate
  echo "설치가 완료되었습니다. 앱을 시작합니다."
fi

# 가상 환경 활성화
source "$VENV_DIR/bin/activate"

# 최신 패키지 설치
echo "최신 패키지를 설치합니다..."
pip install --upgrade pip
pip install -r requirements.txt
pip install 'transformers[torch]' sentencepiece

# Flask 앱 실행
echo "AI 주식 분석 플랫폼 (Flask)을 시작합니다..."
echo "사용 가능한 포트를 자동으로 찾아서 실행합니다..."
echo ""
echo "디버그 모드를 사용하시겠습니까? (개발 시 유용)"
echo "y: 디버그 모드 ON (코드 변경 시 자동 재시작, 상세 오류 정보)"
echo "n: 일반 모드 (안정적이고 빠름)"
echo ""
printf "디버그 모드를 사용하시겠습니까? (y/n): "
read debug_choice

if [[ $debug_choice == "y" || $debug_choice == "Y" ]]; then
    echo "🔧 디버그 모드로 실행합니다..."
    python flask_app.py --debug
else
    echo "🚀 일반 모드로 실행합니다..."
    python flask_app.py
fi
