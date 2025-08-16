#!/bin/zsh

# 스크립트가 위치한 디렉토리로 이동
cd "$(dirname "$0")"

# 가상 환경 폴더 이름
VENV_DIR="venv"

# 가상 환경이 없는 경우 생성 및 패키지 설치
if [ ! -d "$VENV_DIR" ]; then
  echo "가상 환경을 찾을 수 없습니다. 새로 생성하고 패키지를 설치합니다."
  # Python3로 가상 환경 생성
  python3 -m venv "$VENV_DIR"
  
  # 가상 환경 활성화
  source "$VENV_DIR/bin/activate"
  
  # 기본 요구사항 설치
  pip install -r requirements.txt
  pip install pandas-ta

  # NLP 관련 패키지 추가 설치 (requirements.txt에 주석 처리되어 있으므로 별도 명시)
  pip install 'transformers[torch]' sentencepiece
  
  # 비활성화
  deactivate
  echo "설치가 완료되었습니다. 앱을 시작합니다."
fi

# 가상 환경 활성화
source "$VENV_DIR/bin/activate"

# 최신 패키지 설치
echo "최신 패키지를 설치합니다..."
pip install -r requirements.txt
pip install pandas-ta
pip install 'transformers[torch]' sentencepiece

# Streamlit 앱 실행
echo "AI 주식 추천 플랫폼을 시작합니다..."
echo "웹 브라우저에서 앱이 열릴 때까지 잠시 기다려주세요."
./venv/bin/streamlit run app.py