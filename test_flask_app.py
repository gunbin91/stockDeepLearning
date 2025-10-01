#!/usr/bin/env python3
"""
Flask 앱 테스트 스크립트
Flask 앱이 정상적으로 작동하는지 확인합니다.
"""

import os
import sys
import requests
import time
import subprocess
import threading
from datetime import datetime

def test_flask_app():
    """Flask 앱 테스트"""
    print("🚀 Flask 앱 테스트 시작...")
    
    # Flask 앱 프로세스 시작
    print("📡 Flask 앱을 시작합니다...")
    flask_process = subprocess.Popen([
        sys.executable, 'flask_app.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 앱이 시작될 때까지 대기
    time.sleep(5)
    
    try:
        # 기본 페이지 테스트
        print("🔍 기본 페이지 테스트...")
        response = requests.get('http://localhost:5000', timeout=10)
        if response.status_code == 200:
            print("✅ 기본 페이지 로드 성공")
        else:
            print(f"❌ 기본 페이지 로드 실패: {response.status_code}")
            return False
        
        # 주식 추천 페이지 테스트
        print("🔍 주식 추천 페이지 테스트...")
        response = requests.get('http://localhost:5000/', timeout=10)
        if response.status_code == 200:
            print("✅ 주식 추천 페이지 로드 성공")
        else:
            print(f"❌ 주식 추천 페이지 로드 실패: {response.status_code}")
            return False
        
        # 모델 분석 페이지 테스트
        print("🔍 모델 분석 페이지 테스트...")
        response = requests.get('http://localhost:5000/model_analysis', timeout=10)
        if response.status_code == 200:
            print("✅ 모델 분석 페이지 로드 성공")
        else:
            print(f"❌ 모델 분석 페이지 로드 실패: {response.status_code}")
            return False
        
        # 백테스팅 페이지 테스트
        print("🔍 백테스팅 페이지 테스트...")
        response = requests.get('http://localhost:5000/backtest', timeout=10)
        if response.status_code == 200:
            print("✅ 백테스팅 페이지 로드 성공")
        else:
            print(f"❌ 백테스팅 페이지 로드 실패: {response.status_code}")
            return False
        
        print("🎉 모든 테스트 통과!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 네트워크 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False
    finally:
        # Flask 프로세스 종료
        print("🛑 Flask 앱을 종료합니다...")
        flask_process.terminate()
        flask_process.wait()

def check_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인"""
    print("📦 의존성 확인...")
    
    required_packages = [
        'flask', 'flask_socketio', 'pandas', 'numpy', 
        'plotly', 'requests', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (누락)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 모든 의존성이 설치되어 있습니다.")
    return True

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🧪 AI 주식 분석 시스템 - Flask 앱 테스트")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 확인 실패. 테스트를 중단합니다.")
        return False
    
    print("\n" + "=" * 60)
    
    # Flask 앱 테스트
    if test_flask_app():
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("Flask 앱이 정상적으로 작동합니다.")
        return True
    else:
        print("\n❌ 테스트 실패. Flask 앱에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
