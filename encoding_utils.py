# encoding_utils.py
"""
크로스 플랫폼 인코딩 유틸리티
윈도우와 맥북 간의 한글 인코딩 문제를 해결하기 위한 공통 모듈
"""

import sys
import io
import os
import locale

def setup_utf8_encoding():
    """
    UTF-8 인코딩을 설정하여 크로스 플랫폼 호환성을 보장합니다.
    윈도우, 맥북, 리눅스 모든 환경에서 한글이 깨지지 않도록 합니다.
    """
    try:
        # 현재 시스템의 기본 인코딩 확인
        system_encoding = locale.getpreferredencoding()
        
        # Streamlit 환경이 아닌 경우에만 stdout/stderr 재설정
        if not os.environ.get('STREAMLIT_SERVER_PORT'):
            # stdout을 UTF-8로 설정
            if hasattr(sys.stdout, 'detach'):
                try:
                    sys.stdout = io.TextIOWrapper(
                        sys.stdout.detach(), 
                        encoding='utf-8',
                        errors='replace'
                    )
                except Exception:
                    # 이미 detach된 경우 무시
                    pass
            
            # stderr를 UTF-8로 설정
            if hasattr(sys.stderr, 'detach'):
                try:
                    sys.stderr = io.TextIOWrapper(
                        sys.stderr.detach(), 
                        encoding='utf-8',
                        errors='replace'
                    )
                except Exception:
                    # 이미 detach된 경우 무시
                    pass
        
        # 환경 변수 설정 (subprocess에서 사용)
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        return True
        
    except Exception as e:
        # 인코딩 설정 실패 시에도 프로그램이 계속 실행되도록 함
        print(f"인코딩 설정 중 오류 발생 (무시됨): {e}")
        return False

def get_safe_encoding():
    """
    안전한 인코딩을 반환합니다.
    """
    try:
        # UTF-8을 우선 시도
        return 'utf-8'
    except Exception:
        # UTF-8 실패 시 시스템 기본 인코딩 사용
        return locale.getpreferredencoding()

def ensure_utf8_string(text):
    """
    문자열이 UTF-8로 안전하게 인코딩되도록 보장합니다.
    """
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return text.decode('cp949')  # 윈도우 기본 인코딩
            except UnicodeDecodeError:
                return text.decode('utf-8', errors='replace')
    return str(text)

# 모듈 임포트 시 자동으로 인코딩 설정
setup_utf8_encoding()
