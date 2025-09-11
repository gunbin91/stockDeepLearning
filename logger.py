# logger.py

import logging
import os
from datetime import datetime
import sys
import io

class StockAnalysisLogger:
    """주식 분석 시스템용 로거 클래스"""
    
    def __init__(self, name="stock_analysis", log_dir=None):
        self.name = name
        # 프로젝트 루트 디렉토리를 기준으로 로그 디렉토리 설정
        if log_dir is None:
            # 현재 파일의 위치를 기준으로 프로젝트 루트 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # run 디렉토리에서 실행되는 경우를 고려
            if os.path.basename(current_dir) == 'run':
                self.log_dir = os.path.join(current_dir, '..', 'logs')
            else:
                self.log_dir = os.path.join(current_dir, 'logs')
        else:
            self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _is_streamlit_environment(self):
        """Streamlit 환경인지 확인"""
        try:
            # Streamlit이 실행 중인지 확인
            import streamlit as st
            # Streamlit이 실행 중이면 True 반환
            return hasattr(st, '_is_running_with_streamlit')
        except ImportError:
            return False
        except Exception:
            # 다른 예외가 발생해도 Streamlit 환경으로 간주
            return True
    
    def _setup_logger(self):
        """로거 설정"""
        # 로그 디렉토리 생성
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 로거 생성
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 파일 핸들러 설정
        log_file = os.path.join(self.log_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가 (파일 핸들러만 사용)
        self.logger.addHandler(file_handler)
        
        # Streamlit 환경에서는 콘솔 핸들러를 사용하지 않음
        # 파일 로깅만 사용하여 버퍼 분리 문제 방지
        
        # 중복 로그 방지
        self.logger.propagate = False
    
    def info(self, message):
        """정보 로그"""
        try:
            self.logger.info(message)
        except Exception:
            # 로깅 실패 시 무시 (시스템 안정성 우선)
            pass
    
    def warning(self, message):
        """경고 로그"""
        try:
            self.logger.warning(message)
        except Exception:
            # 로깅 실패 시 무시 (시스템 안정성 우선)
            pass
    
    def error(self, message):
        """에러 로그"""
        try:
            self.logger.error(message)
        except Exception:
            # 로깅 실패 시 무시 (시스템 안정성 우선)
            pass
    
    def debug(self, message):
        """디버그 로그"""
        try:
            self.logger.debug(message)
        except Exception:
            # 로깅 실패 시 무시 (시스템 안정성 우선)
            pass
    
    def critical(self, message):
        """치명적 에러 로그"""
        try:
            self.logger.critical(message)
        except Exception:
            # 로깅 실패 시 무시 (시스템 안정성 우선)
            pass

# 전역 로거 인스턴스 (지연 초기화)
logger = None

def _get_logger():
    """로거 인스턴스를 가져오거나 생성"""
    global logger
    if logger is None:
        logger = StockAnalysisLogger()
    return logger

# 편의 함수들
def log_info(message):
    """정보 로그 편의 함수"""
    try:
        _get_logger().info(message)
    except Exception:
        pass

def log_warning(message):
    """경고 로그 편의 함수"""
    try:
        _get_logger().warning(message)
    except Exception:
        pass

def log_error(message):
    """에러 로그 편의 함수"""
    try:
        _get_logger().error(message)
    except Exception:
        pass

def log_debug(message):
    """디버그 로그 편의 함수"""
    try:
        _get_logger().debug(message)
    except Exception:
        pass

def log_critical(message):
    """치명적 에러 로그 편의 함수"""
    try:
        _get_logger().critical(message)
    except Exception:
        pass
