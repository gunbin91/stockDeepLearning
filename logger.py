# logger.py

import logging
import os
from datetime import datetime
import sys
import io
import traceback
from typing import Optional, Dict, Any

class StockAnalysisLogger:
    """주식 분석 시스템용 로거 클래스 - 최적화된 버전"""
    
    def __init__(self, name="stock_analysis", log_dir=None):
        self.name = name
        self.log_dir = self._get_log_directory(log_dir)
        self.logger = None
        self._setup_logger()
        
        # 로그 메시지 중복 방지를 위한 세트
        self._logged_messages = set()
        
    def _get_log_directory(self, log_dir: Optional[str]) -> str:
        """로그 디렉토리 경로 설정"""
        if log_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.basename(current_dir) == 'run':
                return os.path.join(current_dir, '..', 'logs')
            else:
                return os.path.join(current_dir, 'logs')
        return log_dir
    
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
        """로거 설정 - 최적화된 버전"""
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
        
        # 향상된 포맷터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(detailed_formatter)
        
        # 콘솔 핸들러 설정 (간소화된 포맷)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 콘솔용 간소화된 포맷터
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 중복 로그 방지 및 Streamlit 환경에서의 안정성 향상
        self.logger.propagate = False
        
        # Streamlit 환경에서 로깅 안정성을 위한 추가 설정
        if self._is_streamlit_environment():
            self.logger.disabled = False
    
    def _should_log_message(self, message: str, level: str) -> bool:
        """중복 메시지 방지 및 로깅 여부 결정"""
        message_key = f"{level}:{message}"
        if message_key in self._logged_messages:
            return False
        self._logged_messages.add(message_key)
        return True
    
    def _format_message(self, message: str, level: str, context: Optional[Dict[str, Any]] = None) -> str:
        """메시지 포맷팅"""
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            return f"{message} | {context_str}"
        return message
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
        """정보 로그 - 최적화된 버전"""
        if skip_duplicate and not self._should_log_message(message, "INFO"):
            return
            
        formatted_message = self._format_message(message, "INFO", context)
        try:
            self.logger.info(formatted_message)
        except Exception as e:
            # 로깅 실패 시 기본 출력
            print(f"[INFO] {formatted_message}")
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
        """경고 로그 - 최적화된 버전"""
        if skip_duplicate and not self._should_log_message(message, "WARNING"):
            return
            
        formatted_message = self._format_message(message, "WARNING", context)
        try:
            self.logger.warning(formatted_message)
        except Exception as e:
            print(f"[WARNING] {formatted_message}")
    
    def error(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """에러 로그 - 상세 정보 포함"""
        formatted_message = self._format_message(message, "ERROR", context)
        
        if exception:
            # 예외 정보 포함
            error_details = f"{formatted_message} | Exception: {type(exception).__name__}: {str(exception)}"
            stack_trace = traceback.format_exc()
            full_message = f"{error_details}\nStack Trace:\n{stack_trace}"
        else:
            full_message = formatted_message
            
        try:
            self.logger.error(full_message)
        except Exception:
            print(f"[ERROR] {full_message}")
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """디버그 로그"""
        formatted_message = self._format_message(message, "DEBUG", context)
        try:
            self.logger.debug(formatted_message)
        except Exception:
            print(f"[DEBUG] {formatted_message}")
    
    def critical(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """치명적 에러 로그 - 최대 상세 정보"""
        formatted_message = self._format_message(message, "CRITICAL", context)
        
        if exception:
            error_details = f"{formatted_message} | CRITICAL Exception: {type(exception).__name__}: {str(exception)}"
            stack_trace = traceback.format_exc()
            full_message = f"{error_details}\nStack Trace:\n{stack_trace}"
        else:
            full_message = formatted_message
            
        try:
            self.logger.critical(full_message)
        except Exception:
            print(f"[CRITICAL] {full_message}")
    
    def progress(self, message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
        """진행률 로그 - tqdm과 충돌 방지, 같은 줄에서 업데이트"""
        percentage = (current / total * 100) if total > 0 else 0
        progress_message = f"{message} ({current:,}/{total:,} - {percentage:.1f}%)"
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            progress_message += f" | {context_str}"
            
        try:
            # 같은 줄에서 업데이트 (tqdm 스타일)
            print(f"\r{progress_message}", end='', flush=True)
            # 완료 시에만 개행
            if current == total:
                print()  # 개행 추가
        except Exception:
            print(f"[PROGRESS] {progress_message}")
    
    def clear_duplicate_cache(self):
        """중복 메시지 캐시 초기화"""
        self._logged_messages.clear()

# 전역 로거 인스턴스 (지연 초기화)
logger = None

def _get_logger():
    """로거 인스턴스를 가져오거나 생성"""
    global logger
    if logger is None:
        logger = StockAnalysisLogger()
    return logger

# 편의 함수들 - 최적화된 버전
def log_info(message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
    """정보 로그 편의 함수 - 최적화된 버전"""
    try:
        _get_logger().info(message, context, skip_duplicate)
    except Exception:
        print(f"[INFO] {message}")

def log_warning(message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
    """경고 로그 편의 함수 - 최적화된 버전"""
    try:
        _get_logger().warning(message, context, skip_duplicate)
    except Exception:
        print(f"[WARNING] {message}")

def log_error(message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
    """에러 로그 편의 함수 - 상세 정보 포함"""
    try:
        _get_logger().error(message, exception, context)
    except Exception:
        print(f"[ERROR] {message}")

def log_debug(message: str, context: Optional[Dict[str, Any]] = None):
    """디버그 로그 편의 함수"""
    try:
        _get_logger().debug(message, context)
    except Exception:
        print(f"[DEBUG] {message}")

def log_critical(message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
    """치명적 에러 로그 편의 함수 - 최대 상세 정보"""
    try:
        _get_logger().critical(message, exception, context)
    except Exception:
        print(f"[CRITICAL] {message}")

def log_progress(message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
    """진행률 로그 편의 함수 - tqdm과 충돌 방지"""
    try:
        _get_logger().progress(message, current, total, context)
    except Exception:
        percentage = (current / total * 100) if total > 0 else 0
        print(f"[PROGRESS] {message} ({current:,}/{total:,} - {percentage:.1f}%)")

def clear_log_cache():
    """중복 메시지 캐시 초기화"""
    try:
        _get_logger().clear_duplicate_cache()
    except Exception:
        pass
