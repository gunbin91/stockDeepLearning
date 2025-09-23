# logger.py

import logging
import os
from datetime import datetime
import sys
import io
import traceback
import threading
import queue
import time
from typing import Optional, Dict, Any

class StockAnalysisLogger:
    """주식 분석 시스템용 로거 클래스 - 멀티스레드 안전 버전"""
    
    def __init__(self, name="stock_analysis", log_dir=None):
        self.name = name
        self.log_dir = self._get_log_directory(log_dir)
        self.logger = None
        self._setup_logger()
        
        # 스레드 안전성을 위한 락
        self._lock = threading.RLock()
        
        # 로그 메시지 중복 방지를 위한 세트 (스레드 안전)
        self._logged_messages = set()
        
        # 백그라운드 로깅을 위한 큐와 스레드
        self._log_queue = queue.Queue()
        self._background_thread = None
        self._shutdown_event = threading.Event()
        self._start_background_logging()
        
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
    
    def _start_background_logging(self):
        """백그라운드 로깅 스레드 시작"""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._background_thread = threading.Thread(target=self._background_log_worker, daemon=True)
            self._background_thread.start()
    
    def _background_log_worker(self):
        """백그라운드 로깅 워커 스레드"""
        while not self._shutdown_event.is_set():
            try:
                # 큐에서 로그 메시지 가져오기 (타임아웃 설정)
                log_entry = self._log_queue.get(timeout=1.0)
                if log_entry is None:  # 종료 신호
                    break
                    
                level, message, exception, context = log_entry
                self._write_log_safely(level, message, exception, context)
                self._log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # 백그라운드 로깅 실패 시 기본 출력
                print(f"[LOGGER ERROR] {e}")
                time.sleep(0.1)
    
    def _write_log_safely(self, level: str, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """안전한 로그 쓰기 (스레드 안전)"""
        try:
            with self._lock:
                formatted_message = self._format_message(message, level, context)
                
                if exception:
                    error_details = f"{formatted_message} | Exception: {type(exception).__name__}: {str(exception)}"
                    stack_trace = traceback.format_exc()
                    full_message = f"{error_details}\nStack Trace:\n{stack_trace}"
                else:
                    full_message = formatted_message
                
                # 로그 레벨에 따라 적절한 메서드 호출
                if level == "INFO":
                    self.logger.info(full_message)
                elif level == "WARNING":
                    self.logger.warning(full_message)
                elif level == "ERROR":
                    self.logger.error(full_message)
                elif level == "DEBUG":
                    self.logger.debug(full_message)
                elif level == "CRITICAL":
                    self.logger.critical(full_message)
                else:
                    self.logger.info(full_message)
                    
        except Exception as e:
            # 로깅 실패 시 기본 출력으로 폴백
            print(f"[{level}] {message}")
            if exception:
                print(f"Exception: {exception}")
                print(f"Stack Trace: {traceback.format_exc()}")
    
    def _queue_log(self, level: str, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """로그를 큐에 추가 (비동기)"""
        try:
            self._log_queue.put((level, message, exception, context), timeout=5.0)
        except queue.Full:
            # 큐가 가득 찬 경우 즉시 출력
            print(f"[{level}] {message}")
            if exception:
                print(f"Exception: {exception}")
    
    def shutdown(self):
        """로거 종료"""
        try:
            self._shutdown_event.set()
            self._log_queue.put(None)  # 종료 신호
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)
        except Exception:
            pass
    
    def _should_log_message(self, message: str, level: str) -> bool:
        """중복 메시지 방지 및 로깅 여부 결정 (스레드 안전)"""
        with self._lock:
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
        """정보 로그 - 스레드 안전 버전"""
        if skip_duplicate and not self._should_log_message(message, "INFO"):
            return
        
        # 백그라운드 큐에 추가 (비동기)
        self._queue_log("INFO", message, None, context)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
        """경고 로그 - 스레드 안전 버전"""
        if skip_duplicate and not self._should_log_message(message, "WARNING"):
            return
        
        # 백그라운드 큐에 추가 (비동기)
        self._queue_log("WARNING", message, None, context)
    
    def error(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """에러 로그 - 스레드 안전 버전 (상세 정보 포함)"""
        # 에러는 항상 로깅 (중복 방지 제외)
        self._queue_log("ERROR", message, exception, context)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """디버그 로그 - 스레드 안전 버전"""
        self._queue_log("DEBUG", message, None, context)
    
    def critical(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """치명적 에러 로그 - 스레드 안전 버전 (최대 상세 정보)"""
        # 치명적 에러는 항상 로깅 (중복 방지 제외)
        self._queue_log("CRITICAL", message, exception, context)
    
    def progress(self, message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
        """진행률 로그 - 스레드 안전 버전 (tqdm과 충돌 방지)"""
        percentage = (current / total * 100) if total > 0 else 0
        progress_message = f"{message} ({current:,}/{total:,} - {percentage:.1f}%)"
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            progress_message += f" | {context_str}"
        
        try:
            with self._lock:
                # 같은 줄에서 업데이트 (tqdm 스타일)
                print(f"\r{progress_message}", end='', flush=True)
                # 완료 시에만 개행
                if current == total:
                    print()  # 개행 추가
        except Exception:
            print(f"[PROGRESS] {progress_message}")
    
    def clear_duplicate_cache(self):
        """중복 메시지 캐시 초기화 (스레드 안전)"""
        with self._lock:
            self._logged_messages.clear()

# 전역 로거 인스턴스 (지연 초기화)
logger = None

def _get_logger():
    """로거 인스턴스를 가져오거나 생성"""
    global logger
    if logger is None:
        logger = StockAnalysisLogger()
    return logger

# 편의 함수들 - 스레드 안전 버전
def log_info(message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
    """정보 로그 편의 함수 - 스레드 안전 버전"""
    try:
        _get_logger().info(message, context, skip_duplicate)
    except Exception as e:
        print(f"[INFO] {message} | Logger Error: {e}")

def log_warning(message: str, context: Optional[Dict[str, Any]] = None, skip_duplicate: bool = True):
    """경고 로그 편의 함수 - 스레드 안전 버전"""
    try:
        _get_logger().warning(message, context, skip_duplicate)
    except Exception as e:
        print(f"[WARNING] {message} | Logger Error: {e}")

def log_error(message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
    """에러 로그 편의 함수 - 스레드 안전 버전 (상세 정보 포함)"""
    try:
        _get_logger().error(message, exception, context)
    except Exception as e:
        print(f"[ERROR] {message} | Logger Error: {e}")
        if exception:
            print(f"Original Exception: {exception}")
            print(f"Stack Trace: {traceback.format_exc()}")

def log_debug(message: str, context: Optional[Dict[str, Any]] = None):
    """디버그 로그 편의 함수 - 스레드 안전 버전"""
    try:
        _get_logger().debug(message, context)
    except Exception as e:
        print(f"[DEBUG] {message} | Logger Error: {e}")

def log_critical(message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
    """치명적 에러 로그 편의 함수 - 스레드 안전 버전 (최대 상세 정보)"""
    try:
        _get_logger().critical(message, exception, context)
    except Exception as e:
        print(f"[CRITICAL] {message} | Logger Error: {e}")
        if exception:
            print(f"Original Exception: {exception}")
            print(f"Stack Trace: {traceback.format_exc()}")

def log_progress(message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
    """진행률 로그 편의 함수 - 스레드 안전 버전"""
    try:
        _get_logger().progress(message, current, total, context)
    except Exception as e:
        percentage = (current / total * 100) if total > 0 else 0
        print(f"[PROGRESS] {message} ({current:,}/{total:,} - {percentage:.1f}%) | Logger Error: {e}")

def clear_log_cache():
    """중복 메시지 캐시 초기화 - 스레드 안전 버전"""
    try:
        _get_logger().clear_duplicate_cache()
    except Exception as e:
        print(f"[CACHE] Cache clear failed: {e}")

def shutdown_logger():
    """로거 종료 함수"""
    try:
        _get_logger().shutdown()
    except Exception as e:
        print(f"[SHUTDOWN] Logger shutdown failed: {e}")
