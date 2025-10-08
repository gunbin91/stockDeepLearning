"""
통합 로깅 시스템
===============

이 파일은 프로젝트 전반에 걸쳐 일관된 로깅을 제공합니다.
다양한 로그 레벨과 이모지를 사용하여 가독성을 높입니다.

주요 기능:
- 구조화된 로그 메시지
- 이모지와 색상을 통한 시각적 구분
- 스레드 안전 로깅
- 로그 파일 자동 관리
- 분석 보고서 생성
"""

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
    """
    주식 분석 전용 로거 클래스
    
    주식 분석 프로세스에 특화된 로깅 기능을 제공합니다.
    이모지와 색상을 사용하여 로그의 가독성을 높이고,
    분석 진행 상황을 실시간으로 추적할 수 있습니다.
    """
    
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
        
        # 보고서 형식 로그 시스템
        self._report_sections = {
            'header': '📋 주식 분석 보고서',
            'info': '📅 분석 정보',
            'data_collection': '📊 데이터 수집 현황',
            'processing': '🧮 분석 처리 현황',
            'results': '📈 최종 결과',
            'performance': '⏱️ 성능 정보',
            'files': '💾 저장된 파일'
        }
        
        # 진행률 추적
        self._progress_tracker = {
            'current_step': 0,
            'total_steps': 0,
            'step_name': '',
            'start_time': None,
            'estimated_time': None
        }
        
        # 이모지 대체 시스템 (Windows 호환)
        self._emoji_replacements = {
            '🎉': '[SUCCESS]',
            '✅': '[OK]',
            '⚠️': '[WARN]',
            '🔄': '[PROC]',
            '🌐': '[NET]',
            '📅': '[DATE]',
            '❌': '[ERROR]',
            '🔍': '[SEARCH]',
            '💾': '[SAVE]',
            '📊': '[DATA]',
            '💰': '[PRICE]',
            '📈': '[CHART]',
            '🎯': '[TARGET]',
            '📋': '[LIST]',
            '🔧': '[TOOL]',
            '⚡': '[FAST]',
            '🛡️': '[SAFE]',
            '🎪': '[SHOW]',
            '🏆': '[WIN]',
            '💡': '[IDEA]'
        }
        
    def _get_log_directory(self, log_dir: Optional[str]) -> str:
        """로그 디렉토리 경로 설정 (통일된 경로 사용)"""
        if log_dir is None:
            # path_manager를 사용하여 통일된 경로 사용
            try:
                from path_manager import get_logs_dir
                return str(get_logs_dir())
            except ImportError:
                # path_manager가 없는 경우 기본 경로 사용
                current_dir = os.path.dirname(os.path.abspath(__file__))
                return os.path.join(current_dir, 'logs')
        return log_dir
    
    
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
        
        # 중복 로그 방지
        self.logger.propagate = False
        
    
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
        """메시지 포맷팅 - 이모지 대체 및 사용자 친화적 개선"""
        # 이모지 대체
        formatted_message = self._replace_emojis(message)
        
        # 컨텍스트 정보 추가
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            formatted_message = f"{formatted_message} | {context_str}"
        
        return formatted_message
    
    def _replace_emojis(self, message: str) -> str:
        """이모지를 대체 문자로 변환"""
        for emoji, replacement in self._emoji_replacements.items():
            message = message.replace(emoji, replacement)
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
        """진행률 로그 - 스레드 안전 버전 (주식추천 페이지 방식)"""
        percentage = (current / total * 100) if total > 0 else 0
        progress_message = f"[PROGRESS] {message} ({current:,}/{total:,} - {percentage:.1f}%)"
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            progress_message += f" | {context_str}"
        
        try:
            with self._lock:
                # 주식추천 페이지 방식: \r 사용으로 덮어쓰기 (완전히 동일한 방식)
                sys.stdout.write(progress_message + '\r')
                sys.stdout.flush()
                
                # 완료 시 개행 추가 (주식추천 페이지 방식)
                if current == total:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
        except Exception:
            # 예외 발생 시에도 동일한 방식 적용
            sys.stdout.write(progress_message + '\r')
            sys.stdout.flush()
            if current == total:
                sys.stdout.write('\n')
                sys.stdout.flush()
    
    def log_step(self, step_name: str, status: str = "START", context: Optional[Dict[str, Any]] = None):
        """단계별 로그 - 사용자 친화적"""
        status_messages = {
            'START': f"[START] {step_name} 시작",
            'PROCESSING': f"[PROC] {step_name} 처리 중",
            'COMPLETE': f"[COMPLETE] {step_name} 완료",
            'ERROR': f"[ERROR] {step_name} 실패"
        }
        
        message = status_messages.get(status, f"[INFO] {step_name}")
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            message = f"{message} | {context_str}"
        
        # 이모지 대체 적용
        message = self._replace_emojis(message)
        
        if status == 'ERROR':
            self.error(message)
        elif status == 'COMPLETE':
            self.info(message)
        else:
            self.info(message)
    
    def start_analysis_report(self, analysis_date: str):
        """분석 시작 보고서 헤더"""
        with self._lock:
            self._progress_tracker['start_time'] = time.time()
            self._progress_tracker['current_step'] = 0
            self._progress_tracker['total_steps'] = 6
            
        header = f"""
═══════════════════════════════════════════════════════════════
📋 주식 분석 보고서
═══════════════════════════════════════════════════════════════
📅 분석 정보
   • 분석 기준일: {analysis_date}
   • 분석 대상: KOSPI + KOSDAQ 전체 종목
   • 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════
"""
        self.info(header.strip())
    
    def log_data_collection_status(self, step: str, status: str, details: Dict[str, Any] = None):
        """데이터 수집 현황 로그"""
        if step == "start":
            self.info("📊 데이터 수집 현황")
            self.info("   └─ 종목 목록 수집 중...")
        elif step == "stock_list":
            if status == "complete":
                count = details.get('count', 0) if details else 0
                self.info(f"   └─ 종목 목록: {count:,}개 종목 수집 완료")
        elif step == "financial_data":
            if status == "complete":
                count = details.get('count', 0) if details else 0
                coverage = details.get('coverage', 0) if details else 0
                self.info(f"   └─ 재무 데이터: {count:,}개 기업 ({coverage:.1f}% 커버리지)")
        elif step == "macro_data":
            if status == "complete":
                self.info("   └─ 거시경제 데이터: KOSPI, USD/KRW, VIX 수집 완료")
        elif step == "price_data":
            if status == "start":
                total = details.get('total', 0) if details else 0
                batches = details.get('batches', 0) if details else 0
                self.info(f"   └─ 주가 데이터: {total:,}개 종목 처리 중... ({batches}개 그룹)")
            elif status == "progress":
                current = details.get('current', 0) if details else 0
                total = details.get('total', 0) if details else 0
                percent = details.get('percent', 0) if details else 0
                collected = details.get('collected', 0) if details else 0
                self.info(f"   └─ 주가 데이터 수집: {percent:.1f}% ({collected:,}/{total:,}개 종목)")
            elif status == "complete":
                count = details.get('count', 0) if details else 0
                self.info(f"   └─ 주가 데이터: {count:,}개 종목 수집 완료")
    
    def log_processing_status(self, step: str, status: str, details: Dict[str, Any] = None):
        """분석 처리 현황 로그"""
        if step == "start":
            self.info("🧮 분석 처리 현황")
        elif step == "factor_scoring":
            if status == "complete":
                factors = details.get('factors', 0) if details else 0
                self.info(f"   └─ 팩터 점수 계산: 완료 ({factors}개 팩터)")
        elif step == "ml_prediction":
            if status == "complete":
                self.info("   └─ 머신러닝 예측: 완료")
        elif step == "ensemble":
            if status == "complete":
                avg_score = details.get('avg_score', 0) if details else 0
                max_score = details.get('max_score', 0) if details else 0
                self.info(f"   └─ 앙상블 점수 계산: 완료 (평균: {avg_score:.1f}, 최고: {max_score:.1f})")
    
    def log_final_results(self, results: Dict[str, Any]):
        """최종 결과 로그"""
        self.info("📈 최종 결과")
        top_10 = results.get('top_10_count', 0)
        avg_score = results.get('avg_score', 0)
        max_score = results.get('max_score', 0)
        total_stocks = results.get('total_stocks', 0)
        
        self.info(f"   └─ 상위 10위 종목: {top_10}개")
        self.info(f"   └─ 평균 점수: {avg_score:.1f}점")
        self.info(f"   └─ 최고 점수: {max_score:.1f}점")
        self.info(f"   └─ 분석 대상: {total_stocks:,}개 종목")
    
    def log_performance_info(self, performance: Dict[str, Any]):
        """성능 정보 로그"""
        self.info("⏱️ 성능 정보")
        total_time = performance.get('total_time', 0)
        data_time = performance.get('data_time', 0)
        analysis_time = performance.get('analysis_time', 0)
        
        self.info(f"   └─ 총 소요 시간: {total_time:.0f}초")
        self.info(f"   └─ 데이터 수집: {data_time:.0f}초")
        self.info(f"   └─ 분석 처리: {analysis_time:.0f}초")
    
    def log_saved_files(self, files: Dict[str, str]):
        """저장된 파일 로그"""
        self.info("💾 저장된 파일")
        for file_type, file_path in files.items():
            self.info(f"   └─ {file_type}: {file_path}")
    
    def complete_analysis_report(self, results: Dict[str, Any], performance: Dict[str, Any], files: Dict[str, str]):
        """분석 완료 보고서"""
        end_time = time.time()
        start_time = self._progress_tracker.get('start_time', end_time)
        total_time = end_time - start_time
        
        self.info("═══════════════════════════════════════════════════════════════")
        self.info("🎉 분석 완료 보고서")
        self.info("═══════════════════════════════════════════════════════════════")
        
        # 최종 결과
        self.log_final_results(results)
        
        # 성능 정보
        performance['total_time'] = total_time
        self.log_performance_info(performance)
        
        # 저장된 파일
        self.log_saved_files(files)
        
        self.info("═══════════════════════════════════════════════════════════════")
        self.info(f"[SUCCESS] 주식 분석이 성공적으로 완료되었습니다! (총 {total_time:.0f}초 소요)")
        self.info("═══════════════════════════════════════════════════════════════")
    
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
        # [PROGRESS] 접두사 제거 - progress() 메서드에서 이미 추가됨
        # print() 대신 sys.stdout.write() 사용으로 덮어쓰기 가능
        sys.stdout.write(f"{message} ({current:,}/{total:,} - {percentage:.1f}%) | Logger Error: {e}\r")
        sys.stdout.flush()
        if current == total:
            sys.stdout.write('\n')
            sys.stdout.flush()

def log_step(step_name: str, status: str = "START", context: Optional[Dict[str, Any]] = None):
    """단계별 로그 편의 함수 - 사용자 친화적"""
    try:
        _get_logger().log_step(step_name, status, context)
    except Exception as e:
        print(f"[{status}] {step_name} | Logger Error: {e}")

def log_success(message: str, context: Optional[Dict[str, Any]] = None):
    """성공 로그 편의 함수"""
    try:
        _get_logger().info(f"[SUCCESS] {message}", context)
    except Exception as e:
        print(f"[SUCCESS] {message} | Logger Error: {e}")

def log_start(message: str, context: Optional[Dict[str, Any]] = None):
    """시작 로그 편의 함수"""
    try:
        _get_logger().info(f"[START] {message}", context)
    except Exception as e:
        print(f"[START] {message} | Logger Error: {e}")

def log_complete(message: str, context: Optional[Dict[str, Any]] = None):
    """완료 로그 편의 함수"""
    try:
        _get_logger().info(f"[COMPLETE] {message}", context)
    except Exception as e:
        print(f"[COMPLETE] {message} | Logger Error: {e}")

def start_analysis_report(analysis_date: str):
    """분석 시작 보고서 헤더"""
    try:
        _get_logger().start_analysis_report(analysis_date)
    except Exception as e:
        print(f"[REPORT] Analysis report start failed: {e}")

def log_data_collection_status(step: str, status: str, details: Dict[str, Any] = None):
    """데이터 수집 현황 로그"""
    try:
        _get_logger().log_data_collection_status(step, status, details)
    except Exception as e:
        print(f"[REPORT] Data collection status log failed: {e}")

def log_processing_status(step: str, status: str, details: Dict[str, Any] = None):
    """분석 처리 현황 로그"""
    try:
        _get_logger().log_processing_status(step, status, details)
    except Exception as e:
        print(f"[REPORT] Processing status log failed: {e}")

def log_final_results(results: Dict[str, Any]):
    """최종 결과 로그"""
    try:
        _get_logger().log_final_results(results)
    except Exception as e:
        print(f"[REPORT] Final results log failed: {e}")

def log_performance_info(performance: Dict[str, Any]):
    """성능 정보 로그"""
    try:
        _get_logger().log_performance_info(performance)
    except Exception as e:
        print(f"[REPORT] Performance info log failed: {e}")

def log_saved_files(files: Dict[str, str]):
    """저장된 파일 로그"""
    try:
        _get_logger().log_saved_files(files)
    except Exception as e:
        print(f"[REPORT] Saved files log failed: {e}")

def complete_analysis_report(results: Dict[str, Any], performance: Dict[str, Any], files: Dict[str, str]):
    """분석 완료 보고서"""
    try:
        _get_logger().complete_analysis_report(results, performance, files)
    except Exception as e:
        print(f"[REPORT] Complete analysis report failed: {e}")

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
