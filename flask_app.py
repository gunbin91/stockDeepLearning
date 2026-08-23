"""
AI 기반 주식 분석 시스템 - 웹 애플리케이션
===============================================

이 파일은 주식 분석 시스템의 웹 인터페이스를 제공합니다.
사용자가 웹 브라우저를 통해 주식 분석을 요청하고 결과를 확인할 수 있습니다.

주요 기능:
- 주식 분석 요청 및 결과 표시
- 실시간 분석 진행 상황 모니터링
- 백테스팅 기능
- 종목별 상세 차트 및 데이터 조회
"""

# 경고 필터를 최상단에서 설정 (eventlet.monkey_patch() 전에)
import warnings
import os

# 환경 변수로 pandas 경고 비활성화
os.environ['PYTHONWARNINGS'] = 'ignore::pandas.errors.Pandas4Warning'

# yfinance의 pandas deprecated API 경고 무시 (yfinance 라이브러리 자체 문제)
# 모든 방법으로 필터링 시도 - 가장 강력한 설정
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from pandas.errors import Pandas4Warning
    warnings.simplefilter("ignore", Pandas4Warning)
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except (ImportError, AttributeError):
    pass
# 메시지 기반 필터링 (모든 변형)
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*will be removed.*")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*", category=UserWarning)

# Eventlet 몽키 패치 (경고 필터 설정 후)
# 웹소켓 통신을 위한 비동기 처리 라이브러리
import eventlet
eventlet.monkey_patch()

# eventlet.monkey_patch() 후 경고 필터 재설정 (monkey patch가 경고 시스템을 변경할 수 있음)
# 더 강력한 방법: 모든 경고를 무시
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
try:
    from pandas.errors import Pandas4Warning
    warnings.simplefilter("ignore", Pandas4Warning)
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except (ImportError, AttributeError):
    pass
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*will be removed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# sys.stderr를 리다이렉션하여 yfinance 경고만 필터링 (eventlet.monkey_patch()가 경고 시스템을 우회할 수 있음)
import sys
_original_stderr = sys.stderr
class FilteredStderr:
    def __init__(self, original):
        self.original = original
    def write(self, text):
        # yfinance 관련 경고만 필터링 (다른 중요한 메시지는 유지)
        if text and ("Pandas4Warning" in text or "Timestamp.utcnow" in text or 
                     ("deprecated" in text and "yfinance" in text) or
                     ("will be removed" in text and "Timestamp" in text)):
            return  # 경고 메시지 필터링
        self.original.write(text)
    def flush(self):
        self.original.flush()
    def __getattr__(self, name):
        return getattr(self.original, name)

# 경고만 필터링하고 다른 stderr 출력은 유지
sys.stderr = FilteredStderr(sys.stderr)

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import re
import time
from collections import deque
import math
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, g
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import pandas_ta as ta
import joblib

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pandas 호환성 패치: cuDF가 pandas.api.types.is_interval을 사용하는데 최신 pandas에서는 제거됨
def apply_pandas_compatibility_patch():
    """pandas 최신 버전과 cuDF 호환성을 위한 패치 적용"""
    try:
        import pandas.api.types as pd_types
        # is_interval이 없으면 추가 (cuDF 호환성)
        if not hasattr(pd_types, 'is_interval'):
            # pandas 2.0+에서는 IntervalDtype을 사용하여 체크
            def is_interval(arr):
                """Interval 타입 체크 함수"""
                try:
                    from pandas import IntervalDtype
                    return hasattr(arr, 'dtype') and isinstance(arr.dtype, IntervalDtype)
                except:
                    return False
            pd_types.is_interval = is_interval
    except Exception as e:
        # 패치 적용 실패해도 계속 진행 (cuDF가 직접 처리할 수 있음)
        pass

# scikit-learn 호환성 패치: cuML이 BaseEstimator._get_default_requests를 사용하는데 최신 scikit-learn에서는 제거됨
def apply_sklearn_compatibility_patch():
    """scikit-learn 최신 버전과 cuML 호환성을 위한 패치 적용"""
    try:
        from sklearn.base import BaseEstimator
        # _get_default_requests가 없으면 추가 (cuML 호환성)
        if not hasattr(BaseEstimator, '_get_default_requests'):
            # scikit-learn 1.3+에서는 _get_metadata_request를 사용하거나, 없으면 빈 함수로 대체
            if hasattr(BaseEstimator, '_get_metadata_request'):
                # _get_metadata_request를 _get_default_requests로 별칭 생성
                original_get_metadata_request = BaseEstimator._get_metadata_request
                def _get_default_requests(self, *args, **kwargs):
                    return original_get_metadata_request(self, *args, **kwargs)
                BaseEstimator._get_default_requests = _get_default_requests
            else:
                # 둘 다 없으면 빈 함수로 대체
                def _get_default_requests(self, *args, **kwargs):
                    return {}
                BaseEstimator._get_default_requests = _get_default_requests
    except Exception as e:
        # 패치 적용 실패해도 계속 진행
        pass

# 기존 모듈들 임포트
import data_fetcher
import data_processor
import scoring
import ml_model
import ensemble
from logger import log_info, log_warning, log_error, log_critical
from exceptions import DataFetchError, ModelPredictionError, AnalysisError
from path_manager import path_manager, ensure_all_directories

# =============================================================================
# 가중치(최종점수 조합) 파일 관리 유틸/REST API
# =============================================================================

def _get_weights_file_path() -> str:
    """최종 점수 계산에 사용되는 가중치 파일 경로"""
    return str(path_manager.get_weights_path())

def _load_weights_file():
    """가중치 파일을 로드 (없으면 None 반환)"""
    weights_path = _get_weights_file_path()
    if not os.path.exists(weights_path):
        return None
    with open(weights_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("가중치 파일 형식이 올바르지 않습니다. (dict 형태여야 함)")
    return data

def _atomic_write_json(file_path: str, data: dict):
    """JSON을 임시 파일에 쓴 뒤 원자적으로 교체 (Windows 포함)"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, file_path)

def _validate_and_prepare_weights(payload: dict):
    """
    가중치 payload 검증 및 정규화 처리
    - key: 문자열(1~64)
    - value: 실수, NaN/Inf 불가, 음수 불가
    - normalize=true이면 합이 1이 되도록 정규화
    """
    if payload is None:
        raise ValueError("요청 바디가 비어있습니다.")

    # 프로젝트 호환 팩터(컬럼)만 수정 가능: 추후 팩터 추가 시 이 목록만 확장
    allowed_keys = {'ml_pred_proba', 'lgbm_pred_proba', 'catboost_pred_proba'}

    # 입력 포맷 호환: {weights: {...}, normalize: true/false} 또는 {...} 직접
    if isinstance(payload, dict) and 'weights' in payload and isinstance(payload.get('weights'), dict):
        weights_in = payload.get('weights', {})
        normalize = bool(payload.get('normalize', True))
    elif isinstance(payload, dict):
        weights_in = payload
        normalize = True
    else:
        raise ValueError("가중치 데이터 형식이 올바르지 않습니다.")

    if not isinstance(weights_in, dict):
        raise ValueError("weights는 dict 형태여야 합니다.")
    if len(weights_in) == 0:
        raise ValueError("가중치가 비어있습니다.")

    extra_keys = [k for k in weights_in.keys() if isinstance(k, str) and k.strip() not in allowed_keys]
    if extra_keys:
        raise ValueError(f"프로젝트에서 지원하지 않는 팩터 키가 포함되어 있습니다: {extra_keys}. "
                         f"허용 키: {sorted(list(allowed_keys))}")

    cleaned = {}
    for k, v in weights_in.items():
        if not isinstance(k, str):
            raise ValueError("가중치 key는 문자열이어야 합니다.")
        k = k.strip()
        if len(k) == 0 or len(k) > 64:
            raise ValueError("가중치 key 길이는 1~64자여야 합니다.")
        if k not in allowed_keys:
            raise ValueError(f"지원하지 않는 팩터 키입니다: {k}. 허용 키: {sorted(list(allowed_keys))}")

        try:
            fv = float(v)
        except Exception:
            raise ValueError(f"가중치 값이 숫자가 아닙니다: {k}={v}")

        if np.isnan(fv) or np.isinf(fv):
            raise ValueError(f"가중치 값이 NaN/Inf 입니다: {k}={v}")
        if fv < 0:
            raise ValueError(f"가중치 값은 음수일 수 없습니다: {k}={v}")

        cleaned[k] = fv

    # 누락된 허용 키는 0으로 채워 항상 동일한 구조로 저장
    for k in allowed_keys:
        if k not in cleaned:
            cleaned[k] = 0.0

    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("가중치 합이 0 이하입니다. (하나 이상 0보다 커야 함)")

    if normalize:
        cleaned = {k: (v / total) for k, v in cleaned.items()}

    cleaned = {k: float(np.round(v, 10)) for k, v in cleaned.items()}
    return cleaned

# =============================================================================
# 웹 애플리케이션 초기화
# =============================================================================

# Flask 웹 애플리케이션 생성
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock_analysis_secret_key_2024'

# 실시간 통신을 위한 WebSocket 설정
# 사용자에게 분석 진행 상황을 실시간으로 전달하기 위함
socketio = SocketIO(app, 
                   cors_allowed_origins="*",  # 모든 도메인에서 접근 허용
                   ping_timeout=60,           # 연결 유지 시간 60초
                   ping_interval=25           # 연결 확인 간격 25초
)


# =============================================================================
# API 요청 접수/응답 로깅 (myKiwoom 연동 디버깅용)
# =============================================================================

@app.before_request
def _api_request_log_before():
    try:
        g._req_start_ts = time.time()
        path = request.path or ""
        if path.startswith("/api/") or path.startswith("/v1/") or path == "/health":
            # 너무 시끄러운 엔드포인트는 필요 시 제외 가능
            # 요청 바디는 민감할 수 있어 "키 목록"만 로깅
            body_keys = None
            if request.method in ("POST", "PUT", "PATCH") and request.is_json:
                payload = request.get_json(silent=True) or {}
                if isinstance(payload, dict):
                    body_keys = list(payload.keys())
            msg = f"[API] {request.method} {path} from={request.remote_addr}"
            if body_keys is not None:
                msg += f" body_keys={body_keys}"
            log_info(msg)
    except Exception:
        pass


@app.after_request
def _api_request_log_after(response):
    try:
        path = request.path or ""
        if path.startswith("/api/") or path.startswith("/v1/") or path == "/health":
            elapsed_ms = None
            if hasattr(g, "_req_start_ts"):
                elapsed_ms = int((time.time() - g._req_start_ts) * 1000)
            msg = f"[API] {request.method} {path} -> {response.status_code}"
            if elapsed_ms is not None:
                msg += f" {elapsed_ms}ms"
            log_info(msg)
    except Exception:
        pass
    return response

# Jinja2 필터 추가
@app.template_filter('moment')
def moment_filter(value, format_string='YYYY-MM-DD'):
    """Moment.js 스타일 날짜 포맷팅"""
    if value is None:
        return datetime.now().strftime('%Y-%m-%d')
    if isinstance(value, str):
        return value
    return value.strftime('%Y-%m-%d')

# 설정 (통일된 경로 사용)
# cuML 모델 경로 (우선 사용)
CUML_MODEL_PATH = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
# LightGBM 모델 경로
LGBM_MODEL_PATH = str(path_manager.data_dir / 'lgbm_model_metadata.joblib')
# CatBoost 모델 경로
CATBOOST_MODEL_PATH = str(path_manager.data_dir / 'catboost_model_metadata.joblib')
# 기존 모델 경로 (fallback)
MODEL_PATH = str(path_manager.get_model_path())

# =============================================================================
# 전역 변수 관리
# =============================================================================

# 현재 실행 중인 프로세스들을 추적하는 변수들
current_analysis_process = None  # 주식 분석 프로세스
current_backtest_process = None  # 백테스팅 프로세스

# 플래그를 함수로 관리하여 더 안전하게 처리
def get_analysis_running():
    """분석 실행 상태 확인"""
    return getattr(get_analysis_running, 'value', False)

def set_analysis_running(value):
    """분석 실행 상태 설정"""
    get_analysis_running.value = value

def get_backtest_running():
    """백테스팅 실행 상태 확인"""
    return getattr(get_backtest_running, 'value', False)

def set_backtest_running(value):
    """백테스팅 실행 상태 설정"""
    get_backtest_running.value = value

def reset_flags():
    """플래그 초기화"""
    set_analysis_running(False)
    set_backtest_running(False)

def is_process_running(process):
    """프로세스가 실제로 실행 중인지 확인"""
    if process is None:
        return False
    return process.poll() is None

def cleanup_analysis_process():
    """분석 프로세스 정리"""
    global current_analysis_process
    if current_analysis_process and is_process_running(current_analysis_process):
        try:
            current_analysis_process.terminate()
            current_analysis_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_analysis_process.kill()
            current_analysis_process.wait()
        except Exception:
            pass  # 프로세스가 이미 종료된 경우 무시
    set_analysis_running(False)
    current_analysis_process = None

def is_backtest_process_running(process):
    """백테스팅 프로세스가 실제로 실행 중인지 확인"""
    if process is None:
        return False
    return process.poll() is None

def cleanup_backtest_process():
    """백테스팅 프로세스 정리"""
    global current_backtest_process
    if current_backtest_process and is_backtest_process_running(current_backtest_process):
        try:
            current_backtest_process.terminate()
            current_backtest_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_backtest_process.kill()
            current_backtest_process.wait()
        except Exception:
            pass  # 프로세스가 이미 종료된 경우 무시
    set_backtest_running(False)
    current_backtest_process = None

# =============================================================================
# 유틸리티 함수들 - 데이터 포맷팅 및 처리
# =============================================================================

def format_price_with_change(row):
    """가격과 등락율을 함께 포맷팅"""
    current_price = row.get('현재가')
    if current_price is None or pd.isna(current_price):
        return "N/A"

    # NASDAQ 전용: USD 기준으로 표시 (소수 2자리)
    try:
        price = f"{float(current_price):,.2f}"
    except Exception:
        return "N/A"

    change_percent = row.get('등락율')
    if pd.isna(change_percent): 
        return f"${price}"
    sign = '+' if change_percent > 0 else ''
    formatted_change = f"{sign}{change_percent:.2f}%"
    return f"${price}<br>({formatted_change})"

def format_change_rate(change_percent):
    """등락율 포맷팅"""
    if pd.isna(change_percent):
        return "N/A"
    sign = '+' if change_percent > 0 else ''
    return f"{sign}{change_percent:.2f}%"

def load_cached_analysis_result():
    """분석 결과 로드 (캐시 없이)"""
    # 분석 결과 파일 경로 (data 디렉토리 사용)
    result_path = os.path.join(str(path_manager.data_dir), 'analysis_result.json')
    market_path = os.path.join(str(path_manager.data_dir), 'market_condition.json')
    
    if os.path.exists(result_path) and os.path.exists(market_path):
        try:
            # 기존 분석 결과 로드
            final_df = pd.read_json(result_path, orient='records')
            with open(market_path, 'r', encoding='utf-8') as f:
                market_condition = json.load(f)

            # 날짜 형식 변환
            final_df['date'] = pd.to_datetime(final_df['date'])
            analysis_date = final_df['date'].iloc[0].strftime('%Y년 %m월 %d일')

            # 데이터프레임 후처리
            display_df = final_df.copy()

            # 시장구분이 없으면 기본값으로 채움 (기존 캐시 호환)
            if '시장구분' not in display_df.columns:
                display_df['시장구분'] = 'N/A'

            # 숫자 컬럼 정리 (NaN/문자 혼합 대비)
            for col in ['현재가', '기준일가', '전날종가', 'final_score', 'ml_pred_proba', 'lgbm_pred_proba', 'catboost_pred_proba', '시가총액']:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            
            # NASDAQ 티커는 그대로 사용
            if '종목코드' in display_df.columns:
                display_df['종목코드'] = display_df['종목코드'].astype(str).str.strip()
            
            if 'ml_pred_proba' in display_df.columns:
                display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100
            
            # lgbm_pred_proba가 있으면 백분율로 변환 (0-1 범위를 0-100으로)
            if 'lgbm_pred_proba' in display_df.columns:
                # NaN이 아닌 값만 처리
                mask = display_df['lgbm_pred_proba'].notna()
                if mask.any():
                    # 최대값이 1.0 이하이면 0-1 범위로 가정하고 백분율로 변환
                    max_val = display_df.loc[mask, 'lgbm_pred_proba'].max()
                    if max_val <= 1.0:
                        display_df.loc[mask, 'lgbm_pred_proba'] = display_df.loc[mask, 'lgbm_pred_proba'] * 100
            else:
                # LGBM 컬럼이 없더라도 테이블 정렬을 위해 컬럼 생성
                display_df['lgbm_pred_proba'] = np.nan
            
            # catboost_pred_proba가 있으면 백분율로 변환 (0-1 범위를 0-100으로)
            if 'catboost_pred_proba' in display_df.columns:
                # NaN이 아닌 값만 처리
                mask = display_df['catboost_pred_proba'].notna()
                if mask.any():
                    # 최대값이 1.0 이하이면 0-1 범위로 가정하고 백분율로 변환
                    max_val = display_df.loc[mask, 'catboost_pred_proba'].max()
                    if max_val <= 1.0:
                        display_df.loc[mask, 'catboost_pred_proba'] = display_df.loc[mask, 'catboost_pred_proba'] * 100
            else:
                # CatBoost 컬럼이 없더라도 테이블 정렬을 위해 컬럼 생성
                display_df['catboost_pred_proba'] = np.nan
            
            # 등락율 계산 (0 나누기/NaN 방어)
            if '현재가' in display_df.columns and '기준일가' in display_df.columns:
                denom = display_df['기준일가'].replace(0, np.nan)
                display_df['등락율'] = ((display_df['현재가'] - display_df['기준일가']) / denom) * 100
            else:
                display_df['등락율'] = np.nan
            display_df['현재가(USD)_formatted'] = display_df.apply(format_price_with_change, axis=1)
            
            # 전날 종가 대비 등락율 계산 (증권사 표준 방식)
            if '전날종가' in display_df.columns:
                denom_prev = display_df['전날종가'].replace(0, np.nan)
                display_df['전날종가대비등락율'] = ((display_df['현재가'] - display_df['전날종가']) / denom_prev) * 100
                display_df['등락율(%)'] = display_df['전날종가대비등락율'].apply(format_change_rate)
            else:
                # 전날종가 데이터가 없는 경우 기존 로직 사용 (분석기준일 대비)
                display_df['등락율(%)'] = display_df['등락율'].apply(format_change_rate)

            # NASDAQ: 시가총액은 USD 기준으로 저장됨. 화면에는 억달러(USD 100,000,000) 단위로 표시.
            if '시가총액' in display_df.columns:
                display_df['시가총액'] = pd.to_numeric(display_df['시가총액'], errors='coerce')
                display_df['시가총액'] = display_df['시가총액'] / 100000000

            # NASDAQ 전용: 통화/단위 표기를 USD로 통일
            rename_map = { '현재가': '현재가(USD)', '시가총액': '시가총액(억달러)', 'ml_pred_proba': '상승확률(%)', 'final_score': '최종점수(점)', '기준일가': '기준일가(USD)'}
            display_df.rename(columns=rename_map, inplace=True)
            
            display_columns = [
                '최종순위',
                '시장구분',
                '종목명',
                '종목코드',
                '현재가(USD)_formatted',
                '등락율(%)',
                '기준일가(USD)',
                '최종점수(점)',
                '상승확률(%)',
                'lgbm_pred_proba',
                'catboost_pred_proba',
                '시가총액(억달러)'
            ]
            
            result_df = display_df[[col for col in display_columns if col in display_df.columns] + ['등락율']].rename(columns={'현재가(USD)_formatted': '현재가(USD)'})
            
            return result_df, market_condition, analysis_date
        except Exception as e:
            log_warning(f"기존 분석 결과를 불러오는 데 실패했습니다: {e}")
            return None, None, None
    return None, None, None


# =============================================================================
# myKiwoom 연동용 최소 REST API (health / 실시간 분석 실행)
# =============================================================================

def _load_raw_analysis_json():
    """analysis_result.json 원본을 로드하여 myKiwoom 호환 포맷으로 변환"""
    result_path = os.path.join(str(path_manager.data_dir), 'analysis_result.json')
    if not os.path.exists(result_path):
        return None
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return None

    # analysis_date 추출 (YYYY-MM-DD)
    analysis_date = None
    if isinstance(data[0], dict):
        analysis_date = data[0].get('date')

    # 정렬: 최종순위 있으면 기준, 없으면 그대로 (NaN/문자 혼합 방어)
    try:
        if isinstance(data[0], dict) and '최종순위' in data[0]:
            def _safe_rank(item: dict) -> int:
                try:
                    v = item.get('최종순위', None)
                    if v is None:
                        return 999999
                    fv = float(v)
                    if not math.isfinite(fv):
                        return 999999
                    return int(fv)
                except Exception:
                    return 999999

            data_sorted = sorted(data, key=_safe_rank)
        else:
            data_sorted = data
    except Exception:
        data_sorted = data

    return {
        'analysis_date': analysis_date or datetime.now().strftime('%Y-%m-%d'),
        'total_stocks': len(data_sorted),
        'top_stocks': data_sorted[:20],
        'analysis_result': data_sorted
    }


def _sanitize_json_value(obj):
    """
    JSON 응답 안전화:
    - NaN/Inf -> None (JSON null)
    - dict/list/tuple 재귀 처리
    파일 저장/분석 로직은 건드리지 않고, 응답 직전에만 적용
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _sanitize_json_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_json_value(v) for v in obj]
        if isinstance(obj, tuple):
            return [_sanitize_json_value(v) for v in obj]
        return obj
    except Exception:
        return obj


@app.route('/health', methods=['GET'])
def health():
    """myKiwoom에서 연결 테스트용"""
    return jsonify({
        'success': True,
        'service': 'kiwoomDeepLearning',
        'analysis_running': get_analysis_running(),
        'data_dir': str(path_manager.data_dir)
    })


@app.route('/v1/analysis/run', methods=['POST'])
def api_run_analysis():
    """
    myKiwoom 호출 시 '요청 시점에' 실시간 분석을 수행하고 결과를 반환합니다.
    - 응답 포맷: myKiwoom/src/utils/deep_learning.py 가 기대하는 구조 유지
      { success: bool, data: {analysis_date,total_stocks,top_stocks,analysis_result}, message?: str }
    """
    try:
        if get_analysis_running():
            return jsonify({
                'success': False,
                'message': '분석이 이미 실행 중입니다. 잠시 후 다시 시도해주세요.'
            }), 409

        payload = request.get_json(silent=True) or {}
        analysis_date = payload.get('analysis_date') or datetime.now().strftime('%Y-%m-%d')

        start_ts = time.time()
        set_analysis_running(True)
        try:
            # 기존과 동일하게 스크립트를 실행하여 결과 파일 생성
            script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'run_analysis.py')
            cmd = [sys.executable, script_path, '--date', str(analysis_date)]
            log_info(f"[ANALYSIS] run start date={analysis_date} cmd={' '.join(cmd)}")

            # 실시간 로그 출력: stdout/stderr를 라인 단위로 읽어 서버 로그로 전달
            last_lines = deque(maxlen=200)
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )

            if proc.stdout is not None:
                for line in proc.stdout:
                    msg = (line or "").rstrip("\n")
                    if msg.strip():
                        last_lines.append(msg)
                        log_info(f"[ANALYSIS] {msg}")

            return_code = proc.wait()
            elapsed_ms = int((time.time() - start_ts) * 1000)
            log_info(f"[ANALYSIS] run done rc={return_code} elapsed_ms={elapsed_ms}")

            if return_code != 0:
                tail = "\n".join(list(last_lines)[-50:])
                return jsonify({
                    'success': False,
                    'message': '분석 실행 실패',
                    'error_details': {
                        'return_code': return_code,
                        'elapsed_ms': elapsed_ms,
                        'tail': tail
                    }
                }), 500
        finally:
            set_analysis_running(False)

        # 결과 파일이 "이번 요청 이후에" 생성된 것인지 검증 (이전 결과 오염 방지)
        result_path = os.path.join(str(path_manager.data_dir), 'analysis_result.json')
        if not os.path.exists(result_path):
            return jsonify({
                'success': False,
                'message': '분석 결과 파일을 찾을 수 없습니다. (analysis_result.json)'
            }), 500

        try:
            mtime = os.path.getmtime(result_path)
            if mtime < start_ts:
                return jsonify({
                    'success': False,
                    'message': '분석 결과 파일이 이번 요청에서 생성된 것으로 확인되지 않습니다. (이전 결과일 수 있음)',
                    'error_details': {
                        'analysis_date': analysis_date,
                        'result_mtime': mtime,
                        'request_start_ts': start_ts
                    }
                }), 500
        except Exception as e:
            log_warning(f"[ANALYSIS] 결과 파일 mtime 확인 실패(계속 진행): {e}")

        raw = _load_raw_analysis_json()
        if not raw:
            return jsonify({
                'success': False,
                'message': '분석 결과 파일을 찾을 수 없습니다. (analysis_result.json)'
            }), 500

        # 날짜가 요청과 다른 경우도 차단 (이전 파일/다른 날짜 결과 방지)
        if raw.get('analysis_date') and str(raw.get('analysis_date')) != str(analysis_date):
            return jsonify({
                'success': False,
                'message': '분석 결과 날짜가 요청과 다릅니다. (이전 결과일 수 있음)',
                'error_details': {
                    'requested_analysis_date': analysis_date,
                    'result_analysis_date': raw.get('analysis_date')
                }
            }), 500

        # NaN/Inf 등 비표준 JSON 값 제거 (브라우저 파싱 오류 방지)
        raw = _sanitize_json_value(raw)

        elapsed_ms = int((time.time() - start_ts) * 1000)
        return jsonify({
            'success': True,
            'data': raw,
            'meta': {
                'elapsed_ms': elapsed_ms
            }
        })

    except Exception as e:
        set_analysis_running(False)
        return jsonify({
            'success': False,
            'message': f'분석 API 처리 중 오류: {str(e)}'
        }), 500


@app.route('/v1/analysis/result', methods=['GET'])
def api_get_latest_analysis_result():
    """
    가장 최근 분석 결과 파일(data/analysis_result.json)을 myKiwoom 호환 포맷으로 반환
    - 분석이 오래 걸려 /v1/analysis/run 응답을 기다리기 어려운 경우 폴링용으로 사용
    """
    try:
        raw = _load_raw_analysis_json()
        if not raw:
            return jsonify({
                'success': False,
                'message': '분석 결과 파일을 찾을 수 없습니다. (analysis_result.json)',
                'analysis_running': get_analysis_running()
            }), 404

        return jsonify({
            'success': True,
            'data': raw,
            'analysis_running': get_analysis_running()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'분석 결과 조회 중 오류: {str(e)}',
            'analysis_running': get_analysis_running()
        }), 500

def create_stock_chart(ticker_code, stock_name):
    """지정된 종목의 상세 기술적 분석 차트를 생성합니다."""
    try:
        # NASDAQ 티커는 그대로 사용
        normalized_ticker_code = str(ticker_code).strip()
        padded_ticker_code = normalized_ticker_code

        # 데이터는 2년치를 불러와서 장기 이동평균선 계산에 사용
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # NASDAQ 일봉 데이터 수집 (티커 그대로)
        df = None
        try:
            df = data_fetcher.fetch_daily_ohlcv(normalized_ticker_code, start_date, end_date)
        except:
            df = None
        
        if df is None or df.empty:
            return None

        # 이동평균선(MA) 계산
        df.ta.bbands(length=20, std=2, append=True)
        
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        df['MA240'] = df['Close'].rolling(window=240).mean()

        # 네이버 차트 스타일로 개선된 서브플롯 설정
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.02,  # 간격 줄임
            row_heights=[0.75, 0.25],  # 가격 차트 비율 증가
            subplot_titles=('', '')  # 제목 제거로 공간 확보
        )

        # 캔들스틱 (네이버 스타일 색상)
        fig.add_trace(go.Candlestick(
            x=df.index, 
            open=df['Open'], 
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            name='캔들스틱',
            increasing_line_color='#FF0000',  # 빨간색 (상승)
            decreasing_line_color='#0000FF',  # 파란색 (하락)
            increasing_fillcolor='#FF0000',
            decreasing_fillcolor='#0000FF'
        ), row=1, col=1)
        
        # 이동평균선 (흰 바탕에서 잘 보이는 진한 색상)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='#FF0000', width=1)), row=1, col=1)  # 빨간색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='#FFD700', width=1)), row=1, col=1)  # 진한 노란색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60', line=dict(color='#006400', width=1)), row=1, col=1)  # 진한 초록색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], name='MA120', line=dict(color='#FF1493', width=2)), row=1, col=1)  # 핫핑크 굵게
        fig.add_trace(go.Scatter(x=df.index, y=df['MA240'], name='MA240', line=dict(color='#000000', width=2)), row=1, col=1)  # 검은색 굵게
        
        # 볼린저 밴드 (반투명하게)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0_2.0'], name='BB상단', line=dict(color='#888888', width=0.8, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0_2.0'], name='BB하단', line=dict(color='#888888', width=0.8, dash='dot'), 
                                fill='tonexty', fillcolor='rgba(136,136,136,0.05)'), row=1, col=1)
        
        # 거래량 (네이버 스타일 + 상세 호버 정보)
        colors = ['#FF0000' if row['Close'] > row['Open'] else '#0000FF' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name='거래량', 
            marker_color=colors, 
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>' +
                         '거래량: %{y:,.0f}주<br>' +
                         '거래대금: %{customdata:,.0f}원<br>' +
                         '<extra></extra>',
            customdata=df['Volume'] * df['Close']  # 거래대금 계산
        ), row=2, col=1)
        
        # 초기 차트 기간 6개월 및 모든 휴장일 공백 제거
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        missing_dates = full_date_range.difference(df.index)

        six_months_ago = df.index.max() - timedelta(days=183) # 약 6개월
        fig.update_xaxes(
            range=[six_months_ago, df.index.max()],
            rangebreaks=[
                dict(values=missing_dates)  # 주말 및 공휴일을 모두 제외
            ]
        )

        # 네이버 차트 스타일 레이아웃 (인터랙티브 기능 개선)
        fig.update_layout(
            title=dict(
                text=f'<b>{stock_name} ({padded_ticker_code})</b>',
                x=0.5,
                font=dict(size=16, color='#333333')
            ),
            height=600,  # 차트 높이 증가
            margin=dict(l=50, r=50, t=60, b=50),  # 여백 최적화
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.01, 
                xanchor="right", 
                x=1,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            # 인터랙티브 기능 개선
            hovermode='x unified',  # X축 기준 통합 호버
            dragmode='pan',  # 기본 드래그 모드를 팬으로 설정
            selectdirection='d'  # 선택 방향 설정 (d=diagonal)
        )
        
        # Y축 설정 개선
        fig.update_yaxes(
            title_text="가격 (USD)", 
            row=1, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        fig.update_yaxes(
            title_text="거래량", 
            row=2, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        
        # X축 설정 개선
        fig.update_xaxes(
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        return fig
    except Exception as e:
        log_error(f"차트 생성 중 오류 발생: {e}")
        return None

# =============================================================================
# 웹 페이지 라우트 정의 - 사용자가 접근할 수 있는 페이지들
# =============================================================================

@app.route('/')
def index():
    """메인 페이지 - 주식 추천 시스템"""
    # 기존 분석 결과가 있으면 로드
    analysis_result, market_condition, analysis_date = load_cached_analysis_result()
    
    # 현재 날짜를 템플릿에 전달
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                         analysis_result=analysis_result,
                         market_condition=market_condition,
                         analysis_date=analysis_date,
                         current_date=current_date)

@app.route('/model_analysis')
def model_analysis():
    """학습 모델 분석 페이지"""
    model_info = None
    lgbm_model_info = None
    catboost_model_info = None
    error = None
    
    try:
        # 메모리 사용량 확인
        import psutil
        import gc
        memory_usage = psutil.virtual_memory()
        if memory_usage.percent > 85:
            raise MemoryError(f"메모리 사용량이 높습니다: {memory_usage.percent:.1f}%")
        
        # --- 1. RF (cuML) 모델 로드 ---
        model_path = None
        metadata_path = None
        is_cuml_model = False
        
        if os.path.exists(CUML_MODEL_PATH):
            model_path = CUML_MODEL_PATH
            metadata_path = str(path_manager.data_dir / 'cuml_ensemble_model_metadata.joblib')
            is_cuml_model = True
        elif os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
            metadata_path = str(path_manager.data_dir / 'model_metadata.joblib')
            is_cuml_model = False
        
        # RF 모델 정보 로드 (기존 로직)
        if model_path:
            # 호환성 패치 적용 (cuDF/cuML 로드 전에 실행)
            apply_pandas_compatibility_patch()
            apply_sklearn_compatibility_patch()
            
            # scikit-learn 버전 불일치 경고 억제 (모델 로드 시)
            import warnings
            with warnings.catch_warnings():
                try:
                    from sklearn.exceptions import InconsistentVersionWarning
                    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                except (ImportError, AttributeError):
                    pass
                
                # 메타데이터 파일 우선 로드
                if os.path.exists(metadata_path):
                    try:
                        model_data = joblib.load(metadata_path)
                        log_info("RF 메타데이터 로드 완료")
                    except Exception as e:
                        log_warning(f"RF 메타데이터 로드 실패: {e}. 모델 파일 시도")
                        model_data = joblib.load(model_path)
                else:
                    model_data = joblib.load(model_path)

            # 모델 정보 추출 (RF)
            # 메타데이터 파일인지 확인 ('model' 키가 없으면 메타데이터 파일)
            is_metadata_file = 'model' not in model_data and 'models' not in model_data
            
            # cuML 모델인지 확인
            is_cuml_single_model = is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'single_model'
            is_cuml_ensemble_model = is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'mini_batch_ensemble'
            
            if is_cuml_ensemble_model or is_cuml_single_model or is_metadata_file:
                # 메타데이터 기반 로드
                features = model_data.get('features', [])
                training_config = model_data.get('training_config', {})
                optimization_results = model_data.get('optimization_results', {})
                best_params = model_data.get('best_params', {})
                if not best_params and optimization_results:
                    best_params = optimization_results.get('best_params', {})
                parameter_explanations = model_data.get('parameter_explanations', {})
                feature_importances = model_data.get('feature_importances', None)
                permutation_importances = model_data.get('permutation_importances', None)
                
                # 모델 타입 문자열
                if is_cuml_ensemble_model:
                    n_models = training_config.get('n_final_models') or training_config.get('n_mini_batches', 5)
                    model_type_str = f"cuML 앙상블 ({n_models}개 모델)"
                elif is_cuml_single_model:
                    model_type_str = "cuML 단일 모델"
                else:
                    model_type_str = "기존 모델 (메타데이터)"

                model_info = {
                    'model_type': model_type_str,
                    'model_path': model_path,
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                    'oob_score': None,
                    'features': features,
                    'feature_importances': feature_importances,
                    'permutation_importances': permutation_importances,
                    'params': best_params,
                    'training_config': training_config,
                    'optimization_results': optimization_results,
                    'parameter_explanations': parameter_explanations
                }
            
            # 객체 정리
            del model_data
            gc.collect()

        # --- 2. LGBM 모델 로드 ---
        if os.path.exists(LGBM_MODEL_PATH):
            try:
                # scikit-learn 버전 불일치 경고 억제 (모델 로드 시)
                import warnings
                with warnings.catch_warnings():
                    try:
                        from sklearn.exceptions import InconsistentVersionWarning
                        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                    except (ImportError, AttributeError):
                        pass
                    lgbm_data = joblib.load(LGBM_MODEL_PATH)
                log_info("LGBM 메타데이터 로드 완료")
                
                lgbm_model_info = {
                    'model_type': 'LightGBM (GPU)',
                    'model_path': str(path_manager.data_dir / 'lgbm_model.txt'),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(LGBM_MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S'),
                    'features': lgbm_data.get('features', []),
                    'params': lgbm_data.get('best_params', {}),
                    'best_score': lgbm_data.get('best_score', None),
                    'best_iteration': lgbm_data.get('best_iteration', None),
                    # Optuna 탐색 결과(있으면 표시)
                    'optimization_results': lgbm_data.get('optimization_results', None),
                    'training_config': lgbm_data.get('training_config', None),
                }
                
                # 기본 중요도: 모델에서 직접 가져오기
                try:
                    import lightgbm as lgb
                    lgbm_model_file = path_manager.data_dir / 'lgbm_model.txt'
                    if lgbm_model_file.exists():
                        model = lgb.Booster(model_file=str(lgbm_model_file))
                        feature_importance = model.feature_importance(importance_type='gain')
                        features = lgbm_model_info['features']
                        if len(features) == len(feature_importance):
                            # 튜플 리스트로 변환 [(feature, importance), ...]
                            importance_list = list(zip(features, feature_importance.tolist()))
                            # 중요도 높은 순서대로 정렬 (내림차순)
                            importance_list.sort(key=lambda x: x[1], reverse=True)
                            lgbm_model_info['default_importances'] = importance_list
                        else:
                            lgbm_model_info['default_importances'] = None
                    else:
                        lgbm_model_info['default_importances'] = None
                except Exception as e:
                    log_warning(f"LGBM 기본 중요도 로드 실패: {e}")
                    lgbm_model_info['default_importances'] = None
                
                # SHAP 및 Permutation 중요도: CSV 파일에서 로드
                fi_path = path_manager.data_dir / 'lgbm_feature_importance.csv'
                if fi_path.exists():
                    try:
                        fi_df = pd.read_csv(fi_path)
                        
                        # SHAP 중요도
                        if 'shap_importance' in fi_df.columns:
                            shap_df = fi_df[['feature', 'shap_importance']].sort_values(by='shap_importance', ascending=False)
                            lgbm_model_info['shap_importances'] = list(shap_df.itertuples(index=False, name=None))
                        else:
                            lgbm_model_info['shap_importances'] = None
                            
                        # 순열 중요도 (추후 템플릿에 추가 가능)
                        if 'permutation_importance' in fi_df.columns:
                            perm_df = fi_df[['feature', 'permutation_importance']].sort_values(by='permutation_importance', ascending=False)
                            lgbm_model_info['permutation_importances'] = list(perm_df.itertuples(index=False, name=None))
                        else:
                            lgbm_model_info['permutation_importances'] = None
                            
                    except Exception as e:
                        log_warning(f"LGBM 중요도 CSV 파일 처리 중 오류: {e}")
                        lgbm_model_info['shap_importances'] = None
                        lgbm_model_info['permutation_importances'] = None
                else:
                    lgbm_model_info['shap_importances'] = None
                    lgbm_model_info['permutation_importances'] = None
                
                # 기본적으로 기본 중요도를 feature_importances로 설정 (하위 호환성)
                lgbm_model_info['feature_importances'] = lgbm_model_info.get('default_importances') or lgbm_model_info.get('shap_importances')
                    
            except Exception as e:
                log_warning(f"LGBM 정보 로드 실패: {e}")
                lgbm_model_info = None

        # --- 3. CatBoost 모델 로드 ---
        if os.path.exists(CATBOOST_MODEL_PATH):
            try:
                # scikit-learn 버전 불일치 경고 억제 (모델 로드 시)
                import warnings
                with warnings.catch_warnings():
                    try:
                        from sklearn.exceptions import InconsistentVersionWarning
                        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                    except (ImportError, AttributeError):
                        pass
                    catboost_data = joblib.load(CATBOOST_MODEL_PATH)
                log_info("CatBoost 메타데이터 로드 완료")
                
                catboost_model_info = {
                    'model_type': 'CatBoost (GPU)',
                    'model_path': str(path_manager.data_dir / 'catboost_model.cbm'),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(CATBOOST_MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S'),
                    'features': catboost_data.get('features', []),
                    'params': catboost_data.get('best_params', {}),
                    'best_score': catboost_data.get('best_score', None),
                    'best_iteration': catboost_data.get('best_iteration', None),
                    # Optuna 탐색 결과(있으면 표시)
                    'optimization_results': catboost_data.get('optimization_results', None),
                    'training_config': catboost_data.get('training_config', None),
                    'n_trials': catboost_data.get('n_trials', None),
                    'trials_completed': catboost_data.get('trials_completed', None),
                }
                
                # 기본 중요도: 모델에서 직접 가져오기
                try:
                    from catboost import CatBoostClassifier
                    catboost_model_file = path_manager.data_dir / 'catboost_model.cbm'
                    if catboost_model_file.exists():
                        model = CatBoostClassifier()
                        model.load_model(str(catboost_model_file))
                        feature_importance = model.get_feature_importance()
                        features = catboost_model_info['features']
                        if len(features) == len(feature_importance):
                            # 튜플 리스트로 변환 [(feature, importance), ...]
                            importance_list = list(zip(features, feature_importance.tolist()))
                            # 중요도 높은 순서대로 정렬 (내림차순)
                            importance_list.sort(key=lambda x: x[1], reverse=True)
                            catboost_model_info['default_importances'] = importance_list
                        else:
                            catboost_model_info['default_importances'] = None
                    else:
                        catboost_model_info['default_importances'] = None
                except Exception as e:
                    log_warning(f"CatBoost 기본 중요도 로드 실패: {e}")
                    catboost_model_info['default_importances'] = None
                
                # SHAP 및 Permutation 중요도: CSV 파일에서 로드
                fi_path = path_manager.data_dir / 'catboost_feature_importance.csv'
                if fi_path.exists():
                    try:
                        fi_df = pd.read_csv(fi_path)
                        
                        # SHAP 중요도
                        if 'shap_importance' in fi_df.columns:
                            shap_df = fi_df[['feature', 'shap_importance']].sort_values(by='shap_importance', ascending=False)
                            catboost_model_info['shap_importances'] = list(shap_df.itertuples(index=False, name=None))
                        else:
                            catboost_model_info['shap_importances'] = None
                            
                        # 순열 중요도
                        if 'permutation_importance' in fi_df.columns:
                            perm_df = fi_df[['feature', 'permutation_importance']].sort_values(by='permutation_importance', ascending=False)
                            catboost_model_info['permutation_importances'] = list(perm_df.itertuples(index=False, name=None))
                        else:
                            catboost_model_info['permutation_importances'] = None
                            
                    except Exception as e:
                        log_warning(f"CatBoost 중요도 CSV 파일 처리 중 오류: {e}")
                        catboost_model_info['shap_importances'] = None
                        catboost_model_info['permutation_importances'] = None
                else:
                    catboost_model_info['shap_importances'] = None
                    catboost_model_info['permutation_importances'] = None
                
                # 기본적으로 기본 중요도를 feature_importances로 설정 (하위 호환성)
                catboost_model_info['feature_importances'] = catboost_model_info.get('default_importances') or catboost_model_info.get('shap_importances')
                    
            except Exception as e:
                log_warning(f"CatBoost 정보 로드 실패: {e}")
                catboost_model_info = None

        if model_info is None and lgbm_model_info is None and catboost_model_info is None:
             error = "학습된 모델 정보를 찾을 수 없습니다."

    except FileNotFoundError:
        error = "모델 파일을 찾을 수 없습니다. 먼저 모델을 학습해주세요."
    except MemoryError as e:
        error = f"메모리 부족으로 모델을 로드할 수 없습니다: {str(e)}"
    except Exception as e:
        error = f"모델 로드 중 오류: {str(e)}"
        import traceback
        log_error(f"모델 분석 페이지 오류: {traceback.format_exc()}")
    
    return render_template('model_analysis.html', model_info=model_info, lgbm_model_info=lgbm_model_info, catboost_model_info=catboost_model_info, error=error)

@app.route('/backtest')
def backtest():
    """백테스팅 페이지"""
    # 기존 백테스팅 리포트가 있는지 확인 (JSON만)
    json_report_path = str(path_manager.data_dir / 'backtest_report.json')
    has_report = os.path.exists(json_report_path)
    
    return render_template('backtest.html', has_report=has_report)

# =============================================================================
# API 엔드포인트들 - 웹 페이지와 백엔드 간의 데이터 통신
# =============================================================================

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """분석 시작 API"""
    try:
        data = request.get_json()
        analysis_date = data.get('analysis_date')
        
        if not analysis_date:
            return jsonify({'error': '분석 기준일이 필요합니다.'}), 400
        
        # 기존 프로세스 정리
        global current_analysis_process
        if current_analysis_process and is_process_running(current_analysis_process):
            cleanup_analysis_process()
        
        # 분석 중복 실행 방지 (개선된 체크)
        if get_analysis_running():
            return jsonify({'error': '이미 분석이 실행 중입니다. 완료될 때까지 기다려주세요.'}), 400
        
        # 분석 실행 중 플래그 설정
        set_analysis_running(True)
        
        # 분석 프로세스 시작 (subprocess.Popen 방식)
        def run_analysis_process():
            global current_analysis_process
            process = None
            try:
                # 분석 스크립트 실행
                command = [
                    sys.executable, '-u', 
                    os.path.join(os.path.dirname(__file__), 'scripts', 'run_analysis.py'),
                    '--date', analysis_date
                ]
                
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LANG'] = 'ko_KR.UTF-8'
                env['LC_ALL'] = 'ko_KR.UTF-8'
                # yfinance 경고 억제를 위한 환경 변수 추가
                env['PYTHONWARNINGS'] = 'ignore::pandas.errors.Pandas4Warning'
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env
                )
                
                # 현재 분석 프로세스 저장
                current_analysis_process = process
                
                # 실시간 로그 전송
                PROGRESS_REGEX = re.compile(r'\[PROGRESS\].*\(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                last_log_time = time.time()
                process_timeout = 7200  # 2시간 타임아웃
                no_output_timeout = 600  # 10분간 출력이 없으면 타임아웃
                # 완료 이벤트가 안 오는 문제 방지:
                # - 분석 스크립트가 "성공 로그"까지 출력했지만 프로세스가 종료되지 않는 경우(잔여 스레드 등)
                # - 이 경우 팝업이 닫히지 않으므로, 성공 문구를 감지하면 선제적으로 완료 이벤트를 emit
                emitted_complete = False
                
                # 프로세스 상태 모니터링을 위한 변수
                process_start_time = time.time()
                last_activity_time = time.time()
                
                while True:
                    # 프로세스가 종료되었는지 확인
                    if process.poll() is not None:
                        # 프로세스가 종료됨
                        break
                    
                    # 전체 타임아웃 확인 (2시간)
                    elapsed_time = time.time() - process_start_time
                    if elapsed_time > process_timeout:
                        error_msg = f"분석 프로세스가 타임아웃되었습니다 (최대 {process_timeout//60}분 초과)"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                        return
                    
                    # 출력이 없는 시간 확인 (10분)
                    if time.time() - last_activity_time > no_output_timeout:
                        error_msg = f"분석 프로세스가 응답하지 않습니다 (10분간 출력 없음)"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                        return
                    
                    # stdout에서 읽기 (논블로킹)
                    try:
                        # Windows에서는 select가 작동하지 않으므로 직접 읽기 시도
                        line = process.stdout.readline()
                        if line:
                            last_activity_time = time.time()
                            last_log_time = time.time()
                            
                            # 터미널 출력 (진행률 메시지만 특별 처리)
                            if PROGRESS_REGEX.search(line):
                                # 진행률 메시지만 덮어쓰기 처리
                                sys.stdout.write(line.strip() + '\r')
                                last_line_was_tqdm = True
                            else:
                                if last_line_was_tqdm:
                                    sys.stdout.write('\n')
                                sys.stdout.write(line)
                                last_line_was_tqdm = False
                            sys.stdout.flush()
                            
                            # 이모지 대체
                            emoji_replacements = {
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
                            
                            # 이모지 대체 적용
                            processed_line = line.strip()
                            for emoji, replacement in emoji_replacements.items():
                                processed_line = processed_line.replace(emoji, replacement)
                            
                            # 진행률 메시지 감지 및 접두사 추가
                            if PROGRESS_REGEX.search(line):
                                message = f"[PROGRESS] {processed_line}"
                            else:
                                message = processed_line
                            
                            # yfinance 경고 메시지 필터링
                            if 'Pandas4Warning' in message or 'Timestamp.utcnow' in message or ('deprecated' in message and 'yfinance' in message):
                                continue  # 경고 메시지는 전송하지 않음
                            
                            # WebSocket으로 로그 전송 (터미널과 동시)
                            socketio.emit('analysis_log', {'message': message})

                            # 성공 문구 감지 시: 프로세스 종료를 기다리지 않고 완료 이벤트를 먼저 전송
                            # (클라이언트 팝업이 영원히 안 닫히는 현상 방지)
                            if (not emitted_complete) and ("주식 분석이 성공적으로 완료되었습니다" in processed_line):
                                emitted_complete = True
                                socketio.emit('analysis_complete', {'success': True})
                        else:
                            # 더 이상 읽을 데이터가 없으면 잠시 대기
                            socketio.sleep(0.1)
                    except Exception as e:
                        # 읽기 오류 발생 시 로그 출력
                        error_msg = f"프로세스 출력 읽기 오류: {e}"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        socketio.sleep(0.1)
                
                # 프로세스 종료 대기 (타임아웃 10초)
                try:
                    process.stdout.close()
                    return_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # 타임아웃 발생 시 강제 종료
                    process.kill()
                    return_code = process.wait()
                    error_msg = "분석 프로세스가 정상적으로 종료되지 않아 강제 종료되었습니다."
                    socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                    socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                    return
                
                if return_code == 0:
                    # 이미 성공 문구로 완료 이벤트를 보냈을 수 있음
                    if not emitted_complete:
                        socketio.emit('analysis_complete', {'success': True})
                else:
                    error_msg = f'분석 실행 중 오류가 발생했습니다. (종료 코드: {return_code})'
                    socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                    
            except Exception as e:
                socketio.emit('analysis_complete', {'success': False, 'error': str(e)})
            finally:
                # 분석 완료 시 플래그 해제 및 프로세스 초기화
                set_analysis_running(False)
                current_analysis_process = None
        
        # 백그라운드에서 분석 실행 (SocketIO 컨텍스트 유지)
        socketio.start_background_task(run_analysis_process)
        
        return jsonify({'message': '분석이 시작되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시 플래그 해제
        set_analysis_running(False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """분석 중단 API"""
    try:
        global current_analysis_process
        
        # 실제 프로세스 상태 확인
        if current_analysis_process is None or not is_process_running(current_analysis_process):
            # 이미 종료된 프로세스인 경우 플래그만 해제
            cleanup_analysis_process()
            return jsonify({'message': '실행 중인 분석이 없습니다.'})
        
        # 프로세스 종료
        current_analysis_process.terminate()
        try:
            current_analysis_process.wait(timeout=5)  # 타임아웃 증가
        except subprocess.TimeoutExpired:
            current_analysis_process.kill()
            current_analysis_process.wait()
        
        # 플래그 해제 및 프로세스 초기화
        cleanup_analysis_process()
        
        # WebSocket으로 중단 알림
        socketio.emit('analysis_complete', {'success': False, 'error': '사용자에 의해 분석이 중단되었습니다.'})
        
        return jsonify({'message': '분석이 중단되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시에도 플래그 강제 해제
        cleanup_analysis_process()
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_backtest', methods=['POST'])
def start_backtest():
    """백테스팅 시작 API"""
    try:
        data = request.get_json()
        
        # 백테스팅 파라미터 검증
        required_params = ['capital', 'max_hold', 'take_profit', 'stop_loss', 'top_n', 'buy_universe', 'transaction_fee']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'{param} 파라미터가 필요합니다.'}), 400
        
        # 날짜 파라미터 검증 (선택적이지만 제공되면 유효성 검증)
        from datetime import datetime as dt
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if start_date and end_date:
            try:
                start_dt = dt.strptime(start_date, '%Y-%m-%d')
                end_dt = dt.strptime(end_date, '%Y-%m-%d')
                today = dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                if end_dt > today:
                    return jsonify({'error': '종료일은 오늘 이전이어야 합니다.'}), 400
                
                if start_dt >= end_dt:
                    return jsonify({'error': '시작일은 종료일보다 이전이어야 합니다.'}), 400
            except ValueError as e:
                return jsonify({'error': f'날짜 형식이 올바르지 않습니다: {str(e)}'}), 400
        
        # 기존 백테스팅 프로세스 정리
        global current_backtest_process
        if current_backtest_process and is_backtest_process_running(current_backtest_process):
            cleanup_backtest_process()
        
        # 백테스팅 중복 실행 방지
        if get_backtest_running():
            return jsonify({'error': '이미 백테스팅이 실행 중입니다. 완료될 때까지 기다려주세요.'}), 400
        
        # 백테스팅 실행 중 플래그 설정
        set_backtest_running(True)
        
        def run_backtest_process():
            global current_backtest_process
            process = None
            try:
                # 백테스팅 스크립트 실행
                command = [
                    sys.executable, '-u',
                    os.path.join(os.path.dirname(__file__), 'scripts', 'backtest.py'),
                    '--capital', str(data['capital']),
                    '--max-hold', str(data['max_hold']),
                    '--take-profit', str(data['take_profit']),
                    '--stop-loss', str(data['stop_loss']),
                    '--top-n', str(data['top_n']),
                    '--buy-universe', str(data['buy_universe']),
                    '--fee', str(data['transaction_fee'])
                ]
                
                # 날짜 파라미터 추가 (제공된 경우)
                if start_date:
                    command.extend(['--start-date', start_date])
                if end_date:
                    command.extend(['--end-date', end_date])
                
                # 캐시 사용 파라미터 추가
                if data.get('use_cache', False):
                    command.append('--use-cache')
                
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LANG'] = 'ko_KR.UTF-8'
                env['LC_ALL'] = 'ko_KR.UTF-8'
                # yfinance 경고 억제를 위한 환경 변수 추가
                env['PYTHONWARNINGS'] = 'ignore::pandas.errors.Pandas4Warning'
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env
                )
                
                # 현재 백테스팅 프로세스 저장
                current_backtest_process = process
                
                # 실시간 로그 전송
                PROGRESS_REGEX = re.compile(r'\[PROGRESS\].*\(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                
                for line in iter(process.stdout.readline, ''):
                    # 터미널 출력 (진행률 메시지만 특별 처리)
                    if PROGRESS_REGEX.search(line):
                        # 진행률 메시지만 덮어쓰기 처리
                        sys.stdout.write(line.strip() + '\r')
                        last_line_was_tqdm = True
                    else:
                        if last_line_was_tqdm:
                            sys.stdout.write('\n')
                        sys.stdout.write(line)
                        last_line_was_tqdm = False
                    sys.stdout.flush()
                    
                    # 이모지 대체
                    emoji_replacements = {
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
                    
                    # 이모지 대체 적용
                    processed_line = line.strip()
                    for emoji, replacement in emoji_replacements.items():
                        processed_line = processed_line.replace(emoji, replacement)
                    
                    # 진행률 메시지 감지 및 접두사 추가
                    if PROGRESS_REGEX.search(line):
                        message = f"[PROGRESS] {processed_line}"
                    else:
                        message = processed_line
                    
                    # WebSocket으로 로그 전송 (터미널과 동시)
                    socketio.emit('backtest_log', {'message': message})
                    
                    # 실행 양보 (이벤트 루프가 블로킹되지 않도록)
                    socketio.sleep(0.01)
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    socketio.emit('backtest_complete', {'success': True})
                else:
                    socketio.emit('backtest_complete', {'success': False, 'error': '백테스팅 실행 중 오류가 발생했습니다.'})
                    
            except Exception as e:
                socketio.emit('backtest_complete', {'success': False, 'error': str(e)})
            finally:
                # 백테스팅 완료 시 플래그 해제 및 프로세스 초기화
                set_backtest_running(False)
                current_backtest_process = None
        
        # 백그라운드에서 백테스팅 실행 (SocketIO 컨텍스트 유지)
        socketio.start_background_task(run_backtest_process)
        
        return jsonify({'message': '백테스팅이 시작되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시 플래그 해제
        set_backtest_running(False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_backtest', methods=['POST'])
def stop_backtest():
    """백테스팅 중단 API"""
    try:
        global current_backtest_process
        
        # 실제 프로세스 상태 확인
        if current_backtest_process is None or not is_backtest_process_running(current_backtest_process):
            # 이미 종료된 프로세스인 경우 플래그만 해제
            cleanup_backtest_process()
            return jsonify({'message': '실행 중인 백테스팅이 없습니다.'})
        
        # 프로세스 종료
        current_backtest_process.terminate()
        try:
            current_backtest_process.wait(timeout=5)  # 타임아웃 증가
        except subprocess.TimeoutExpired:
            current_backtest_process.kill()
            current_backtest_process.wait()
        
        # 플래그 해제 및 프로세스 초기화
        cleanup_backtest_process()
        
        # WebSocket으로 중단 알림
        socketio.emit('backtest_complete', {'success': False, 'error': '사용자에 의해 백테스팅이 중단되었습니다.'})
        
        return jsonify({'message': '백테스팅이 중단되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시에도 플래그 강제 해제
        cleanup_backtest_process()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_chart/<ticker_code>')
def get_stock_chart(ticker_code):
    """종목 차트 데이터 API"""
    try:
        # 종목명 가져오기 (캐시 우선, 없으면 티커 사용)
        normalized_ticker = str(ticker_code).strip()
        # 기존 파일 호환: run_analysis에서 zfill(6)로 저장된 티커(예: 00TRSG)도 들어올 수 있음
        candidates = [normalized_ticker]
        if normalized_ticker.lstrip('0') and normalized_ticker.lstrip('0') != normalized_ticker:
            candidates.append(normalized_ticker.lstrip('0'))

        stock_name = None
        resolved_ticker = normalized_ticker
        try:
            cached_features_path = os.path.join(str(path_manager.data_dir), 'cached_features.json')
            if os.path.exists(cached_features_path):
                cached_df = pd.read_json(cached_features_path, orient='records', dtype={'종목코드': str})
                norm_series = cached_df['종목코드'].astype(str).str.strip()
                for cand in candidates:
                    row = cached_df[norm_series == cand]
                    if not row.empty:
                        stock_name = row.iloc[0].get('종목명')
                        resolved_ticker = cand
                        break
        except Exception:
            stock_name = None

        if not stock_name:
            # 종목명 조회 실패해도 차트 생성 시도 (티커 그대로 사용)
            stock_name = resolved_ticker
        
        # 차트 생성
        fig = create_stock_chart(resolved_ticker, stock_name)
        if fig is None:
            return jsonify({'error': '차트를 생성할 수 없습니다.'}), 500
        
        # Plotly JSON으로 변환
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'chart': chart_json})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_features/<ticker_code>')
def get_stock_features(ticker_code):
    """종목 피처 데이터 API"""
    try:
        cached_features_path = os.path.join(str(path_manager.data_dir), 'cached_features.json')
        if not os.path.exists(cached_features_path):
            return jsonify({'error': '피처 데이터를 찾을 수 없습니다.'}), 404
        
        cached_features_df = pd.read_json(cached_features_path, orient='records', dtype={'종목코드': str})
        normalized_ticker = str(ticker_code).strip()
        candidates = [normalized_ticker]
        if normalized_ticker.lstrip('0') and normalized_ticker.lstrip('0') != normalized_ticker:
            candidates.append(normalized_ticker.lstrip('0'))

        norm_series = cached_features_df['종목코드'].astype(str).str.strip()
        selected_stock_features = cached_features_df[norm_series.isin(candidates)]
        
        if selected_stock_features.empty:
            return jsonify({'error': '해당 종목의 피처 데이터를 찾을 수 없습니다.'}), 404
        
        # 학습 모델에서 사용하는 피처 리스트 (공통 26개)
        model_features = [
            'log_mktcap',
            '52주_신고가_비율',
            'ADX_14',
            'disparity_120',
            'disparity_240',
            'disparity_20',
            'IXIC_disparity_20',
            'Trend_Pullback_Score',
            'Position_Range_60',
            'MA20_Slope',
            'MA120_Slope',
            'MA240_Slope',
            'IXIC_MA20_Slope',
            'RVOL',
            '시총 회전율(1W)',
            '시총 회전율(3M)',
            'RSI_Signal_Oscillator',
            'ATRr_5',
            'ATRr_20',
            'ATRr_60',
            'HV_Volatility_5',
            'HV_Volatility_20',
            'HV_Volatility_60',
            'VWAP_Disparity_5',
            'Max_Drawdown_20',
            '등락율(5D)',
        ]
        
        # 등락율 계산 등 다른 부분에 영향있는 필수 피처 (표시용)
        essential_features = ['시가', '종가', '현재가', '기준일가', '전날종가', '종목명', '시가총액']
        
        # 학습 모델 피처 + 필수 피처만 필터링
        allowed_features = set(model_features + essential_features)
        
        # 피처 데이터 정리
        display_features = selected_stock_features.drop(columns=['종목코드', 'date'], errors='ignore')
        
        # 학습 모델에서 사용하는 피처만 필터링
        filtered_features = {col: display_features[col] for col in display_features.columns if col in allowed_features}
        
        # 객체를 문자열로 변환
        features_dict = {}
        for column, series in filtered_features.items():
            value = series.iloc[0]
            if pd.isna(value):
                features_dict[column] = "N/A"
            elif isinstance(value, (dict, list)):
                # 중첩된 객체를 JSON 문자열로 변환
                features_dict[column] = json.dumps(value, ensure_ascii=False)
            else:
                # 일반 값은 문자열로 변환
                features_dict[column] = str(value)
        
        return jsonify({'features': features_dict})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest_report')
def get_backtest_report():
    """백테스팅 리포트 JSON 데이터 제공"""
    json_report_path = str(path_manager.data_dir / 'backtest_report.json')
    
    if os.path.exists(json_report_path):
        try:
            with open(json_report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            return jsonify(report_data)
        except Exception as e:
            log_error(f"백테스팅 리포트 로드 실패: {e}")
            return jsonify({'error': '리포트를 로드할 수 없습니다.'}), 500
    else:
        return jsonify({'error': '백테스팅 리포트를 찾을 수 없습니다.'}), 404

@app.route('/api/delete_backtest_cache', methods=['POST'])
def delete_backtest_cache():
    """백테스팅 캐시 파일 삭제 API (단일 파일)"""
    try:
        # 백테스팅 스크립트의 캐시 삭제 함수 사용
        import sys
        import importlib.util
        
        backtest_path = os.path.join(os.path.dirname(__file__), 'scripts', 'backtest.py')
        spec = importlib.util.spec_from_file_location("backtest_module", backtest_path)
        backtest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backtest_module)
        
        deleted = backtest_module.delete_backtest_cache()
        
        if deleted:
            return jsonify({'message': '캐시 파일이 삭제되었습니다.'})
        else:
            return jsonify({'error': '삭제할 캐시 파일이 없습니다.'}), 404
            
    except Exception as e:
        log_error(f"백테스팅 캐시 삭제 실패: {e}")
        import traceback
        log_error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """favicon.ico 서빙"""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/api/status')
def get_status():
    """서버 상태 확인 API (디버그용)"""
    return jsonify({
        'analysis_running': get_analysis_running(),
        'backtest_running': get_backtest_running(),
        'process_exists': current_analysis_process is not None,
        'process_running': is_process_running(current_analysis_process) if current_analysis_process else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analysis_status')
def get_analysis_status():
    """분석 상태 상세 확인 API"""
    return jsonify({
        'analysis_running': get_analysis_running(),
        'process_exists': current_analysis_process is not None,
        'process_running': is_process_running(current_analysis_process) if current_analysis_process else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/backtest_status')
def get_backtest_status():
    """백테스팅 상태 상세 확인 API"""
    return jsonify({
        'backtest_running': get_backtest_running(),
        'process_exists': current_backtest_process is not None,
        'process_running': is_backtest_process_running(current_backtest_process) if current_backtest_process else False,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/weights', methods=['GET'])
def get_weights():
    """가중치 파일 조회 API (주식추천/백테스트 공용)"""
    try:
        # ensemble.py 기본값과 동일한 디폴트
        default_weights = {
            'ml_pred_proba': 1.0,
        }

        weights_path = _get_weights_file_path()
        file_exists = os.path.exists(weights_path)
        file_weights = _load_weights_file() if file_exists else None

        merged = default_weights.copy()
        if isinstance(file_weights, dict):
            # volatility_score는 제거되었으므로 필터링
            file_weights = {k: v for k, v in file_weights.items() if k != 'volatility_score'}
            merged.update(file_weights)

        allowed_keys = ['ml_pred_proba', 'lgbm_pred_proba', 'catboost_pred_proba']
        # UI 혼란 방지: 허용된 키만 반환/표시
        merged_filtered = {k: merged.get(k, 0.0) for k in allowed_keys}
        file_weights_filtered = {k: (file_weights or {}).get(k, 0.0) for k in allowed_keys}

        return jsonify({
            'file_path': weights_path,
            'file_exists': file_exists,
            'allowed_keys': allowed_keys,
            'default_weights': default_weights,
            'file_weights': file_weights_filtered,
            'weights': merged_filtered
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weights', methods=['POST'])
def save_weights():
    """가중치 파일 저장 API (정규화 기본 적용)"""
    try:
        payload = request.get_json(silent=True)
        normalized_weights = _validate_and_prepare_weights(payload)

        weights_path = _get_weights_file_path()
        _atomic_write_json(weights_path, normalized_weights)

        return jsonify({
            'success': True,
            'file_path': weights_path,
            'weights': normalized_weights
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/optimize_weights', methods=['POST'])
def optimize_weights():
    """가중치 최적화 API - 여러 가중치 조합을 테스트하여 최적 조합 찾기"""
    try:
        # 백테스팅이 이미 실행 중이면 거부
        if get_backtest_running():
            return jsonify({'error': '이미 백테스팅이 실행 중입니다. 완료될 때까지 기다려주세요.'}), 400
        
        # 백테스팅 파라미터 수집 (request context가 있는 동안)
        data = request.get_json() or {}
        
        # 날짜 기본값 설정 (None이면 기본값 사용)
        from datetime import datetime, timedelta
        default_end_date = datetime.now().strftime('%Y-%m-%d')
        default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        backtest_params = {
            'capital': data.get('capital', 10000000),
            'max_hold': data.get('max_hold', 7),
            'take_profit': data.get('take_profit', 8.0),
            'stop_loss': data.get('stop_loss', 8.0),
            'top_n': data.get('top_n', 5),
            'buy_universe': data.get('buy_universe', 20),
            'transaction_fee': data.get('transaction_fee', 0.015),
            'start_date': data.get('start_date') or default_start_date,
            'end_date': data.get('end_date') or default_end_date,
            'use_cache': True  # 캐시 사용 강제
        }
        
        # 날짜 유효성 검증
        if not backtest_params['start_date'] or not backtest_params['end_date']:
            return jsonify({'error': '시작일과 종료일이 필요합니다.'}), 400
        
        # request context에서 sid 가져오기 (백그라운드 태스크에서 사용)
        # HTTP 요청에서는 sid가 없으므로 None으로 설정
        client_sid = None
        try:
            # WebSocket 연결이 있는 경우에만 sid 가져오기
            from flask import has_request_context
            if has_request_context():
                # request.sid는 WebSocket 연결에서만 사용 가능
                # HTTP 요청에서는 AttributeError 발생하므로 try-except로 처리
                try:
                    client_sid = request.sid
                except AttributeError:
                    # HTTP 요청인 경우 sid가 없음 (정상)
                    client_sid = None
        except Exception:
            client_sid = None
        
        # 최적화를 비동기로 실행
        def run_optimization():
            try:
                # Flask application context 설정 (백그라운드 태스크에서 필요)
                with app.app_context():
                    optimize_weights_sync(backtest_params, client_sid=client_sid)
            except Exception as e:
                log_error(f"가중치 최적화 실패: {e}")
                with app.app_context():
                    socketio.emit('optimize_complete', {'success': False, 'error': str(e)}, room=client_sid if client_sid else None)
        
        # 백그라운드에서 실행
        socketio.start_background_task(run_optimization)
        
        return jsonify({'message': '가중치 최적화가 시작되었습니다.'})
        
    except Exception as e:
        log_error(f"가중치 최적화 시작 실패: {e}")
        return jsonify({'error': str(e)}), 500

def optimize_weights_sync(backtest_params, client_sid=None):
    """가중치 최적화 동기 실행 함수
    
    Args:
        backtest_params: 백테스팅 파라미터 딕셔너리
        client_sid: WebSocket 클라이언트 세션 ID (None이면 모든 클라이언트에게 브로드캐스트)
    """
    try:
        # backtest_params는 이미 파라미터로 받았으므로 그대로 사용
        
        # 가중치 조합 생성 (0.0~1.0, 0.1 단위, 합=1.0)
        combinations = []
        for ml in range(11):  # 0.0, 0.1, ..., 1.0
            ml_weight = round(ml * 0.1, 1)
            for lgbm in range(11):
                lgbm_weight = round(lgbm * 0.1, 1)
                catboost_weight = round(1.0 - ml_weight - lgbm_weight, 1)
                
                # 합이 1.0이고 모든 가중치가 0 이상인 조합만
                if abs(ml_weight + lgbm_weight + catboost_weight - 1.0) < 0.01 and catboost_weight >= 0:
                    combinations.append({
                        'ml_pred_proba': ml_weight,
                        'lgbm_pred_proba': lgbm_weight,
                        'catboost_pred_proba': catboost_weight
                    })
        
        log_info(f"가중치 최적화 시작: {len(combinations)}개 조합 테스트")
        
        # 기존 가중치 백업
        weights_path = _get_weights_file_path()
        backup_weights = None
        if os.path.exists(weights_path):
            backup_weights = _load_weights_file()
        
        results = []
        total_combinations = len(combinations)
        
        # 진행률 전송을 위한 함수
        def emit_progress(current, total, current_weights, status_msg=None):
            progress_pct = int((current / total) * 100)
            # client_sid가 있으면 해당 클라이언트에게만, 없으면 모든 클라이언트에게 브로드캐스트
            progress_data = {
                'current': current,
                'total': total,
                'progress': progress_pct,
                'current_weights': current_weights
            }
            if status_msg:
                progress_data['status'] = status_msg
            socketio.emit('optimize_progress', progress_data, room=client_sid if client_sid else None)
            # 이벤트 루프에 양보 (WebSocket 전송 보장)
            socketio.sleep(0)
        
        # 초기 진행률 전송 (0%)
        emit_progress(0, total_combinations, None, "가중치 최적화 준비 중...")
        
        # 종목 리스트를 한 번만 수집하여 재사용 (가중치 최적화 성능 최적화)
        import data_processor
        shared_stock_list = None
        try:
            emit_progress(0, total_combinations, None, "종목 리스트 수집 중...")
            shared_stock_list = data_processor.fetch_stock_list()
            if shared_stock_list is not None and not shared_stock_list.empty:
                log_info(f"가중치 최적화용 종목 리스트 수집 완료: {len(shared_stock_list)}개 종목")
                emit_progress(0, total_combinations, None, f"종목 리스트 수집 완료 ({len(shared_stock_list)}개 종목)")
            else:
                log_warning("가중치 최적화용 종목 리스트 수집 실패 (각 백테스팅마다 수집 시도)")
                emit_progress(0, total_combinations, None, "종목 리스트 수집 실패 (백테스팅 중 수집 시도)")
        except Exception as e:
            log_warning(f"가중치 최적화용 종목 리스트 수집 실패: {e} (각 백테스팅마다 수집 시도)")
            emit_progress(0, total_combinations, None, f"종목 리스트 수집 실패: {str(e)[:50]}")
        
        # 각 조합 테스트
        first_error = None
        for idx, weights in enumerate(combinations, 1):
            try:
                # 진행률 전송 (각 조합 시작 시)
                emit_progress(idx - 1, total_combinations, weights, f"가중치 조합 {idx}/{total_combinations} 테스트 중...")
                
                # 가중치 파일 저장
                _atomic_write_json(weights_path, weights)
                
                log_info(f"  [{idx}/{total_combinations}] 가중치 {weights} 테스트 시작...")
                
                # 백테스팅 실행 전 진행률 업데이트
                emit_progress(idx - 1, total_combinations, weights, f"백테스팅 실행 중... ({idx}/{total_combinations})")
                
                # 백테스팅 실행 (직접 함수 호출)
                import scripts.backtest as backtest_module
                backtest_result = backtest_module.run_final_backtest(
                    initial_capital=backtest_params['capital'],
                    max_hold_period=backtest_params['max_hold'],
                    take_profit_pct=backtest_params['take_profit'],
                    stop_loss_pct=backtest_params['stop_loss'],
                    top_n=backtest_params['top_n'],
                    buy_universe_rank=backtest_params['buy_universe'],
                    transaction_fee_rate=backtest_params['transaction_fee'],
                    start_date=backtest_params.get('start_date'),
                    end_date=backtest_params.get('end_date'),
                    use_cache=True,
                    shutdown_logger_after=False,  # 최적화 중에는 logger 종료하지 않음
                    skip_report=True,  # 가중치 최적화에서는 리포트 생성 스킵
                    shared_stock_list=shared_stock_list  # 재사용할 종목 리스트 전달
                )
                
                # 백테스팅 완료 후 진행률 업데이트 (백테스팅이 오래 걸릴 수 있으므로)
                emit_progress(idx, total_combinations, weights, f"백테스팅 완료 ({idx}/{total_combinations})")
                
                # 결과 수집
                if backtest_result and isinstance(backtest_result, dict) and 'total_return' in backtest_result:
                    results.append({
                        'weights': weights,
                        'total_return': float(backtest_result.get('total_return', 0)),
                        'win_rate': float(backtest_result.get('win_rate', 0)),
                        'annual_return': float(backtest_result.get('annual_return', 0)),
                        'sharpe_ratio': float(backtest_result.get('sharpe_ratio', 0)),
                        'mdd': float(backtest_result.get('mdd', 0)),
                        'final_asset': float(backtest_result.get('final_asset', 0))
                    })
                    log_info(f"  [{idx}/{total_combinations}] 가중치 {weights} - 총수익률: {backtest_result.get('total_return', 0)*100:.2f}%")
                else:
                    error_msg = f"백테스팅 결과가 올바르지 않음: {type(backtest_result)}"
                    if backtest_result:
                        error_msg += f", keys: {list(backtest_result.keys()) if isinstance(backtest_result, dict) else 'N/A'}"
                    log_warning(f"  [{idx}/{total_combinations}] 가중치 {weights} - {error_msg}")
                    if first_error is None:
                        first_error = error_msg
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                log_error(f"  [{idx}/{total_combinations}] 가중치 {weights} 테스트 실패: {e}")
                log_error(f"  상세 에러:\n{error_detail}")
                if first_error is None:
                    first_error = f"{str(e)}\n{error_detail[:500]}"  # 처음 500자만
                continue
        
        # 최종 진행률 전송 (100%)
        emit_progress(total_combinations, total_combinations, None, "가중치 최적화 완료")
        
        # 백업된 가중치 복원
        if backup_weights:
            _atomic_write_json(weights_path, backup_weights)
        
        # 모든 테스트 완료 후 logger 종료
        try:
            from logger import shutdown_logger
            shutdown_logger()
        except Exception:
            pass
        
        # 총수익률 기준 정렬
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        # Top 20 반환
        top_results = results[:20]
        
        log_info(f"가중치 최적화 완료: {len(results)}개 성공, Top 20 반환")
        
        response_data = {
            'success': True,
            'total_tested': total_combinations,
            'successful': len(results),
            'results': top_results
        }
        
        # 에러가 있었으면 첫 번째 에러 메시지 포함
        if first_error and len(results) == 0:
            response_data['error_message'] = f"모든 테스트 실패. 첫 번째 에러: {first_error[:500]}"
            log_error(f"가중치 최적화 실패: 모든 조합 테스트 실패. 첫 번째 에러: {first_error[:500]}")
        
        # WebSocket으로 결과 전송
        socketio.emit('optimize_complete', response_data, room=client_sid if client_sid else None)
        
    except Exception as e:
        log_error(f"가중치 최적화 동기 실행 실패: {e}")
        import traceback
        error_detail = traceback.format_exc()
        socketio.emit('optimize_complete', {
            'success': False,
            'error': f"{str(e)}\n{error_detail[:500]}"
        }, room=client_sid if client_sid else None)

@app.route('/api/feature_correlation/calculate', methods=['POST'])
def calculate_feature_correlation():
    """피처 상관관계 계산 API"""
    try:
        import glob
        from pathlib import Path
        
        # 학습 데이터 경로 확인
        data_path = os.path.expanduser("~/stock_data/processed_feather")
        feather_files = glob.glob(os.path.join(data_path, "*.feather"))
        
        # 학습 모델에서 사용하는 피처 리스트 (공통 26개)
        model_features = [
            'log_mktcap',
            '52주_신고가_비율',
            'ADX_14',
            'disparity_120',
            'disparity_240',
            'disparity_20',
            'IXIC_disparity_20',
            'Trend_Pullback_Score',
            'Position_Range_60',
            'MA20_Slope',
            'MA120_Slope',
            'MA240_Slope',
            'IXIC_MA20_Slope',
            'RVOL',
            '시총 회전율(1W)',
            '시총 회전율(3M)',
            'RSI_Signal_Oscillator',
            'ATRr_5',
            'ATRr_20',
            'ATRr_60',
            'HV_Volatility_5',
            'HV_Volatility_20',
            'HV_Volatility_60',
            'VWAP_Disparity_5',
            'Max_Drawdown_20',
            '등락율(5D)',
        ]
        
        data_source = None
        correlation_matrix = None
        feature_data = None
        
        # 1단계: 학습 데이터 파일이 있으면 사용
        if feather_files and len(feather_files) > 0:
            log_info(f"학습 데이터 파일 사용: {len(feather_files)}개 파일")
            data_source = "training_data"
            
            # 모든 feather 파일 로드 및 합치기
            dfs = []
            loaded_count = 0
            for feather_file in feather_files[:500]:  # 최대 500개 파일만 로드 (메모리 절약)
                try:
                    df = pd.read_feather(feather_file)
                    # model_features에 있는 피처만 선택
                    available_features = [f for f in model_features if f in df.columns]
                    if available_features:
                        df_subset = df[available_features].copy()
                        dfs.append(df_subset)
                        loaded_count += 1
                except Exception as e:
                    log_warning(f"Feather 파일 로드 실패: {feather_file}, {e}")
                    continue
            
            if dfs:
                feature_data = pd.concat(dfs, ignore_index=True)
                log_info(f"학습 데이터 로드 완료: {len(feature_data):,}행, {loaded_count}개 파일")
            else:
                log_warning("학습 데이터 파일을 로드할 수 없습니다. 실시간 데이터 수집으로 전환합니다.")
                data_source = None
        
        # 2단계: 학습 데이터가 없으면 1년치 실시간 데이터 수집
        if data_source is None or feature_data is None or feature_data.empty:
            log_info("1년치 실시간 데이터 수집 시작...")
            data_source = "realtime_data"
            
            # 1년 전 날짜 계산
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # data_processor를 통해 데이터 수집
            feature_data = data_processor.get_preprocessed_data(start_date, end_date, skip_factor_scores=True)
            
            if feature_data is None or feature_data.empty:
                return jsonify({'error': '데이터를 수집할 수 없습니다.'}), 500
            
            # model_features에 있는 피처만 선택
            available_features = [f for f in model_features if f in feature_data.columns]
            if available_features:
                feature_data = feature_data[available_features].copy()
            else:
                return jsonify({'error': '필요한 피처가 데이터에 없습니다.'}), 500
            
            log_info(f"실시간 데이터 수집 완료: {len(feature_data):,}행")
        
        # 3단계: 상관관계 계산
        log_info("상관관계 계산 중...")
        
        # 숫자형 컬럼만 선택
        numeric_cols = feature_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data_numeric = feature_data[numeric_cols].copy()
        
        # 결측치 처리 (중앙값으로 대체)
        feature_data_numeric = feature_data_numeric.fillna(feature_data_numeric.median())
        
        # 상관관계 계산
        correlation_matrix = feature_data_numeric.corr()
        
        # NaN이나 Inf 값 처리
        correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], np.nan)
        correlation_matrix = correlation_matrix.fillna(0)
        
        # 상관관계 행렬을 JSON으로 변환 가능한 형태로 변환
        correlation_dict = correlation_matrix.to_dict()
        
        # 피처 리스트 (행/열 순서)
        feature_list = list(correlation_matrix.columns)
        
        # 상관관계 높은 쌍 찾기 (|r| > 0.7)
        high_correlation_pairs = []
        for i, feat1 in enumerate(feature_list):
            for j, feat2 in enumerate(feature_list):
                if i < j:  # 중복 방지
                    corr_value = correlation_matrix.loc[feat1, feat2]
                    if abs(corr_value) > 0.7:
                        high_correlation_pairs.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': float(corr_value)
                        })
        
        # 상관계수 절댓값 기준으로 정렬
        high_correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        log_info(f"상관관계 계산 완료: {len(feature_list)}개 피처, {len(high_correlation_pairs)}개 높은 상관관계 쌍")
        
        return jsonify({
            'success': True,
            'data_source': data_source,
            'feature_list': feature_list,
            'correlation_matrix': correlation_dict,
            'high_correlation_pairs': high_correlation_pairs,
            'data_rows': len(feature_data),
            'message': f'상관관계 계산 완료 ({data_source}, {len(feature_data):,}행)'
        })
        
    except Exception as e:
        log_error(f"피처 상관관계 계산 중 오류: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# WebSocket 이벤트 핸들러 - 실시간 통신 처리
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 시"""
    emit('connected', {'message': '연결되었습니다.'})

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 시"""
    pass

# =============================================================================
# 포트 찾기 함수 - 사용 가능한 포트 자동 검색
# =============================================================================

def find_available_port(start_port=5500, max_port=5600):
    """사용 가능한 포트를 찾는 함수"""
    import socket
    
    # 첫 번째 시도: 기본 범위 (5500-5600)
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Windows와 macOS 호환성을 위해 0.0.0.0으로 바인딩 테스트
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # 두 번째 시도: 확장 범위 (8000-8100)
    print("⚠️  기본 포트 범위(5000-5100)에서 사용 가능한 포트를 찾을 수 없습니다.")
    print("🔍 확장 범위(8000-8100)에서 포트를 검색합니다...")
    
    for port in range(8000, 8100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # 모든 포트가 사용 중인 경우
    raise RuntimeError(f"포트 {start_port}-{max_port} 및 8000-8100 범위에서 사용 가능한 포트를 찾을 수 없습니다.")

# =============================================================================
# 메인 실행 - 웹 서버 시작
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='AI 주식 분석 시스템 - Flask 앱')
    parser.add_argument('--port', type=int, default=None, help='사용할 포트 번호 (기본값: 5500 우선 자동 검색)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='바인딩할 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    args = parser.parse_args()
    
    # 플래그 초기화
    reset_flags()
    
    # 통일된 경로로 필요한 디렉토리 생성
    ensure_all_directories()
    
    # 포트 결정
    if args.port:
        # 사용자가 지정한 포트 사용
        port = args.port
        print(f"🚀 Flask 앱을 지정된 포트 {port}에서 시작합니다...")
    else:
        # 사용 가능한 포트 자동 검색
        try:
            port = find_available_port()
            print(f"🚀 Flask 앱을 포트 {port}에서 시작합니다...")
        except RuntimeError as e:
            print(f"❌ 오류: {e}")
            print("💡 해결 방법:")
            print("   1. 다른 Flask 앱을 종료하세요")
            print("   2. 또는 수동으로 포트를 지정하세요: python flask_app.py --port 5501")
            print("   3. 더 넓은 범위의 포트를 시도해보세요: python flask_app.py --port 8000")
            sys.exit(1)
    
    print(f"🌐 브라우저에서 http://localhost:{port} 으로 접속하세요.")
    print(f"🔧 디버그 모드: {'활성화' if args.debug else '비활성화'}")
    
    # Flask 앱 실행
    if args.debug:
        # 디버그 모드: 파일 변경 감지, 상세 오류 정보, 자동 재시작
        print("🔧 디버그 모드 활성화: 파일 변경 시 자동 재시작됩니다.")
        socketio.run(app, debug=True, host=args.host, port=port, use_reloader=True, log_output=True)
    else:
        # 일반 모드: 안정적이고 빠름
        socketio.run(app, debug=False, host=args.host, port=port, use_reloader=False, log_output=False)
