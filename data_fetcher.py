"""
주식 데이터 수집 모듈
====================

이 파일은 주식 분석에 필요한 모든 데이터를 수집합니다.
- 종목 목록 수집 (KOSPI, KOSDAQ)
- 재무 데이터 수집 (PER, PBR, ROE 등)
- 주가 데이터 수집 (가격, 거래량, 기술적 지표)
- 거시경제 데이터 수집 (KOSPI 지수, 환율, VIX 등)

주요 기능:
- 실시간 데이터 수집
- 기술적 지표 자동 계산
- 데이터 품질 검증 및 정제
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
import concurrent.futures
from tqdm import tqdm
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
import time
import gc
import locale
import platform

# Windows 환경에서 로케일 설정 (FinanceDataReader 내부 오류 방지)
if platform.system() == 'Windows':
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 로케일 설정 실패 시 기본값 유지
        pass

from logger import (log_info, log_warning, log_error, log_critical, log_progress, log_step, log_success, log_start, log_complete,
                   log_data_collection_status)
from exceptions import DataFetchError, DataValidationError

from path_manager import path_manager
from cross_sectional import CROSS_SECTIONAL_FEATURE_COLS, add_cross_sectional_ranks
import requests

# =================================================================
# KRX 세션 로그인 및 requests 패치
# =================================================================
_krx_session = requests.Session()

def login_krx(login_id: str, login_pw: str) -> bool:
    """
    KRX data.krx.co.kr 로그인 후 세션 쿠키(JSESSIONID)를 갱신합니다.
    """
    _LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    _LOGIN_JSP  = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    _LOGIN_URL  = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    try:
        # 초기 세션 발급
        _krx_session.get(_LOGIN_PAGE, headers={"User-Agent": _UA}, timeout=15)
        _krx_session.get(_LOGIN_JSP, headers={"User-Agent": _UA, "Referer": _LOGIN_PAGE}, timeout=15)

        payload = {
            "mbrNm": "", "telNo": "", "di": "", "certType": "",
            "mbrId": login_id, "pw": login_pw,
        }
        headers = {"User-Agent": _UA, "Referer": _LOGIN_PAGE}

        # 로그인 POST
        resp = _krx_session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
        data = resp.json()
        error_code = data.get("_error_code", "")

        # CD011 중복 로그인 처리
        if error_code == "CD011":
            payload["skipDup"] = "Y"
            resp = _krx_session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
            data = resp.json()
            error_code = data.get("_error_code", "")

        if error_code == "CD001":
            _patch_requests_for_fdr()
            log_info("KRX 로그인 성공 및 세션 적용 완료")
            return True
        else:
            log_warning(f"KRX 로그인 실패: {error_code}")
            return False
    except Exception as e:
        log_error(f"KRX 로그인 과정 중 오류 발생: {e}")
        return False

def _patch_requests_for_fdr():
    """FinanceDataReader 등이 전역 세션을 사용하도록 requests.get, requests.post를 패치합니다."""
    if not hasattr(requests, '_original_get'):
        requests._original_get = requests.get
        requests._original_post = requests.post

    def patched_get(url, **kwargs):
        if isinstance(url, str) and ('krx.co.kr' in url or 'data.krx' in url or 'kind.krx' in url):
            return _krx_session.get(url, **kwargs)
        return requests._original_get(url, **kwargs)

    def patched_post(url, data=None, json=None, **kwargs):
        if isinstance(url, str) and ('krx.co.kr' in url or 'data.krx' in url or 'kind.krx' in url):
            return _krx_session.post(url, data=data, json=json, **kwargs)
        return requests._original_post(url, data=data, json=json, **kwargs)

    requests.get = patched_get
    requests.post = patched_post

# =================================================================
# FDR 성공 소스 캐시 (폴백 제거용)
# 최초 1회 성공한 소스(NAVER/KRX)를 저장해 두고 이후 종목들은 바로 그 소스만 사용
_SUCCESSFUL_FDR_SOURCE = None  # (source_name, symbol_prefix) 튜플

# =================================================================
# 지연 초기화: KRX 로그인 (프로세스당 1회만 실행)
_KRX_LOGIN_DONE = False

def _ensure_krx_login():
    """KRX 로그인 보장 (지연 초기화, 프로세스당 1회)"""
    global _KRX_LOGIN_DONE
    if _KRX_LOGIN_DONE:
        return True
    try:
        ok = login_krx("gunbin52", "!rjsqls1357")
        _KRX_LOGIN_DONE = ok
        return ok
    except Exception as e:
        log_warning(f"KRX 로그인 지연 초기화 실패: {e}")
        return False

# =================================================================
# 재무데이터(pykrx) 수집 사용 여부
# - 현재 모델/피처 목록에서는 재무데이터(PER/PBR/ROE/EPS/BPS 등)를 사용하지 않음
# - 불필요한 API 호출/속도 저하를 막기 위해 기본값은 비활성화
# - 추후 필요 시 False로 바꾸면 재무데이터 수집 로직이 다시 활성화됨
# =================================================================
DISABLE_PYKRX_FINANCIAL_FETCH = True

# =================================================================
# FDR 데이터 수집 공통 헬퍼 (폴백 + 소스 로그)
# =================================================================
# KRX 제한이 잦아, FDR에서 NAVER로 가능한 데이터는 NAVER를 먼저 시도한다.
# FDR은 symbol 형식으로 백엔드를 1개만 고르며, 자동 폴백은 없다.
# 아래 폴백은 우리 코드가 FDR을 symbol만 바꿔 순차 호출하는 방식이다.
#
# - 한국 6자리 종목 OHLCV: NAVER(기본, ticker만 넘김) → KRX → NAVER(명시)
#   FDR 소스코드: 6자리 한국 종목은 NaverDailyReader가 기본 (Yahoo 아님)
# - KOSPI 지수: NAVER(NAVER:KOSPI) → KRX(KS11) → YAHOO(^KS11)
#   FDR에서 KS11 단독 호출은 KRX만 사용하며 자동 폴백 없음
# - USD/KRW, VIX: FDR 기준 YAHOO 전용 (NAVER 경로 없음)
# - 종목목록/시가총액(StockListing KRX-MARCAP): FDR 기준 KRX 전용 (NAVER bulk 없음)

KOSPI_INDEX_ATTEMPTS = [
    ('NAVER', 'NAVER:KOSPI'),
    ('KRX', 'KS11'),
    ('YAHOO', '^KS11'),
]


def _build_stock_price_attempts(ticker):
    """한국 주식 OHLCV 수집 순서: NAVER → KRX → NAVER(명시)"""
    return [
        ('NAVER', ticker),
        ('KRX', f'KRX:{ticker}'),
        ('NAVER', f'NAVER:{ticker}'),
    ]


def _log_fetch_attempt(category, source, symbol):
    log_info(f"[{category}] 수집 시도: {source} (symbol={symbol})")


def _normalize_fdr_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index, format='mixed', errors='coerce')
    return df


# FDR 요청 타임아웃 (초) - 네트워크 지연 시 무한 대기 방지
_FDR_REQUEST_TIMEOUT = 5
_FDR_MAX_RETRIES = 2  # 연결 에러 시 2회 재시도

def _fetch_fdr_direct(symbol, start_date, end_date):
    """FDR 직접 호출"""
    return fdr.DataReader(symbol, start_date, end_date)


def _fetch_fdr_with_timeout(symbol, start_date, end_date, timeout=_FDR_REQUEST_TIMEOUT):
    """타임아웃 래퍼"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch_fdr_direct, symbol, start_date, end_date)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"FDR 요청 타임아웃 ({timeout}초): {symbol}")
        except Exception as e:
            raise e


def _fetch_fdr_with_retry(symbol, start_date, end_date, max_retries=_FDR_MAX_RETRIES):
    """연결 에러 시 재시도 (지수 백오프)"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _fetch_fdr_with_timeout(symbol, start_date, end_date)
        except (TimeoutError, ConnectionError, OSError) as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s 지수 백오프
                time.sleep(wait_time)
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(0.3)
                continue
            raise
    raise last_error


def fetch_fdr_ohlcv_with_fallback(category, attempts, start_date, end_date, verbose=True):
    """
    FDR OHLCV 수집 (소스별 폴백 + 타임아웃 + 재시도).

    Args:
        verbose: True면 시도/성공/실패 로그 출력. 주가 배치 수집은 False 권장.

    Returns:
        tuple: (DataFrame, source_name) 또는 (None, None)
    """
    start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    last_error = None
    for source, symbol in attempts:
        if verbose:
            _log_fetch_attempt(category, source, symbol)
        try:
            df = _fetch_fdr_with_retry(symbol, start_date_str, end_date_str)
            if df is not None and not df.empty:
                if verbose:
                    log_info(f"[{category}] 수집 성공: {source} (symbol={symbol}, rows={len(df)})")
                return df, source
            if verbose:
                log_warning(f"[{category}] 수집 결과 없음: {source} (symbol={symbol})")
        except TimeoutError as e:
            last_error = e
            if verbose:
                log_warning(f"[{category}] 수집 타임아웃: {source} (symbol={symbol}) - {e}")
        except Exception as e:
            last_error = e
            if verbose:
                log_warning(f"[{category}] 수집 실패: {source} (symbol={symbol}) - {e}")

    if verbose:
        if last_error:
            log_warning(f"[{category}] 모든 소스 수집 실패 (마지막 오류: {last_error})")
        else:
            log_warning(f"[{category}] 모든 소스 수집 실패")
    return None, None


def fetch_stock_ohlcv_with_fallback(ticker, start_date, end_date, category=None, verbose=False):
    """한국 주식 OHLCV 수집 (캐시된 소스 우선 → 폴백).

    배치 수집 시 종목별 로그 폭주를 막기 위해 기본 verbose=False.
    """
    global _SUCCESSFUL_FDR_SOURCE
    category = category or f"주가/{ticker}"

    # 캐시된 성공 소스가 있으면 가장 먼저 시도 (타임아웃/재시도 적용)
    if _SUCCESSFUL_FDR_SOURCE is not None:
        source_name, symbol_prefix = _SUCCESSFUL_FDR_SOURCE
        if verbose:
            _log_fetch_attempt(category, source_name + "(cached)", symbol_prefix + ticker)
        try:
            df = _fetch_fdr_with_retry(symbol_prefix + ticker, start_date, end_date)
            if df is not None and not df.empty:
                if verbose:
                    log_info(f"[{category}] 수집 성공: {source_name}(cached) (symbol={symbol_prefix + ticker}, rows={len(df)})")
                return df, source_name
        except TimeoutError as e:
            if verbose:
                log_warning(f"[{category}] 캐시 소스 타임아웃, 폴백 진행: {source_name} - {e}")
        except Exception as e:
            if verbose:
                log_warning(f"[{category}] 캐시 소스 실패, 폴백 진행: {source_name} - {e}")

    # 캐시 없거나 실패 시 기존 폴백 로직 수행
    attempts = _build_stock_price_attempts(ticker)
    df, source = fetch_fdr_ohlcv_with_fallback(category, attempts, start_date, end_date, verbose=verbose)

    # 성공 시 캐시 업데이트
    if df is not None and source:
        # symbol_prefix 추출 (NAVER:, KRX: 등)
        for s_name, sym in attempts:
            if s_name == source:
                _SUCCESSFUL_FDR_SOURCE = (source, sym.replace(ticker, ''))
                break

    return df, source


def probe_stock_ohlcv_source(ticker, end_date=None):
    """첫 종목으로 주가 수집 소스를 1회 판별하고 로그 한 줄만 남긴다.

    Returns:
        성공한 소스명(str) 또는 None
    """
    end = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.now()
    probe_start = (end - timedelta(days=30)).strftime('%Y-%m-%d')
    probe_end = end.strftime('%Y-%m-%d')
    _, source = fetch_stock_ohlcv_with_fallback(
        ticker, probe_start, probe_end, category=f'주가소스판별/{ticker}', verbose=False
    )
    if source:
        log_info(f"주가 수집 소스: {source} (첫 종목 {ticker}으로 확인, 이후 종목은 이 소스 우선 사용)")
    else:
        log_warning(f"주가 수집 소스 판별 실패 (첫 종목 {ticker})")
    return source


def _extract_close_series(df):
    df = _normalize_fdr_index(df)
    series = df['Close'].dropna()
    return series[~series.index.duplicated(keep='first')]


def fetch_close_series_with_fallback(category, attempts, start_date, end_date, verbose=True):
    """FDR Close 시계열 수집 (소스별 폴백). 거시경제 등은 verbose=True 유지."""
    df, source = fetch_fdr_ohlcv_with_fallback(
        category, attempts, start_date, end_date, verbose=verbose
    )
    if df is None:
        return None, None
    return _extract_close_series(df), source


# =================================================================
# 유틸리티 함수: 정규화된 선형회귀기울기 계산 (최신 값만)
# =================================================================

def calculate_normalized_linear_regression_slope_latest(series, window=5):
    """
    정규화된 선형회귀기울기 계산 (최신 값만 반환)
    
    Args:
        series: pandas Series (예: 변동성 값)
        window: 선형회귀에 사용할 기간
    
    Returns:
        최신 기울기 값 (스칼라) 또는 np.nan
    """
    if len(series) < window:
        return np.nan
    
    # 최신 window일간의 값만 사용
    y = series.iloc[-window:].values
    current_value = series.iloc[-1]
    
    # 안전성 검사
    if current_value == 0 or np.isnan(current_value) or np.isnan(y).any():
        return np.nan
    
    # 시간 인덱스
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_centered_sq_sum = np.sum(x_centered ** 2)
    
    if x_centered_sq_sum == 0:
        return np.nan
    
    # 선형회귀 기울기 계산
    y_mean = np.nanmean(y)
    y_centered = y - y_mean
    numerator = np.sum(x_centered * y_centered)
    
    # 절대 기울기
    abs_slope = numerator / x_centered_sq_sum
    
    # 정규화: 현재 값으로 나누고 100 곱하기 (백분율)
    normalized_slope = (abs_slope / current_value) * 100
    
    return normalized_slope


# =================================================================
# 벡터화된 기술적 지표 계산 함수들 (pandas_ta 대체, ProcessPoolExecutor 호환)
# =================================================================

def _calc_atr_vectorized(high, low, close, length=14):
    """ATR (Average True Range) 벡터화 계산 - Wilder's Smoothing (EMA) 방식"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing: alpha = 1/length
    atr = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    return atr


def _calc_rsi_vectorized(close, length=14):
    """RSI 벡터화 계산 - Wilder's Smoothing 방식"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's smoothing (EMA with alpha=1/length)
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_adx_vectorized(high, low, close, length=14):
    """ADX (Average Directional Index) 벡터화 계산"""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    
    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smoothed DM
    plus_di = 100 * plus_dm.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean() / (atr + 1e-10)
    
    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = dx.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    
    return adx, plus_di, minus_di


def get_actual_trading_date(selected_analysis_date):
    """
    실제 거래일을 확인하는 함수
    
    주말이나 공휴일을 선택한 경우, 가장 가까운 실제 거래일을 찾아줍니다.
    삼성전자(005930) 주가 데이터를 기준으로 실제 거래일을 확인합니다.
    
    Args:
        selected_analysis_date: 사용자가 선택한 분석 기준일
        
    Returns:
        datetime.date: 실제 거래일
    """
    today = datetime.now().date()
    selected_date = selected_analysis_date.date()
    
    # 삼성전자(005930)로 실제 거래일 확인
    try:
        sample_fetch_start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        sample_df = fdr.DataReader('005930', sample_fetch_start, datetime.now().strftime('%Y-%m-%d'))
        if not sample_df.empty:
            sample_analysis_date_ts = pd.Timestamp(selected_analysis_date)
            sample_temp = sample_df[sample_df.index <= sample_analysis_date_ts]
            if not sample_temp.empty:
                actual_trading_date = sample_temp.index.max().date()
                log_info(f"[DATE] 분석기준일: {selected_analysis_date.strftime('%Y-%m-%d')} → 실제 거래일: {actual_trading_date}")
                return actual_trading_date
            else:
                log_warning("[WARN] 실제 거래일을 확인할 수 없어 분석기준일을 사용합니다")
                return selected_date
        else:
            log_warning("[WARN] 샘플 데이터를 가져올 수 없어 분석기준일을 사용합니다")
            return selected_date
    except Exception as e:
        log_warning(f"[WARN] 실제 거래일 확인 중 오류 발생: {e}, 분석기준일을 사용합니다")
        return selected_date

def get_fs_data_from_pit(stock_list, selected_analysis_date, use_cache=True):
    """
    재무 데이터 수집 함수
    
    기업의 재무 정보(PER, PBR, ROE 등)를 실시간으로 수집합니다.
    이 데이터는 주식의 가치를 평가하는 데 사용됩니다.
    
    Args:
        stock_list: 분석할 종목 목록
        selected_analysis_date: 분석 기준일
        use_cache: 캐시 사용 여부 (현재는 실시간 수집만 지원)
        
    Returns:
        pandas.DataFrame: 재무 데이터가 포함된 데이터프레임
    """
    # 재무데이터 비활성화 모드: 빈 DF 반환 (파이프라인은 계속 진행)
    if DISABLE_PYKRX_FINANCIAL_FETCH:
        log_info("ℹ️ 재무데이터 수집이 비활성화되어 있습니다. (DISABLE_PYKRX_FINANCIAL_FETCH=True)")
        return pd.DataFrame()

    # 항상 실시간 재무데이터 수집 (캐시 사용 안함)
    log_step("실시간 재무데이터 수집", "START", {"모드": "실시간 수집"})
    return _fetch_realtime_financial_data(stock_list, selected_analysis_date)

def _fetch_realtime_financial_data(stock_list, selected_analysis_date):
    """실시간 재무데이터 수집"""
    try:
        log_info("📊 재무데이터 수집 중...")
        
        # 실제 거래일 확인
        actual_trading_date = get_actual_trading_date(selected_analysis_date)
        analysis_date_str = actual_trading_date.strftime('%Y%m%d')
        
        log_info(f"📅 거래일 기준 재무데이터 수집: {actual_trading_date} ({analysis_date_str})")
        
        # pykrx API로 재무데이터 수집
        df_fundamental = stock.get_market_fundamental(analysis_date_str, market="ALL")
        
        if df_fundamental.empty:
            log_warning("재무데이터를 가져올 수 없습니다. 재시도합니다.")
            return _get_historical_financial_data(stock_list, selected_analysis_date)
        
        # 데이터 정제
        if 'PBR' in df_fundamental.columns:
            df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
        
        if df_fundamental.empty:
            log_warning("유효한 재무데이터가 없습니다. 재시도합니다.")
            return _get_historical_financial_data(stock_list, selected_analysis_date)
        
        # 컬럼명 정리
        df_fundamental.reset_index(inplace=True)
        df_fundamental.rename(columns={'티커': '종목코드'}, inplace=True)
        df_fundamental['date'] = pd.to_datetime(analysis_date_str, format='%Y%m%d')
        
        # 요청된 종목만 필터링
        requested_tickers = set(stock_list['종목코드'].astype(str))
        df_fundamental = df_fundamental[df_fundamental['종목코드'].astype(str).isin(requested_tickers)]
        
        # stock_list와 병합
        result_df = pd.merge(stock_list[['종목코드']], df_fundamental, on='종목코드', how='left')
        
        log_success(f"재무데이터 수집 완료: {len(result_df.dropna())}개 종목")
        return result_df
        
    except Exception as e:
        log_error(f"재무데이터 수집 실패: {e}")
        log_info("재무데이터 수집을 재시도합니다.")
        return _fetch_realtime_financial_data(stock_list, selected_analysis_date)



# FDR 0.9.110+ 가 KRX-MARCAP에 쓰는 GitHub 일자별 CSV (오늘 파일이 아직 없으면 404)
# listing 캐시는 최근 구간만 존재 → 과거 월별 시총은 marcap 연도 parquet 폴백 필요
_FDR_KRX_MARCAP_CACHE_BASE = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/"
    "refs/heads/master/data/listing/krx"
)
_MARCAP_YEAR_PARQUET_BASE = (
    "https://raw.githubusercontent.com/FinanceData/marcap/master/data"
)
_marcap_year_df_cache = {}
_marcap_year_lock = threading.Lock()


def _get_krx_max_work_date():
    """KRX에서 최신 거래일(max_work_dt)만 조회. 실패 시 None."""
    try:
        url = (
            "http://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd"
            "?baseName=krx.mdc.i18n.component&key=B128.bld"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
        }
        j = requests.get(url, headers=headers, timeout=15).json()
        date_str = j["result"]["output"][0]["max_work_dt"]
        return datetime.strptime(date_str, "%Y%m%d")
    except Exception as e:
        log_warning(f"KRX 최신 거래일 조회 실패: {e}")
        return None


def _normalize_preferred_datetime(preferred_date=None):
    start = preferred_date
    if start is None:
        start = _get_krx_max_work_date()
    if start is None:
        start = datetime.now()
    return pd.to_datetime(start).to_pydatetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _count_valid_marcap_rows(df):
    """Marcap 컬럼의 유효(non-null, 숫자) 행 수."""
    if df is None or getattr(df, "empty", True) or "Marcap" not in df.columns:
        return 0
    return int(pd.to_numeric(df["Marcap"], errors="coerce").notna().sum())


def _marcap_listing_cache_dir() -> Path:
    """종목목록용 일자별 시총 스냅샷 로컬 캐시 디렉터리."""
    cache_dir = Path(path_manager.data_dir) / "marcap_listing_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _save_local_marcap_listing(df, day) -> bool:
    """시총 유효 스냅샷을 로컬에 저장. 유효행 없으면 저장하지 않음."""
    if _count_valid_marcap_rows(df) == 0:
        return False
    try:
        day = pd.to_datetime(day).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cache_dir = _marcap_listing_cache_dir()
        out_path = cache_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
        tmp_path = cache_dir / f"{day.strftime('%Y-%m-%d')}.parquet.tmp"
        out = df.copy()
        if "Code" in out.columns:
            out["Code"] = out["Code"].astype(str).str.zfill(6)
        if tmp_path.exists():
            tmp_path.unlink()
        out.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        log_info(
            f"로컬 시총 리스팅 캐시 저장: {out_path.name} "
            f"(유효 Marcap {_count_valid_marcap_rows(out):,}행)"
        )
        return True
    except Exception as e:
        log_warning(f"로컬 시총 리스팅 캐시 저장 실패: {e}")
        return False


def _load_local_marcap_listing(preferred_date=None, lookback_days=60):
    """로컬에 저장된 시총 리스팅 스냅샷을 preferred_date부터 거슬러가며 로드."""
    start = _normalize_preferred_datetime(preferred_date)
    cache_dir = _marcap_listing_cache_dir()

    for i in range(lookback_days + 1):
        day = start - timedelta(days=i)
        path = cache_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if _count_valid_marcap_rows(df) > 0:
                if "Code" in df.columns:
                    df = df.copy()
                    df["Code"] = df["Code"].astype(str).str.zfill(6)
                return df, day
        except Exception as e:
            log_warning(f"로컬 시총 리스팅 캐시 로드 실패({path.name}): {e}")
            continue
    return None, None


def _load_fdr_krx_marcap_cache_csv(preferred_date=None, lookback_days=15, log_failures=True):
    """FDR GitHub KRX-MARCAP CSV를 preferred_date부터 거슬러가며 로드.

    Code만 있고 Marcap이 전부 비어 있는 soft-fail CSV는 건너뛴다.

    Returns:
        (DataFrame, 사용된 날짜 datetime) 또는 실패 시 (None, None)
    """
    start = _normalize_preferred_datetime(preferred_date)

    last_error = None
    skipped_empty_marcap = 0
    for i in range(lookback_days + 1):
        day = start - timedelta(days=i)
        url = f"{_FDR_KRX_MARCAP_CACHE_BASE}/{day.strftime('%Y-%m-%d')}.csv"
        try:
            df = pd.read_csv(
                url,
                index_col=0,
                dtype={"Code": str, "Dept": str, "ChangeCode": str, "MarketId": str},
            )
            df = df.reset_index(drop=True)
            if df is None or df.empty or "Code" not in df.columns:
                continue
            # 오늘자처럼 파일은 있으나 시총이 비어 있는 경우 → 다음 날로 계속
            if _count_valid_marcap_rows(df) == 0:
                skipped_empty_marcap += 1
                continue
            return df, day
        except Exception as e:
            last_error = e
            continue

    if log_failures:
        log_warning(
            f"FDR GitHub KRX-MARCAP CSV 폴백 실패 "
            f"(시작일={start.strftime('%Y-%m-%d')}, lookback={lookback_days}, "
            f"시총공백스킵={skipped_empty_marcap}): {last_error}"
        )
    return None, None


def _prepare_marcap_year_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "Date" in out.columns:
        out = out.copy()
        out["Date"] = pd.to_datetime(out["Date"])
    if "Code" in out.columns:
        out = out.copy()
        out["Code"] = out["Code"].astype(str).str.zfill(6)
    return out


def _get_marcap_year_dataframe(year: int):
    """FinanceData/marcap 연도 parquet (로컬 캐시 + 메모리 캐시, 락+원자적 저장)."""
    year = int(year)
    with _marcap_year_lock:
        cached = _marcap_year_df_cache.get(year)
        if cached is not None:
            return cached

        cache_dir = Path(path_manager.data_dir) / "marcap_year_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"marcap-{year}.parquet"
        tmp_path = cache_dir / f"marcap-{year}.parquet.tmp"

        df = None
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
            except Exception as e:
                log_warning(f"깨진 marcap 연도 캐시 삭제({year}): {e}")
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if df is None or getattr(df, "empty", True):
            url = f"{_MARCAP_YEAR_PARQUET_BASE}/marcap-{year}.parquet"
            log_info(f"FinanceData/marcap 연도 파일 다운로드: {year}")
            df = pd.read_parquet(url)
            if df is None or df.empty:
                raise RuntimeError(f"marcap-{year}.parquet 이 비어 있습니다.")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                df.to_parquet(tmp_path, index=False)
                pd.read_parquet(tmp_path)
                tmp_path.replace(cache_path)
            except Exception as e:
                log_warning(f"marcap 연도 로컬 캐시 저장 실패({year}): {e}")
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        df = _prepare_marcap_year_dataframe(df)
        _marcap_year_df_cache[year] = df
        return df


def _load_marcap_dataset_snapshot(preferred_date=None, lookback_days=15):
    """FinanceData/marcap 연도 데이터에서 일자 스냅샷 로드."""
    start = _normalize_preferred_datetime(preferred_date)
    last_error = None

    for i in range(lookback_days + 1):
        day = start - timedelta(days=i)
        if day.weekday() >= 5:
            continue
        try:
            year_df = _get_marcap_year_dataframe(day.year)
            day_ts = pd.Timestamp(day.date())
            day_df = year_df[year_df["Date"] == day_ts]
            if day_df is None or day_df.empty:
                continue
            out = day_df.drop(columns=["Date"], errors="ignore").reset_index(drop=True)
            if "Code" in out.columns and _count_valid_marcap_rows(out) > 0:
                return out, day
        except Exception as e:
            last_error = e
            continue

    log_warning(
        f"FinanceData/marcap 연도 parquet 폴백 실패 "
        f"(시작일={start.strftime('%Y-%m-%d')}, lookback={lookback_days}): {last_error}"
    )
    return None, None


def preload_marcap_year_files(years):
    """월별 시총 수집 전, 필요 연도 parquet를 순차로 미리 로드."""
    uniq = sorted({int(y) for y in years})
    for year in uniq:
        _get_marcap_year_dataframe(year)
        log_info(f"marcap 연도 캐시 준비 완료: {year}")


def _fdr_listing_csv_exists(day) -> bool:
    """FDR GitHub listing CSV 존재 여부."""
    day = pd.to_datetime(day).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    url = f"{_FDR_KRX_MARCAP_CACHE_BASE}/{day.strftime('%Y-%m-%d')}.csv"
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            return True
        r = requests.get(url, timeout=15, stream=True)
        ok = r.status_code == 200
        r.close()
        return ok
    except Exception:
        return False


def _load_fdr_listing_csv_exact(day):
    """특정 일자의 FDR listing CSV만 로드 (날짜 lookback 없음)."""
    day = pd.to_datetime(day).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    url = f"{_FDR_KRX_MARCAP_CACHE_BASE}/{day.strftime('%Y-%m-%d')}.csv"
    df = pd.read_csv(
        url,
        index_col=0,
        dtype={"Code": str, "Dept": str, "ChangeCode": str, "MarketId": str},
    )
    df = df.reset_index(drop=True)
    if df is None or df.empty or "Code" not in df.columns or "Marcap" not in df.columns:
        raise RuntimeError(f"FDR listing CSV 무효: {day.strftime('%Y-%m-%d')}")
    if _count_valid_marcap_rows(df) == 0:
        raise RuntimeError(
            f"FDR listing CSV 시총 공백: {day.strftime('%Y-%m-%d')} "
            f"(행={len(df)}, Marcap 유효=0)"
        )
    df = df.copy()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df


def probe_fdr_listing_csv_range(end_date=None, max_lookback_days=800):
    """FDR listing CSV로 수집 가능한 (oldest, latest) 평일 범위를 프로브.

    Returns:
        (oldest_datetime, latest_datetime) 또는 실패 시 (None, None)
    """
    end = _normalize_preferred_datetime(end_date)

    latest = None
    for i in range(0, 30):
        d = end - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        if _fdr_listing_csv_exists(d):
            latest = d
            break
    if latest is None:
        log_warning("FDR listing CSV: 최근 구간에서 파일을 찾지 못했습니다.")
        return None, None

    oldest_known = latest
    probe = latest
    stepped_out = False
    for _ in range(max_lookback_days // 30 + 1):
        probe = probe - timedelta(days=30)
        while probe.weekday() >= 5:
            probe -= timedelta(days=1)
        if probe < latest - timedelta(days=max_lookback_days):
            break
        if _fdr_listing_csv_exists(probe):
            oldest_known = probe
            continue

        lo, hi = probe, oldest_known
        while (hi - lo).days > 1:
            mid = lo + timedelta(days=(hi - lo).days // 2)
            while mid.weekday() >= 5:
                mid += timedelta(days=1)
            if mid >= hi:
                mid = hi - timedelta(days=1)
                while mid.weekday() >= 5:
                    mid -= timedelta(days=1)
            if mid <= lo:
                break
            if _fdr_listing_csv_exists(mid):
                hi = mid
                oldest_known = mid
            else:
                lo = mid
        stepped_out = True
        break

    d = oldest_known
    while True:
        prev = d - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        if (latest - prev).days > max_lookback_days:
            break
        if _fdr_listing_csv_exists(prev):
            d = prev
        else:
            break

    log_info(
        f"FDR listing CSV 가능 구간: {d.strftime('%Y-%m-%d')} ~ {latest.strftime('%Y-%m-%d')}"
        + (" (프로브 경계 확정)" if stepped_out else "")
    )
    return d, latest


def _normalize_marcap_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """시총 스냅샷을 Code/Marcap/Stocks 최소 스키마로 정규화."""
    out = df.copy()
    if "Code" not in out.columns:
        raise ValueError("시총 스냅샷에 Code 컬럼이 없습니다.")
    if "Marcap" not in out.columns:
        raise ValueError("시총 스냅샷에 Marcap 컬럼이 없습니다.")
    out["Code"] = out["Code"].astype(str).str.zfill(6)
    cols = ["Code", "Marcap"]
    if "Stocks" in out.columns:
        cols.append("Stocks")
    return out[cols]


def _load_marcap_parquet_range(start_date, end_date) -> pd.DataFrame:
    """FinanceData/marcap 연도 parquet에서 [start, end] 구간 일별 패널 로드."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        return pd.DataFrame(columns=["Code", "date", "Marcap", "Stocks"])

    frames = []
    for year in range(start.year, end.year + 1):
        try:
            year_df = _get_marcap_year_dataframe(year)
        except Exception as e:
            log_warning(f"marcap 연도 로드 실패({year}): {e}")
            continue
        part = year_df[(year_df["Date"] >= start) & (year_df["Date"] <= end)].copy()
        if part.empty:
            continue
        part["Code"] = part["Code"].astype(str).str.zfill(6)
        keep = ["Code", "Date", "Marcap"]
        if "Stocks" in part.columns:
            keep.append("Stocks")
        part = part[keep].rename(columns={"Date": "date"})
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["Code", "date", "Marcap", "Stocks"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out


def build_daily_marcap_panel(start_date, end_date) -> pd.DataFrame:
    """학습/백테스트용 일별 시총 패널.

    - FDR listing CSV 가능 구간: FDR 일자 CSV (캐시 parquet 미사용)
    - 그 이전(및 FDR 밖): FinanceData/marcap 연도 parquet
    - 두 소스 공백은 로그로 남기고, 피처 단계에서 merge_asof / Stocks×종가로 보완
    """
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        raise ValueError(f"시총 기간이 올바르지 않습니다: {start_date} ~ {end_date}")

    log_info(f"일별 시가총액 패널 구축 시작: {start.date()} ~ {end.date()}")

    fdr_oldest, fdr_latest = probe_fdr_listing_csv_range(end_date=end)
    years_needed = list(range(start.year, end.year + 1))
    preload_marcap_year_files(years_needed)

    frames = []
    fdr_days_ok = 0
    fdr_days_fail = 0

    # 1) FDR 가능 구간: 일자 CSV 직접 로드
    if fdr_oldest is not None and fdr_latest is not None:
        fdr_start = max(start, pd.Timestamp(fdr_oldest).normalize())
        fdr_end = min(end, pd.Timestamp(fdr_latest).normalize())
        if fdr_start <= fdr_end:
            fdr_days = pd.bdate_range(fdr_start, fdr_end)
            log_info(
                f"FDR listing CSV 구간 수집: {fdr_start.date()} ~ {fdr_end.date()} "
                f"({len(fdr_days)} 평일)"
            )

            def _one_fdr_day(day_ts):
                day = pd.Timestamp(day_ts).to_pydatetime()
                snap = _normalize_marcap_snapshot(_load_fdr_listing_csv_exact(day))
                snap = snap.copy()
                snap["date"] = pd.Timestamp(day_ts).normalize()
                return snap

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_one_fdr_day, d): d for d in fdr_days}
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    done += 1
                    try:
                        frames.append(fut.result())
                        fdr_days_ok += 1
                    except Exception as e:
                        fdr_days_fail += 1
                        day = futures[fut]
                        log_warning(f"FDR listing CSV 로드 실패({pd.Timestamp(day).date()}): {e}")
                    if done % 20 == 0 or done == len(futures):
                        log_progress("FDR 일별 시총", done, len(futures))
        else:
            fdr_start = fdr_end = None
    else:
        fdr_start = fdr_end = None
        log_warning("FDR listing CSV 구간 없음 → 전 구간 marcap parquet 사용")

    # 2) FDR 밖: marcap parquet
    marcap_ranges = []
    if fdr_start is None:
        marcap_ranges.append((start, end))
    else:
        if start < fdr_start:
            marcap_ranges.append((start, fdr_start - pd.Timedelta(days=1)))
        if fdr_end is not None and fdr_end < end:
            marcap_ranges.append((fdr_end + pd.Timedelta(days=1), end))

    marcap_rows = 0
    for m_start, m_end in marcap_ranges:
        if m_end < m_start:
            continue
        log_info(f"marcap parquet 구간 로드: {m_start.date()} ~ {m_end.date()}")
        part = _load_marcap_parquet_range(m_start, m_end)
        if part is not None and not part.empty:
            frames.append(part)
            marcap_rows += len(part)
            # 공백 탐지(평일 기준)
            have = set(pd.to_datetime(part["date"]).dt.normalize().unique())
            need = set(pd.bdate_range(m_start, m_end))
            missing = sorted(need - have)
            if missing:
                log_warning(
                    f"marcap 구간 내 결측 평일 {len(missing)}개 "
                    f"(예: {pd.Timestamp(missing[0]).date()} ~ {pd.Timestamp(missing[-1]).date()})"
                )

    if not frames:
        raise RuntimeError(
            f"일별 시가총액 패널이 비었습니다 ({start.date()} ~ {end.date()})"
        )

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    if "Stocks" not in df.columns:
        df["Stocks"] = np.nan
    df = df.drop_duplicates(subset=["Code", "date"], keep="last")
    df.sort_values(by=["Code", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log_info(
        f"✅ 일별 시가총액 패널 완료: {len(df):,}행 "
        f"(FDR일 {fdr_days_ok}성공/{fdr_days_fail}실패, marcap행 {marcap_rows:,})"
    )
    return df


def fetch_krx_marcap_snapshot(date_yyyymmdd: str, lookback_days: int = 15):
    """
    특정 일자 KRX-MARCAP 스냅샷 수집.
    1) FDR listing CSV(해당일)  2) lookback CSV  3) FinanceData/marcap parquet
    ※ StockListing 날짜 인자는 FDR 0.9.x에서 무시되므로 사용하지 않음.
    """
    date_yyyymmdd = str(date_yyyymmdd).strip()
    preferred = datetime.strptime(date_yyyymmdd, '%Y%m%d')

    try:
        df = _load_fdr_listing_csv_exact(preferred)
        return df
    except Exception:
        pass

    df, used_day = _load_fdr_krx_marcap_cache_csv(
        preferred_date=preferred, lookback_days=lookback_days, log_failures=False
    )
    source = "GitHub listing CSV"

    if df is None or getattr(df, "empty", True):
        df, used_day = _load_marcap_dataset_snapshot(
            preferred_date=preferred, lookback_days=lookback_days
        )
        source = "FinanceData/marcap parquet"

    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"KRX-MARCAP 스냅샷을 찾지 못함: {date_yyyymmdd}")

    if used_day is not None and used_day.strftime('%Y%m%d') != date_yyyymmdd:
        log_info(
            f"KRX-MARCAP 폴백({source}): 요청={date_yyyymmdd} → 사용={used_day.strftime('%Y-%m-%d')}"
        )
    # 성공한 스냅샷은 로컬에 쌓아 다음 soft-fail에 대비
    save_day = used_day if used_day is not None else preferred
    _save_local_marcap_listing(df, save_day)
    return df


def _get_stock_list_from_marcap(analysis_date=None, min_marcap=10_000_000_000):
    """KRX-MARCAP에서 주식 목록 가져오기 (통일된 함수)
    
    Args:
        analysis_date: 분석 기준일 (None이면 현재 날짜)
        min_marcap: 최소 시가총액 (기본값: 100억원 = 10,000,000,000원)
    """
    try:
        df_marcap = None
        cache_fallback_date = None
        primary_error = None
        used_source = None

        try:
            if analysis_date:
                # 과거 날짜용
                date_str = analysis_date.strftime('%Y%m%d')
                log_info(f"종목목록/시가총액 수집 소스: KRX (StockListing KRX-MARCAP, 기준일={date_str})")
                log_info(f"FinanceDataReader를 통해 {date_str} 기준 종목 시가총액 정보 수집 (KRX-MARCAP)...")
                df_marcap = fdr.StockListing('KRX-MARCAP', date_str)
            else:
                # 현재 날짜용
                log_info("종목목록/시가총액 수집 소스: KRX (StockListing KRX-MARCAP, FDR에 NAVER bulk 경로 없음)")
                log_info("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
                df_marcap = fdr.StockListing('KRX-MARCAP')
            if _count_valid_marcap_rows(df_marcap) > 0:
                used_source = "StockListing"
        except Exception as e:
            primary_error = e
            log_warning(f"FDR StockListing(KRX-MARCAP) 실패, 폴백 시도: {e}")
            df_marcap = None

        # 빈 DF뿐 아니라 "행은 있으나 Marcap이 전부 비어 있는" soft-fail도 폴백
        listing_rows = 0 if df_marcap is None else len(df_marcap)
        valid_marcap = _count_valid_marcap_rows(df_marcap)
        need_fallback = (
            df_marcap is None
            or (hasattr(df_marcap, "empty") and df_marcap.empty)
            or valid_marcap == 0
        )
        preferred = analysis_date if analysis_date is not None else None
        if need_fallback:
            if listing_rows > 0 and valid_marcap == 0:
                log_warning(
                    f"StockListing Marcap 유효행 0/{listing_rows} "
                    f"(시세 공백 soft-fail) → GitHub CSV / 로컬캐시 / parquet 폴백"
                )

            # 1) GitHub CSV (시총 유효한 날만; 찾은 날은 로컬에 저장)
            df_fb, cache_fallback_date = _load_fdr_krx_marcap_cache_csv(preferred_date=preferred)
            if _count_valid_marcap_rows(df_fb) > 0:
                df_marcap = df_fb
                used_source = "GitHub CSV"
                _save_local_marcap_listing(df_fb, cache_fallback_date)
                log_info(
                    f"종목목록/시총 스냅샷 폴백 사용: FDR GitHub CSV "
                    f"({cache_fallback_date.strftime('%Y-%m-%d')})"
                )
            else:
                # 2) 로컬에 모아 둔 최근 정상 스냅샷
                df_local, local_day = _load_local_marcap_listing(preferred_date=preferred)
                if _count_valid_marcap_rows(df_local) > 0:
                    df_marcap = df_local
                    cache_fallback_date = local_day
                    used_source = "local listing cache"
                    log_info(
                        f"종목목록/시총 스냅샷 폴백 사용: 로컬 캐시 "
                        f"({local_day.strftime('%Y-%m-%d')})"
                    )
                else:
                    # 3) 기존 스냅샷 체인(CSV lookback → marcap parquet)
                    snap_day = preferred
                    if snap_day is None:
                        snap_day = _get_krx_max_work_date() or datetime.now()
                    snap_day = pd.to_datetime(snap_day)
                    df_snap = fetch_krx_marcap_snapshot(snap_day.strftime('%Y%m%d'))
                    if _count_valid_marcap_rows(df_snap) == 0:
                        raise primary_error or RuntimeError(
                            "KRX-MARCAP 폴백 후에도 Marcap 유효행이 없습니다."
                        )
                    df_marcap = df_snap
                    cache_fallback_date = snap_day
                    used_source = "fetch_krx_marcap_snapshot"
                    _save_local_marcap_listing(df_snap, snap_day)
                    log_info(
                        f"종목목록/시총 스냅샷 폴백 사용: fetch_krx_marcap_snapshot "
                        f"(요청={snap_day.strftime('%Y-%m-%d')})"
                    )

        # StockListing이 곧바로 성공한 경우에도 로컬에 쌓아 둔다
        if used_source == "StockListing":
            save_day = preferred
            if save_day is None:
                save_day = _get_krx_max_work_date() or datetime.now()
            _save_local_marcap_listing(df_marcap, save_day)

        if 'Name' not in df_marcap.columns:
            df_marcap = df_marcap.copy()
            df_marcap['Name'] = ''
        if 'Stocks' not in df_marcap.columns:
            df_marcap = df_marcap.copy()
            df_marcap['Stocks'] = 0
        df_marcap['Marcap'] = pd.to_numeric(df_marcap['Marcap'], errors='coerce')

        # 스팩, 리츠 제외
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        
        # KONEX 제외 (KOSPI, KOSDAQ만 포함)
        if 'Market' in df_marcap.columns:
            df_marcap = df_marcap[df_marcap['Market'].isin(['KOSPI', 'KOSDAQ'])].copy()
            log_info(f"KONEX 제외 후 종목 수: {len(df_marcap)}개")
        
        # 상장주식수가 있는 경우만 필터링
        if 'Stocks' in df_marcap.columns:
            df_marcap = df_marcap[pd.to_numeric(df_marcap['Stocks'], errors='coerce').fillna(0) > 0]
        
        # 시가총액 필터링 (전 구간 공통 유니버스 컷)
        if 'Marcap' in df_marcap.columns:
            min_marcap_billion = min_marcap / 1_000_000_000  # 억원 단위로 변환
            before_count = len(df_marcap)
            df_marcap = df_marcap[df_marcap['Marcap'] >= min_marcap].copy()
            excluded_count = before_count - len(df_marcap)
            if excluded_count > 0:
                log_info(f"시가총액 {min_marcap_billion:.0f}억 미만 종목 {excluded_count}개 제외")
        
        # 컬럼명 정리 및 종목코드 6자리 패딩
        stock_list = df_marcap[['Code', 'Name', 'Stocks', 'Marcap']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수', 'Marcap': '시가총액'}, inplace=True)
        
        # 종목코드를 6자리로 패딩
        stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.zfill(6)
        
        # 시가총액 오름차순 정렬 (큰 종목이 마지막에 처리되도록)
        stock_list.sort_values(by='시가총액', ascending=True, inplace=True)
        
        # 시가총액 컬럼 제거 (기존 로직과 호환성 유지)
        stock_list.drop(columns=['시가총액'], inplace=True)
        
        log_info(f"총 {len(stock_list)}개 종목을 찾았습니다. (시가총액 오름차순 정렬됨)")
        return stock_list
        
    except Exception as e:
        error_msg = f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}"
        log_error(error_msg)
        raise DataFetchError(error_msg, source="FinanceDataReader")

def fetch_stock_list(min_marcap=10_000_000_000):
    """현재 날짜 기준 주식 목록 가져오기 (캐시 없이 실시간 수집)
    
    Args:
        min_marcap: 최소 시가총액 (기본값: 100억원 = 10,000,000,000원)
    """
    return _get_stock_list_from_marcap(analysis_date=None, min_marcap=min_marcap)

def fetch_stock_list_for_date(analysis_date, min_marcap=10_000_000_000):
    """특정 날짜 기준 주식 목록 가져오기 (캐시 없음)
    
    Args:
        analysis_date: 분석 기준일
        min_marcap: 최소 시가총액 (기본값: 100억원 = 10,000,000,000원)
    """
    return _get_stock_list_from_marcap(analysis_date=analysis_date, min_marcap=min_marcap)

def _fetch_macro_data(start_date, end_date):
    log_info(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
    try:
        start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        macro_data = {}

        # KOSPI: NAVER → KRX → YAHOO (모델 피처: KOSPI_disparity_20, KOSPI_MA20_Slope)
        kospi_series, _ = fetch_close_series_with_fallback(
            '거시/KOSPI', KOSPI_INDEX_ATTEMPTS, start_date_str, end_date_str
        )
        if kospi_series is not None:
            macro_data['KOSPI'] = kospi_series

        # USD/KRW: YAHOO만 (모델 피처 아님, 시장현황 표시용)
        usdkrw_series, _ = fetch_close_series_with_fallback(
            '거시/USD-KRW', [('YAHOO', 'USD/KRW')], start_date_str, end_date_str
        )
        if usdkrw_series is not None:
            macro_data['USDKRW'] = usdkrw_series

        # VIX: YAHOO만 (모델 피처 아님, 시장현황 표시용)
        vix_series, _ = fetch_close_series_with_fallback(
            '거시/VIX', [('YAHOO', '^VIX')], start_date_str, end_date_str
        )
        if vix_series is not None:
            macro_data['VIX'] = vix_series
            
        if macro_data:
            macro_df = pd.concat(macro_data.values(), axis=1, keys=macro_data.keys()).ffill()
            # pct_1d는 KOSPI, USDKRW만 생성 (VIX는 생성하지 않음)
            if 'KOSPI' in macro_df.columns:
                macro_df['KOSPI_pct_1d'] = macro_df['KOSPI'].pct_change(1)
            if 'USDKRW' in macro_df.columns:
                macro_df['USDKRW_pct_1d'] = macro_df['USDKRW'].pct_change(1)
            if 'VIX' in macro_df.columns:
                # VIX_pct_1d는 생성하지 않음 (삭제 요청)
                pass
            
            # KOSPI 변동성 및 이격도 계산 (종목 데이터와 동일한 방식)
            if 'KOSPI' in macro_df.columns:
                kospi_close = macro_df['KOSPI']
                
                # 이격도 계산 (종목 데이터와 동일한 방식)
                # 이격도 = (현재가 / 이동평균) * 100
                for period in [20]:
                    ma = kospi_close.rolling(window=period).mean()
                    macro_df[f'KOSPI_disparity_{period}'] = (kospi_close / ma) * 100
                
                # KOSPI 변동성 1M 계산 (20일 기준) - 2024년 12월 제거
                # try:
                #     kospi_std_20 = kospi_close.rolling(window=20).std()
                #     kospi_mean_20 = kospi_close.rolling(window=20).mean()
                #     macro_df['KOSPI_변동성(1M)'] = kospi_std_20 / kospi_mean_20
                # except Exception as e:
                #     log_warning(f"KOSPI 변동성(1M) 계산 실패: {e}")
                #     macro_df['KOSPI_변동성(1M)'] = np.nan
                
                # KOSPI_MA20_Slope 계산 (KOSPI 20일 이동평균선 기울기)
                try:
                    from data_processor import calculate_normalized_linear_regression_slope
                    kospi_ma20 = kospi_close.rolling(window=20).mean()
                    macro_df['KOSPI_MA20_Slope'] = calculate_normalized_linear_regression_slope(kospi_ma20, window=5)
                except Exception as e:
                    log_warning(f"KOSPI_MA20_Slope 계산 실패: {e}")
                    macro_df['KOSPI_MA20_Slope'] = np.nan
            
            # 인덱스를 date 컬럼으로 변환 (merge_asof를 위해)
            macro_df.reset_index(inplace=True)
            macro_df.rename(columns={'index': 'date'}, inplace=True)
            log_info("✅ 거시 경제 지표 수집 완료.")
            return macro_df
        else:
            log_warning("거시 경제 지표 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        error_msg = f"거시 경제 지표 수집 실패: {e}"
        log_error(error_msg)
        # 오류 발생 시 빈 DataFrame 반환 (분석 중단 방지)
        return pd.DataFrame()

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']; shares = stock_info['상장주식수']
    try:
        fetch_start = (pd.to_datetime(start_date_for_fetch) - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # NAVER → KRX → NAVER(명시) 폴백 (종목별 소스 로그는 배치 시작 시 1회만)
        df_price_full, _ = fetch_stock_ohlcv_with_fallback(
            ticker, fetch_start, end_date_for_fetch, verbose=False
        )
        
        if df_price_full is None or df_price_full.empty or len(df_price_full) < 251 + 60: return None, None
        df_price_full.rename(columns={'Open':'시가', 'Close':'종가', 'High':'고가', 'Low':'저가', 'Volume':'거래량'}, inplace=True)

        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None
        actual_analysis_date = df_temp.index.max()
        reference_date_price = df_temp.loc[actual_analysis_date]['종가']
        latest_current_price = df_price_full.iloc[-1]['종가']
        
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        latest_data = {} 

        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        
        # RVOL (상대 거래량) 계산
        try:
            거래량_20일_평균 = df_for_indicators['거래량'].rolling(window=20).mean().iloc[-1]
            현재_거래량 = df_for_indicators['거래량'].iloc[-1]
            if pd.notna(거래량_20일_평균) and 거래량_20일_평균 > 0 and pd.notna(현재_거래량):
                latest_data['RVOL'] = 현재_거래량 / 거래량_20일_평균
            else:
                latest_data['RVOL'] = np.nan
        except Exception as e:
            log_warning(f"RVOL 계산 실패 ({ticker}): {e}")
            latest_data['RVOL'] = np.nan
        
        # 시총 회전율 계산
        try:
            시가총액 = reference_date_price * shares
            if 시가총액 > 0:
                # 시총 회전율(1W): 5일 평균 거래대금 / 시가총액 * 100
                거래대금_5일_평균 = df_for_indicators['거래대금'].rolling(window=5).mean().iloc[-1]
                if pd.notna(거래대금_5일_평균):
                    latest_data['시총 회전율(1W)'] = (거래대금_5일_평균 / 시가총액) * 100
                else:
                    latest_data['시총 회전율(1W)'] = np.nan
                
                # 시총 회전율(3M): 60일 평균 거래대금 / 시가총액 * 100
                거래대금_60일_평균 = df_for_indicators['거래대금'].rolling(window=60).mean().iloc[-1]
                if pd.notna(거래대금_60일_평균):
                    latest_data['시총 회전율(3M)'] = (거래대금_60일_평균 / 시가총액) * 100
                else:
                    latest_data['시총 회전율(3M)'] = np.nan
            else:
                latest_data['시총 회전율(1W)'] = np.nan
                latest_data['시총 회전율(3M)'] = np.nan
        except Exception as e:
            log_warning(f"시총 회전율 계산 실패 ({ticker}): {e}")
            latest_data['시총 회전율(1W)'] = np.nan
            latest_data['시총 회전율(3M)'] = np.nan
        
        latest_data['log_mktcap'] = np.log(reference_date_price * shares) if (reference_date_price * shares) > 0 else np.nan
        # 시가총액은 원 단위로 통일 (백테스트 data_processor와 동일). Exclude cond3도 원 기준.
        try:
            latest_data['시가총액'] = float(reference_date_price) * float(shares)
        except Exception:
            latest_data['시가총액'] = np.nan
        
        # ATR 계산 (5일, 20일, 60일) - 벡터화된 구현 사용
        atr_5 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=5)
        atr_20 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=20)
        atr_60 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=60)
        
        # ATRr_5 계산 (기준 - 1W): "최근 1주일 변동성 수준"
        if atr_5 is not None and len(atr_5) > 0:
            latest_data['ATRr_5'] = (atr_5.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_5'] = np.nan
        
        # ATRr_20 계산 (기준 - 1M): "이 종목의 기초 체급은?"
        if atr_20 is not None and len(atr_20) > 0:
            latest_data['ATRr_20'] = (atr_20.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_20'] = np.nan
        
        # ATRr_60 계산 (기준 - 3M): "중기 추세적 변동성"
        if atr_60 is not None and len(atr_60) > 0:
            latest_data['ATRr_60'] = (atr_60.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_60'] = np.nan
        
        # ADX 계산 - 벡터화된 구현 사용
        adx_14, _, _ = _calc_adx_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=14)
        if adx_14 is not None and len(adx_14) > 0:
            latest_data['ADX_14'] = adx_14.iloc[-1]
        else:
            latest_data['ADX_14'] = np.nan
        
        # =================================================================
        # [단일 핵심 피처] CLV (Close Location Value, 종가 위치 지수)
        # 캔들 내 매수/매도 힘의 우위를 -1(매도 압승) ~ +1(매수 압승)로 수치화
        # =================================================================
        try:
            # 수식: (2 * 종가 - 고가 - 저가) / (고가 - 저가)
            # 분모가 0이 되는 것(예: 점상한가/점하한가)을 방지하기 위해 1e-8 추가
            close_last = df_for_indicators['종가'].iloc[-1]
            high_last = df_for_indicators['고가'].iloc[-1]
            low_last = df_for_indicators['저가'].iloc[-1]
            
            denominator = high_last - low_last + 1e-8
            clv_value = (2 * close_last - high_last - low_last) / denominator
            
            # 부동소수점 오차로 인해 -1이나 1을 미세하게 넘는 경우를 방지
            latest_data['CLV'] = float(np.clip(clv_value, -1.0, 1.0))
            
        except Exception as e:
            log_warning(f"CLV 계산 실패 ({ticker}): {e}")
            latest_data['CLV'] = np.nan
        
        # RSI_14 계산 - 벡터화된 구현 사용
        rsi_14 = _calc_rsi_vectorized(df_for_indicators['종가'], length=14)
        
        # RSI_Signal_Oscillator 계산: RSI_14 - RSI_14.rolling(9).mean()
        # MACD 원리를 RSI에 적용한 것으로, 양수면 RSI가 평균을 뚫고 올라가는 중(골든크로스)
        if rsi_14 is not None and len(rsi_14) >= 9:
            rsi_14_ma9 = rsi_14.rolling(window=9).mean()
            if pd.notna(rsi_14_ma9.iloc[-1]) and pd.notna(rsi_14.iloc[-1]):
                latest_data['RSI_Signal_Oscillator'] = rsi_14.iloc[-1] - rsi_14_ma9.iloc[-1]
            else:
                latest_data['RSI_Signal_Oscillator'] = np.nan
        else:
            latest_data['RSI_Signal_Oscillator'] = np.nan
        
        # 이격도 계산 (120일, 240일) - disparity_20 제거
        for p in [120, 240]:
            ma = df_for_indicators['종가'].rolling(window=p).mean()
            latest_data[f'disparity_{p}'] = ((df_for_indicators['종가'] / ma) * 100).iloc[-1]

        # MA120_Slope 계산 (120일 이동평균선 기울기)
        try:
            ma120 = df_for_indicators['종가'].rolling(window=120).mean()
            latest_data['MA120_Slope'] = calculate_normalized_linear_regression_slope_latest(ma120, window=5)
        except Exception as e:
            log_warning(f"MA120_Slope 계산 실패 ({ticker}): {e}")
            latest_data['MA120_Slope'] = np.nan

        # MA20_Slope 계산 (20일 이동평균선 기울기) - MA120/MA240과 동일한 방식
        try:
            ma20 = df_for_indicators['종가'].rolling(window=20).mean()
            latest_data['MA20_Slope'] = calculate_normalized_linear_regression_slope_latest(ma20, window=5)
        except Exception as e:
            log_warning(f"MA20_Slope 계산 실패 ({ticker}): {e}")
            latest_data['MA20_Slope'] = np.nan

        # MA240_Slope 계산 (240일 이동평균선 기울기)
        try:
            ma240 = df_for_indicators['종가'].rolling(window=240).mean()
            latest_data['MA240_Slope'] = calculate_normalized_linear_regression_slope_latest(ma240, window=5)
        except Exception as e:
            log_warning(f"MA240_Slope 계산 실패 ({ticker}): {e}")
            latest_data['MA240_Slope'] = np.nan

        # =================================================================
        # 랭킹 제외 규칙 (요구사항 반영)
        # - MA20이 MA120, MA240 둘 다 아래에 있고 종가가 MA60 아래에 있으면 제외
        # - 또는, MA20이 14거래일 연속 하락(전일 대비 변화량 < 0)하고 종가가 MA20 아래에 있으면 제외
        # - 또는, 당일 시가총액이 1000억 미만이면 제외 (당일 조건)
        #
        # 조건:
        #   ( (MA20 < MA120) & (MA20 < MA240) & (종가 < MA60) ) |
        #   ( (MA20이 14거래일 연속 하락) & (종가 < MA20) ) |
        #   ( 시가총액 < 1000억 )
        # =================================================================
        try:
            ma20_lvl = df_for_indicators['종가'].rolling(window=20).mean()
            ma60_lvl = df_for_indicators['종가'].rolling(window=60).mean()
            ma120_lvl = df_for_indicators['종가'].rolling(window=120).mean()
            ma240_lvl = df_for_indicators['종가'].rolling(window=240).mean()

            # Exclude_Rank (일자별/시점별 동적 평가)
            if len(ma20_lvl) and len(ma60_lvl) and len(ma120_lvl) and len(ma240_lvl):
                ma20_last = ma20_lvl.iloc[-1]
                ma60_last = ma60_lvl.iloc[-1]
                ma120_last = ma120_lvl.iloc[-1]
                ma240_last = ma240_lvl.iloc[-1]
                close_last = df_for_indicators['종가'].iloc[-1]
                
                cond1 = (
                    pd.notna(ma20_last) and pd.notna(ma60_last) and pd.notna(ma240_last) and pd.notna(ma120_last)
                    and pd.notna(close_last)
                    and (ma20_last < ma120_last)
                    and (ma20_last < ma240_last)
                    and (close_last < ma60_last)
                )

                if len(ma20_lvl) >= 15:
                    ma20_diff = ma20_lvl.diff().dropna()
                    if len(ma20_diff) >= 14:
                        ma20_down_14_days = (ma20_diff.iloc[-14:] < 0).all()
                        cond2 = bool(ma20_down_14_days and (close_last < ma20_last))
                    else:
                        cond2 = False
                else:
                    cond2 = False
                
                # 시가총액 1000억 미만 종목 제외 (원 단위, 백테스트와 동일)
                try:
                    market_cap = latest_data.get('시가총액', None)
                    if pd.isna(market_cap):
                        cond3 = False
                    else:
                        market_cap = float(market_cap)
                        cond3 = bool(market_cap < 100_000_000_000)
                except Exception:
                    cond3 = False

                latest_data['Exclude_Rank'] = bool(cond1 or cond2 or cond3)
            else:
                latest_data['Exclude_Rank'] = False
        except Exception:
            # 계산 실패 시에도 파이프라인은 계속 진행 (제외는 적용하지 않음)
            latest_data['Exclude_Rank'] = False

        latest_data['52주_신고가_비율'] = (df_for_indicators['종가'] / df_for_indicators['종가'].rolling(250).max()).iloc[-1]
        
        # Position_Range_60 계산 (Donchian)
        high_60 = df_for_indicators['고가'].rolling(60).max().iloc[-1]
        low_60 = df_for_indicators['저가'].rolling(60).min().iloc[-1]
        if pd.notna(high_60) and pd.notna(low_60) and (high_60 - low_60) != 0:
            latest_data['Position_Range_60'] = (df_for_indicators['종가'].iloc[-1] - low_60) / (high_60 - low_60)
            latest_data['Position_Range_60'] = max(0, min(1, latest_data['Position_Range_60']))
        else:
            latest_data['Position_Range_60'] = 0.5

        # BB_Position (%B): (종가-하단)/(상단-하단). 1 초과=상단 돌파, 0 미만=하단 이탈. clip 금지.
        try:
            close_s = df_for_indicators['종가']
            bb_mid = close_s.rolling(window=20).mean().iloc[-1]
            bb_std = close_s.rolling(window=20).std().iloc[-1]
            if pd.notna(bb_mid) and pd.notna(bb_std) and bb_std > 0:
                bb_upper = bb_mid + 2.0 * bb_std
                bb_lower = bb_mid - 2.0 * bb_std
                bb_width = bb_upper - bb_lower
                latest_data['BB_Position'] = float(
                    (close_s.iloc[-1] - bb_lower) / bb_width
                ) if bb_width > 0 else 0.5
            else:
                latest_data['BB_Position'] = np.nan
        except Exception as e:
            log_warning(f"BB_Position 계산 실패 ({ticker}): {e}")
            latest_data['BB_Position'] = np.nan
        
        # ADX_14 (pandas_ta adx append 결과)
        if 'ADX_14' in df_for_indicators.columns:
            latest_data['ADX_14'] = df_for_indicators['ADX_14'].iloc[-1]
            
        # HV 변동성 (5일, 20일, 60일)
        try:
            log_ret_1d = np.log(df_for_indicators['종가'] / df_for_indicators['종가'].shift(1))
            latest_data['HV_Volatility_5'] = log_ret_1d.rolling(window=5).std().iloc[-1] if len(log_ret_1d) >= 5 else np.nan
            latest_data['HV_Volatility_20'] = log_ret_1d.rolling(window=20).std().iloc[-1] if len(log_ret_1d) >= 20 else np.nan
            latest_data['HV_Volatility_60'] = log_ret_1d.rolling(window=60).std().iloc[-1] if len(log_ret_1d) >= 60 else np.nan
        except Exception as e:
            log_warning(f"HV 변동성 계산 실패 ({ticker}): {e}")
            latest_data['HV_Volatility_5'] = np.nan
            latest_data['HV_Volatility_20'] = np.nan
            latest_data['HV_Volatility_60'] = np.nan
            

        # Phase C (PLAN_model_improvement_v2): asymmetric / directional features (scalar last row)
        try:
            ret = df_for_indicators['종가'].pct_change()
            if '거래대금' not in df_for_indicators.columns:
                df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
            vol = df_for_indicators['거래대금']
            down = ret.where(ret < 0)
            up = ret.where(ret > 0)
            downside = down.rolling(20, min_periods=10).std()
            up_vol = up.rolling(20, min_periods=10).std()
            latest_data['Downside_Vol_20'] = downside.iloc[-1] if len(downside) else np.nan
            ratio = up_vol / (downside + 1e-8)
            latest_data['Upside_Downside_Vol_Ratio_20'] = ratio.iloc[-1] if len(ratio) else np.nan
            skew = ret.rolling(60, min_periods=30).skew()
            latest_data['Return_Skew_60'] = skew.iloc[-1] if len(skew) else np.nan
            ddr = (ret < 0).rolling(20, min_periods=10).mean()
            latest_data['Down_Day_Ratio_20'] = ddr.iloc[-1] if len(ddr) else np.nan
            down_vol_sum = vol.where(ret < 0, 0.0).rolling(20, min_periods=10).sum()
            all_vol_sum = vol.rolling(20, min_periods=10).sum()
            dvr = down_vol_sum / (all_vol_sum + 1e-8)
            latest_data['Down_Volume_Ratio_20'] = dvr.iloc[-1] if len(dvr) else np.nan
            gap = df_for_indicators['시가'] / df_for_indicators['종가'].shift(1) - 1.0
            gap_m = gap.rolling(20, min_periods=10).mean()
            latest_data['Gap_Mean_20'] = gap_m.iloc[-1] if len(gap_m) else np.nan
            intraday = (df_for_indicators['종가'] - df_for_indicators['시가']) / (
                df_for_indicators['고가'] - df_for_indicators['저가'] + 1e-8
            )
            is20 = intraday.rolling(20, min_periods=10).mean()
            latest_data['Intraday_Strength_20'] = is20.iloc[-1] if len(is20) else np.nan
            amihud = (ret.abs() / (vol + 1e-8)).rolling(20, min_periods=10).mean()
            ami = np.log1p(amihud)
            latest_data['Amihud_Illiq_20'] = ami.iloc[-1] if len(ami) else np.nan
            for _c in (
                'Downside_Vol_20', 'Upside_Downside_Vol_Ratio_20', 'Return_Skew_60',
                'Down_Day_Ratio_20', 'Down_Volume_Ratio_20', 'Gap_Mean_20',
                'Intraday_Strength_20', 'Amihud_Illiq_20',
            ):
                v = latest_data.get(_c, np.nan)
                try:
                    if v is not None and np.isinf(float(v)):
                        latest_data[_c] = np.nan
                except Exception:
                    pass
        except Exception as e:
            log_warning(f"Phase C asymmetric feature calc failed ({ticker}): {e}")
            for _c in (
                'Downside_Vol_20', 'Upside_Downside_Vol_Ratio_20', 'Return_Skew_60',
                'Down_Day_Ratio_20', 'Down_Volume_Ratio_20', 'Gap_Mean_20',
                'Intraday_Strength_20', 'Amihud_Illiq_20',
            ):
                latest_data[_c] = np.nan


        # disparity_20 추가
        try:
            ma20 = df_for_indicators['종가'].rolling(window=20).mean()
            latest_data['disparity_20'] = (df_for_indicators['종가'] / ma20 * 100).iloc[-1]
        except Exception as e:
            log_warning(f"disparity_20 계산 실패 ({ticker}): {e}")
            latest_data['disparity_20'] = np.nan
        
        # VWAP Disparity(1W) (VWAP 괴리율 1주)
        try:
            if len(df_for_indicators) >= 5:
                tp = (df_for_indicators['고가'] + df_for_indicators['저가'] + df_for_indicators['종가']) / 3
                money = tp * df_for_indicators['거래량']
                
                sum_money_5 = money.rolling(window=5).sum().iloc[-1]
                sum_vol_5 = df_for_indicators['거래량'].rolling(window=5).sum().iloc[-1]
                
                if sum_vol_5 > 0:
                    vwap_5 = sum_money_5 / sum_vol_5
                    latest_data['VWAP_Disparity_5'] = (df_for_indicators['종가'].iloc[-1] / vwap_5 - 1) * 100
                else:
                    latest_data['VWAP_Disparity_5'] = np.nan
            else:
                latest_data['VWAP_Disparity_5'] = np.nan
        except Exception as e:
            log_warning(f"VWAP_Disparity_5 계산 실패 ({ticker}): {e}")
            latest_data['VWAP_Disparity_5'] = np.nan

        # Max_Drawdown_20 (최근 20일 최대 낙폭, %)
        # roll_max = 고가.rolling(20).max()
        # daily_dd = (저가 / roll_max) - 1
        # Max_Drawdown_20 = daily_dd.rolling(20).min() * 100
        try:
            roll_max_20 = df_for_indicators['고가'].rolling(window=20).max()
            daily_dd_20 = (df_for_indicators['저가'] / roll_max_20) - 1
            latest_data['Max_Drawdown_20'] = (daily_dd_20.rolling(window=20).min() * 100).iloc[-1]
        except Exception as e:
            log_warning(f"Max_Drawdown_20 계산 실패 ({ticker}): {e}")
            latest_data['Max_Drawdown_20'] = np.nan

        # 등락율(5D): 5거래일 전 종가 대비 누적 등락율 (%)
        try:
            latest_data['등락율(5D)'] = (df_for_indicators['종가'].pct_change(5) * 100).iloc[-1]
        except Exception as e:
            log_warning(f"등락율(5D) 계산 실패 ({ticker}): {e}")
            latest_data['등락율(5D)'] = np.nan

        # Trend_Pullback_Score (내부 MA20_Slope 활용)
        try:
            ma20 = df_for_indicators['종가'].rolling(window=20).mean()
            ma20_slope = calculate_normalized_linear_regression_slope_latest(ma20, window=5)
            mean_20 = df_for_indicators['종가'].rolling(20).mean()
            std_20 = df_for_indicators['종가'].rolling(20).std()
            
            # std_20이 0인 경우 처리 (변동성이 없으면 z_score를 0으로 설정)
            if len(std_20) > 0 and len(mean_20) > 0:
                std_20_clean = std_20.replace(0, np.nan)
                z_score_20 = (df_for_indicators['종가'] - mean_20) / std_20_clean
                z_score_20 = z_score_20.fillna(0)  # std가 0인 경우 z_score를 0으로 설정
                
                if len(z_score_20) > 0:
                    z_score_latest = z_score_20.iloc[-1]
                    # NaN 값 처리
                    if pd.isna(ma20_slope) or pd.isna(z_score_latest):
                        latest_data['Trend_Pullback_Score'] = np.nan
                    else:
                        # 기본 점수 계산: abs(z_score) * ma20_slope
                        base_score = np.abs(z_score_latest) * ma20_slope
                        
                        # 조건별 가중치 적용
                        if ma20_slope > 0 and z_score_latest < 0:
                            # 상승 추세 + 눌림: 최고 점수
                            latest_data['Trend_Pullback_Score'] = base_score * 1.0
                        elif ma20_slope > 0 and z_score_latest >= 0:
                            # 상승 추세 + 과열: 낮은 점수
                            latest_data['Trend_Pullback_Score'] = base_score * 0.3
                        elif ma20_slope <= 0:
                            # 하락 추세: 매우 낮은 점수
                            latest_data['Trend_Pullback_Score'] = base_score * 0.1
                        else:
                            latest_data['Trend_Pullback_Score'] = 0.0
                else:
                    latest_data['Trend_Pullback_Score'] = np.nan
            else:
                latest_data['Trend_Pullback_Score'] = np.nan
        except Exception as e:
            log_warning(f"Trend_Pullback_Score 계산 실패 ({ticker}): {e}")
            latest_data['Trend_Pullback_Score'] = np.nan
        
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price
        latest_data['기준일가'] = reference_date_price
        latest_data['전날종가'] = df_price_full.iloc[-2]['종가'] if len(df_price_full) >= 2 else latest_current_price
        latest_data['종목코드'] = ticker

        return latest_data, actual_analysis_date
    except Exception as e:
        return None, None


# =================================================================
# 2단계: 순수 계산 함수 (ProcessPoolExecutor 용, 네트워크 호출 없음)
# =================================================================

def _calculate_indicators_for_ticker(args):
    """
    단일 종목 지표 계산 (순수 CPU, 네트워크 호출 없음).
    ProcessPoolExecutor에서 호출되도록 설계됨.
    
    Args:
        args: (ticker, name, shares, df_price_dict, selected_analysis_date_ts)
        
    Returns:
        (latest_data_dict, actual_analysis_date) 또는 (None, None)
    """
    ticker, name, shares, price_data, selected_analysis_date_ts = args
    
    try:
        # 가격 데이터 복원 (dict -> DataFrame)
        df_price_full = pd.DataFrame(price_data)
        if df_price_full.empty or len(df_price_full) < 311:
            return None, None
        
        df_price_full['date'] = pd.to_datetime(df_price_full['date'])
        df_price_full.set_index('date', inplace=True)
        df_price_full = df_price_full.sort_index()
        
        # 컬럼명 정규화
        df_price_full.rename(columns={'Open':'시가', 'Close':'종가', 'High':'고가', 'Low':'저가', 'Volume':'거래량'}, inplace=True)
        
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty:
            return None, None
        actual_analysis_date = df_temp.index.max()
        reference_date_price = float(df_temp.loc[actual_analysis_date]['종가'])
        latest_current_price = float(df_price_full.iloc[-1]['종가'])
        
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        latest_data = {}
        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        
        # RVOL
        try:
            vol_20_mean = df_for_indicators['거래량'].rolling(window=20).mean().iloc[-1]
            curr_vol = df_for_indicators['거래량'].iloc[-1]
            latest_data['RVOL'] = curr_vol / vol_20_mean if (pd.notna(vol_20_mean) and vol_20_mean > 0 and pd.notna(curr_vol)) else np.nan
        except:
            latest_data['RVOL'] = np.nan
        
        # 시총 회전율
        try:
            mkt_cap = reference_date_price * shares
            if mkt_cap > 0:
                amt_5_mean = df_for_indicators['거래대금'].rolling(window=5).mean().iloc[-1]
                latest_data['시총 회전율(1W)'] = (amt_5_mean / mkt_cap) * 100 if pd.notna(amt_5_mean) else np.nan
                amt_60_mean = df_for_indicators['거래대금'].rolling(window=60).mean().iloc[-1]
                latest_data['시총 회전율(3M)'] = (amt_60_mean / mkt_cap) * 100 if pd.notna(amt_60_mean) else np.nan
            else:
                latest_data['시총 회전율(1W)'] = np.nan
                latest_data['시총 회전율(3M)'] = np.nan
        except:
            latest_data['시총 회전율(1W)'] = np.nan
            latest_data['시총 회전율(3M)'] = np.nan
        
        latest_data['log_mktcap'] = np.log(mkt_cap) if mkt_cap > 0 else np.nan
        latest_data['시가총액'] = float(mkt_cap) if mkt_cap > 0 else np.nan
        
        # ATR (벡터화)
        atr_5 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=5)
        atr_20 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=20)
        atr_60 = _calc_atr_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=60)
        
        last_close = df_for_indicators['종가'].iloc[-1]
        latest_data['ATRr_5'] = (atr_5.iloc[-1] / last_close) * 100 if len(atr_5) and pd.notna(atr_5.iloc[-1]) else np.nan
        latest_data['ATRr_20'] = (atr_20.iloc[-1] / last_close) * 100 if len(atr_20) and pd.notna(atr_20.iloc[-1]) else np.nan
        latest_data['ATRr_60'] = (atr_60.iloc[-1] / last_close) * 100 if len(atr_60) and pd.notna(atr_60.iloc[-1]) else np.nan
        
        # ADX
        adx_14, _, _ = _calc_adx_vectorized(df_for_indicators['고가'], df_for_indicators['저가'], df_for_indicators['종가'], length=14)
        latest_data['ADX_14'] = adx_14.iloc[-1] if len(adx_14) and pd.notna(adx_14.iloc[-1]) else np.nan
        
        # CLV
        try:
            h, l, c = df_for_indicators['고가'].iloc[-1], df_for_indicators['저가'].iloc[-1], last_close
            latest_data['CLV'] = float(np.clip((2*c - h - l) / (h - l + 1e-8), -1.0, 1.0))
        except:
            latest_data['CLV'] = np.nan
        
        # RSI
        rsi_14 = _calc_rsi_vectorized(df_for_indicators['종가'], length=14)
        if len(rsi_14) >= 9:
            rsi_ma9 = rsi_14.rolling(9).mean().iloc[-1]
            latest_data['RSI_Signal_Oscillator'] = rsi_14.iloc[-1] - rsi_ma9 if pd.notna(rsi_14.iloc[-1]) and pd.notna(rsi_ma9) else np.nan
        else:
            latest_data['RSI_Signal_Oscillator'] = np.nan
        
        # 이격도
        for p in [120, 240]:
            ma = df_for_indicators['종가'].rolling(p).mean()
            latest_data[f'disparity_{p}'] = (last_close / ma.iloc[-1] * 100) if len(ma) and pd.notna(ma.iloc[-1]) else np.nan
        
        # MA Slope
        for period, key in [(120, 'MA120_Slope'), (20, 'MA20_Slope'), (240, 'MA240_Slope')]:
            ma = df_for_indicators['종가'].rolling(period).mean()
            latest_data[key] = calculate_normalized_linear_regression_slope_latest(ma, window=5) if len(ma) >= 5 else np.nan
        
        # Exclude_Rank
        try:
            ma20 = df_for_indicators['종가'].rolling(20).mean()
            ma60 = df_for_indicators['종가'].rolling(60).mean()
            ma120 = df_for_indicators['종가'].rolling(120).mean()
            ma240 = df_for_indicators['종가'].rolling(240).mean()
            
            cond1 = cond2 = cond3 = False
            if all(len(m) for m in [ma20, ma60, ma120, ma240]):
                m20, m60, m120, m240 = ma20.iloc[-1], ma60.iloc[-1], ma120.iloc[-1], ma240.iloc[-1]
                if all(pd.notna(x) for x in [m20, m60, m120, m240, last_close]):
                    cond1 = (m20 < m120) and (m20 < m240) and (last_close < m60)
                
                if len(ma20) >= 15:
                    diff = ma20.diff().dropna()
                    cond2 = len(diff) >= 14 and (diff.iloc[-14:] < 0).all() and (last_close < m20)
                
                cond3 = latest_data['시가총액'] < 100_000_000_000 if pd.notna(latest_data['시가총액']) else False
            
            latest_data['Exclude_Rank'] = bool(cond1 or cond2 or cond3)
        except:
            latest_data['Exclude_Rank'] = False
        
        # 52주 신고가 비율
        latest_data['52주_신고가_비율'] = (last_close / df_for_indicators['종가'].rolling(250).max().iloc[-1]) if len(df_for_indicators) >= 250 else np.nan
        
        # Position_Range_60
        try:
            h60 = df_for_indicators['고가'].rolling(60).max().iloc[-1]
            l60 = df_for_indicators['저가'].rolling(60).min().iloc[-1]
            latest_data['Position_Range_60'] = max(0, min(1, (last_close - l60) / (h60 - l60))) if pd.notna(h60) and pd.notna(l60) and h60 != l60 else 0.5
        except:
            latest_data['Position_Range_60'] = 0.5
        
        # BB_Position
        try:
            bb_mid = df_for_indicators['종가'].rolling(20).mean().iloc[-1]
            bb_std = df_for_indicators['종가'].rolling(20).std().iloc[-1]
            if pd.notna(bb_mid) and pd.notna(bb_std) and bb_std > 0:
                bb_up = bb_mid + 2*bb_std
                bb_low = bb_mid - 2*bb_std
                latest_data['BB_Position'] = (last_close - bb_low) / (bb_up - bb_low) if bb_up != bb_low else 0.5
            else:
                latest_data['BB_Position'] = np.nan
        except:
            latest_data['BB_Position'] = np.nan
        
        # HV 변동성
        try:
            log_ret = np.log(df_for_indicators['종가'] / df_for_indicators['종가'].shift(1))
            latest_data['HV_Volatility_5'] = log_ret.rolling(5).std().iloc[-1] if len(log_ret) >= 5 else np.nan
            latest_data['HV_Volatility_20'] = log_ret.rolling(20).std().iloc[-1] if len(log_ret) >= 20 else np.nan
            latest_data['HV_Volatility_60'] = log_ret.rolling(60).std().iloc[-1] if len(log_ret) >= 60 else np.nan
        except:
            latest_data['HV_Volatility_5'] = latest_data['HV_Volatility_20'] = latest_data['HV_Volatility_60'] = np.nan
        
        # Phase C asymmetric features
        try:
            ret = df_for_indicators['종가'].pct_change()
            vol = df_for_indicators['거래대금']
            down = ret.where(ret < 0)
            up = ret.where(ret > 0)
            downside = down.rolling(20, min_periods=10).std()
            up_vol = up.rolling(20, min_periods=10).std()
            latest_data['Downside_Vol_20'] = downside.iloc[-1] if len(downside) else np.nan
            latest_data['Upside_Downside_Vol_Ratio_20'] = (up_vol.iloc[-1] / (downside.iloc[-1] + 1e-8)) if len(up_vol) and len(downside) and pd.notna(downside.iloc[-1]) else np.nan
            latest_data['Return_Skew_60'] = ret.rolling(60, min_periods=30).skew().iloc[-1] if len(ret) >= 60 else np.nan
            latest_data['Down_Day_Ratio_20'] = (ret < 0).rolling(20, min_periods=10).mean().iloc[-1] if len(ret) >= 20 else np.nan
            down_vol_sum = vol.where(ret < 0, 0).rolling(20, min_periods=10).sum()
            all_vol_sum = vol.rolling(20, min_periods=10).sum()
            latest_data['Down_Volume_Ratio_20'] = (down_vol_sum.iloc[-1] / (all_vol_sum.iloc[-1] + 1e-8)) if len(down_vol_sum) and len(all_vol_sum) else np.nan
            gap = df_for_indicators['시가'] / df_for_indicators['종가'].shift(1) - 1
            latest_data['Gap_Mean_20'] = gap.rolling(20, min_periods=10).mean().iloc[-1] if len(gap) >= 20 else np.nan
            intraday = (df_for_indicators['종가'] - df_for_indicators['시가']) / (df_for_indicators['고가'] - df_for_indicators['저가'] + 1e-8)
            latest_data['Intraday_Strength_20'] = intraday.rolling(20, min_periods=10).mean().iloc[-1] if len(intraday) >= 20 else np.nan
            amihud = (ret.abs() / (vol + 1e-8)).rolling(20, min_periods=10).mean()
            latest_data['Amihud_Illiq_20'] = np.log1p(amihud).iloc[-1] if len(amihud) >= 20 else np.nan
        except:
            for k in ['Downside_Vol_20', 'Upside_Downside_Vol_Ratio_20', 'Return_Skew_60', 'Down_Day_Ratio_20', 'Down_Volume_Ratio_20', 'Gap_Mean_20', 'Intraday_Strength_20', 'Amihud_Illiq_20']:
                latest_data[k] = np.nan
        
        # disparity_20
        ma20 = df_for_indicators['종가'].rolling(20).mean()
        latest_data['disparity_20'] = (last_close / ma20.iloc[-1] * 100) if len(ma20) and pd.notna(ma20.iloc[-1]) else np.nan
        
        # VWAP Disparity 5
        try:
            if len(df_for_indicators) >= 5:
                tp = (df_for_indicators['고가'] + df_for_indicators['저가'] + df_for_indicators['종가']) / 3
                money = tp * df_for_indicators['거래량']
                vol_sum = df_for_indicators['거래량'].rolling(5).sum().iloc[-1]
                money_sum = money.rolling(5).sum().iloc[-1]
                # 거래량 0 방지 (경고 제거)
                if pd.notna(vol_sum) and vol_sum > 0 and pd.notna(money_sum):
                    vwap5 = money_sum / vol_sum
                    latest_data['VWAP_Disparity_5'] = (last_close / vwap5 - 1) * 100 if vwap5 > 0 else np.nan
                else:
                    latest_data['VWAP_Disparity_5'] = np.nan
            else:
                latest_data['VWAP_Disparity_5'] = np.nan
        except:
            latest_data['VWAP_Disparity_5'] = np.nan
        
        # Max_Drawdown_20
        try:
            roll_max = df_for_indicators['고가'].rolling(20).max()
            dd = (df_for_indicators['저가'] / roll_max - 1).rolling(20).min()
            latest_data['Max_Drawdown_20'] = dd.iloc[-1] * 100 if len(dd) and pd.notna(dd.iloc[-1]) else np.nan
        except:
            latest_data['Max_Drawdown_20'] = np.nan
        
        # 등락율(5D)
        latest_data['등락율(5D)'] = df_for_indicators['종가'].pct_change(5).iloc[-1] * 100 if len(df_for_indicators) >= 6 else np.nan
        
        # Trend_Pullback_Score
        try:
            ma20_slope = calculate_normalized_linear_regression_slope_latest(ma20, window=5)
            mean_20 = df_for_indicators['종가'].rolling(20).mean()
            std_20 = df_for_indicators['종가'].rolling(20).std().replace(0, np.nan)
            z_score = (df_for_indicators['종가'] - mean_20) / std_20
            z_score = z_score.fillna(0)
            z_latest = z_score.iloc[-1]
            
            if pd.notna(ma20_slope) and pd.notna(z_latest):
                base = abs(z_latest) * ma20_slope
                if ma20_slope > 0 and z_latest < 0:
                    latest_data['Trend_Pullback_Score'] = base * 1.0
                elif ma20_slope > 0:
                    latest_data['Trend_Pullback_Score'] = base * 0.3
                else:
                    latest_data['Trend_Pullback_Score'] = base * 0.1
            else:
                latest_data['Trend_Pullback_Score'] = np.nan
        except:
            latest_data['Trend_Pullback_Score'] = np.nan
        
        latest_data['종목명'] = name
        latest_data['현재가'] = latest_current_price
        latest_data['기준일가'] = reference_date_price
        latest_data['전날종가'] = float(df_price_full.iloc[-2]['종가']) if len(df_price_full) >= 2 else latest_current_price
        latest_data['종목코드'] = ticker
        
        return latest_data, actual_analysis_date
    except Exception as e:
        return None, None


def _fetch_all_price_data(stock_records, start_date, end_date, max_workers=8):
    """
    1단계: 전 종목 주가 데이터 병렬 수집 (ThreadPoolExecutor, I/O 바운드)
    - 타임아웃 8초 + 재시도 1회로 블로킹 방지
    - 전체 배치 타임아웃: 300초 (5분)
    - 멈춘 워커 즉시 강제 해제
    """
    # KRX 로그인 보장 (메인 프로세스에서 1회)
    _ensure_krx_login()
    
    # 소스 프로브 (첫 종목만)
    if stock_records:
        # 프로브로 성공 소스 결정 (캐시에 저장됨: _SUCCESSFUL_FDR_SOURCE)
        probe_stock_ohlcv_source(stock_records[0]['종목코드'], end_date=end_date)
    
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y-%m-%d')
    
    # 성공한 소스명에서 직접 prefix 구성 (NAVER: 또는 KRX:)
    source_name = _SUCCESSFUL_FDR_SOURCE[0] if _SUCCESSFUL_FDR_SOURCE else 'NAVER'
    source_prefix = 'NAVER:' if source_name == 'NAVER' else 'KRX:'
    log_info(f"주가 수집 소스 고정: {source_prefix} (폴백 생략)")
    
    price_data_map = {}
    failed_tickers = []
    
    # ThreadPoolExecutor로 fetching (I/O 바운드, 세션 재사용)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_start_time = time.time()
        BATCH_TIMEOUT = 1800  # 30분 전체 타임아웃 (2500종목 여유있게)
        
        # 최소 필요 데이터 일수: 핵심 지표(20, 60, 120, 240일 MA) + 버퍼
        MIN_DATA_DAYS = 250  # 240일 MA + 여유
        
        def _fetch_one_direct(stock_info):
            """캐시된 소스로 직접 요청 (폴백 없음, 타임아웃 8초)"""
            ticker = stock_info['종목코드']
            try:
                symbol = f"{source_prefix}{ticker}"
                df = _fetch_fdr_with_retry(symbol, fetch_start, end_date)
                if df is not None and not df.empty and len(df) >= MIN_DATA_DAYS:
                    df_reset = df.reset_index()
                    df_reset.columns = ['date'] + [c for c in df_reset.columns[1:]]
                    df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%d')
                    return ticker, df_reset.to_dict('records')
                elif df is not None and not df.empty:
                    # 데이터는 있지만 부족한 경우 - 디버그 레벨로만 로깅 (신규상장 등 정상 케이스)
                    log_info(f"데이터 부족 스킵 ({ticker}): {len(df)}일 (최소 {MIN_DATA_DAYS}일 필요)")
                else:
                    log_warning(f"데이터 없음 ({ticker})")
            except Exception as e:
                log_warning(f"수집 에러 ({ticker}): {type(e).__name__}: {e}")
            return ticker, None
        
        # 세마포어로 동시 요청 수 제한 (API 과부하 방지)
        import threading
        request_semaphore = threading.Semaphore(4)  # 동시 4개만 허용
        
        def _fetch_one_direct_limited(stock_info):
            with request_semaphore:
                time.sleep(0.05)  # 요청 간 50ms 간격
                return _fetch_one_direct(stock_info)
        
        futures = {executor.submit(_fetch_one_direct_limited, r): r for r in stock_records}
        completed = 0
        total = len(stock_records)
        pending = set(futures.keys())
        
        while pending:
            if time.time() - batch_start_time > BATCH_TIMEOUT:
                log_warning(f"배치 타임아웃({BATCH_TIMEOUT}초) - 남은 {len(pending)}개 종목 강제 취소")
                for fut in pending:
                    fut.cancel()
                    failed_tickers.append(futures[fut]['종목코드'])
                break
            
            done, pending = concurrent.futures.wait(pending, timeout=1.0, 
                                                     return_when=concurrent.futures.FIRST_COMPLETED)
            
            for fut in done:
                try:
                    ticker, data = fut.result()
                    if data:
                        price_data_map[ticker] = data
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    ticker = futures[fut].get('종목코드', 'Unknown')
                    failed_tickers.append(ticker)
                    log_warning(f"종목 처리 예외 ({ticker}): {e}")
                completed += 1
                
                if completed % 10 == 0 or completed == total:
                    log_progress(f"주가 수집 진행", completed, total)
        
        for fut in pending:
            if not fut.done():
                fut.cancel()
                failed_tickers.append(futures[fut]['종목코드'])
    
    if failed_tickers:
        log_warning(f"수집 실패 종목 ({len(failed_tickers)}개): {failed_tickers[:20]}{'...' if len(failed_tickers) > 20 else ''}")
    
    log_info(f"✅ 주가 데이터 수집 완료: {len(price_data_map)}/{total}개 종목 (실패: {len(failed_tickers)}개)")
    return price_data_map


def fetch_all_data(stock_list, selected_analysis_date, use_cache=True):
    """
    전체 데이터 수집 메인 함수 (2단계 파이프라인)
    
    1단계: ThreadPoolExecutor로 주가 데이터 수집 (I/O 바운드)
    2단계: ProcessPoolExecutor로 지표 계산 (CPU 바운드)
    """
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=450)).strftime('%Y-%m-%d')

    latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date, use_cache)

    if selected_analysis_date.date() < today.date():
        log_info(f"과거 분석(기준일={selected_analysis_date.strftime('%Y-%m-%d')}): 기준일의 시가총액 데이터를 수집합니다.")
        try:
            df_marcap_past = fetch_stock_list_for_date(selected_analysis_date)
            stock_list = pd.merge(
                stock_list[['종목코드', '종목명']],
                df_marcap_past[['종목코드', '상장주식수']],
                on='종목코드', how='inner'
            )
            log_info(f"기준일({selected_analysis_date.strftime('%Y-%m-%d')})에 존재했던 {len(stock_list)}개 종목으로 필터링되었습니다.")
        except Exception as e:
            log_warning(f"기준일 시가총액 데이터 수집 실패: {e}")
            log_info("최신 상장주식수 정보를 사용하여 분석을 계속합니다.")
            
    actual_trading_date = get_actual_trading_date(selected_analysis_date)
    selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
    
    # 거시경제 데이터 수집
    log_info("🔄 실시간 거시경제 데이터를 수집합니다")
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        log_warning("거시 경제 데이터 수집 실패, 분석 계속")
    
    stock_records = stock_list.to_dict('records')
    total_stocks = len(stock_records)
    
    # 워커 수 설정
    import os
    io_workers = 6  # NAVER API 연결 제한 고려해 6개로 제한
    cpu_workers = min(8, os.cpu_count() or 4)
    
    log_info(f"📊 1단계: 주가 데이터 수집 시작 ({total_stocks:,}개 종목, I/O 워커 {io_workers}개)")
    
    # 1단계: 주가 데이터 수집 (메인 프로세스, ThreadPool)
    price_data_map = _fetch_all_price_data(stock_records, start_date_for_fetch, end_date_for_fetch, max_workers=io_workers)
    
    if not price_data_map:
        log_error("수집된 주가 데이터가 없습니다.")
        return pd.DataFrame(), None
    
    # 2단계: 지표 계산 (ProcessPoolExecutor, CPU 바운드)
    log_info(f"🧮 2단계: 기술적 지표 계산 시작 (CPU 워커 {cpu_workers}개)")
    
    # 계산에 필요한 인자만 준비 (피클링 최소화)
    calc_args = []
    for r in stock_records:
        ticker = r['종목코드']
        if ticker in price_data_map:
            calc_args.append((
                ticker,
                r['종목명'],
                r['상장주식수'],
                price_data_map[ticker],  # dict list (직렬화 효율적)
                selected_analysis_date_ts
            ))
    
    all_feature_data = []
    all_actual_dates = []
    batch_size = 100
    total_batches = (len(calc_args) + batch_size - 1) // batch_size
    
    for i in range(0, len(calc_args), batch_size):
        batch = calc_args[i:i + batch_size]
        current_batch = i // batch_size + 1
        
        if current_batch == 1:
            log_info("기술적 지표 계산 중...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            futures = {executor.submit(_calculate_indicators_for_ticker, arg): arg for arg in batch}
            completed = 0
            
            for fut in concurrent.futures.as_completed(futures):
                result, analysis_date = fut.result()
                if result and analysis_date:
                    all_feature_data.append(result)
                    all_actual_dates.append(analysis_date)
                completed += 1
                if completed % 20 == 0 or completed == len(batch):
                    log_progress(f"지표 계산 배치 {current_batch}/{total_batches}", completed, len(batch))
        
        log_info(f"   ✅ 배치 {current_batch}/{total_batches} 완료 ({len([x for x in all_feature_data if x])}개 누적)")
        gc.collect()
    
    log_info(f"🎉 전체 종목 지표 계산 완료: {len(all_feature_data):,}개 종목")
    
    if not all_feature_data:
        return pd.DataFrame(), None
    
    # 후처리 (기존 로직 유지)
    final_df = pd.DataFrame(all_feature_data)
    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)
    
    final_df['date'] = pd.to_datetime(final_df['date']).astype('datetime64[ns]')
    macro_df.index = pd.to_datetime(macro_df.index).astype('datetime64[ns]')
    
    try:
        final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    except Exception as e:
        log_warning(f"merge_asof 실패, 일반 merge 사용: {e}")
        macro_df_reset = macro_df.reset_index().rename(columns={'index': 'date'})
        macro_df_reset['date'] = pd.to_datetime(macro_df_reset['date']).astype('datetime64[ns]')
        final_df['date'] = pd.to_datetime(final_df['date']).astype('datetime64[ns]')
        final_df = pd.merge(final_df, macro_df_reset, on='date', how='left')
    
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)
    
    # 횡단면 정규화
    before_cols = set(final_df.columns)
    add_cross_sectional_ranks(final_df, CROSS_SECTIONAL_FEATURE_COLS)
    added_cs = sorted(set(final_df.columns) - before_cols)
    if added_cs:
        log_info(f"   ✅ 횡단면 정규화(_cs) 추가: {len(added_cs)}개 컬럼")
    
    if final_df.empty:
        log_error("피처 생성 후 유효 데이터 없음")
        return pd.DataFrame(), None
    
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0])
    log_info(f"✅ 모든 피처 데이터 생성 완료: {len(final_df):,}개 종목, {len(final_df.columns)}개 피처")
    return final_df, actual_analysis_date_final
