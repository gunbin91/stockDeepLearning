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
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm
import os
from datetime import datetime, timedelta
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

# 모듈 로드 시 즉시 로그인 수행 (아이디: gunbin52, 비밀번호: !rjsqls1357)
login_krx("gunbin52", "!rjsqls1357")

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


def fetch_fdr_ohlcv_with_fallback(category, attempts, start_date, end_date, verbose=True):
    """
    FDR OHLCV 수집 (소스별 폴백).

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
            df = fdr.DataReader(symbol, start_date_str, end_date_str)
            if df is not None and not df.empty:
                if verbose:
                    log_info(f"[{category}] 수집 성공: {source} (symbol={symbol}, rows={len(df)})")
                return df, source
            if verbose:
                log_warning(f"[{category}] 수집 결과 없음: {source} (symbol={symbol})")
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
    """한국 주식 OHLCV 수집 (NAVER → KRX → NAVER 폴백).

    배치 수집 시 종목별 로그 폭주를 막기 위해 기본 verbose=False.
    """
    category = category or f"주가/{ticker}"
    return fetch_fdr_ohlcv_with_fallback(
        category,
        _build_stock_price_attempts(ticker),
        start_date,
        end_date,
        verbose=verbose,
    )


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
        log_info(f"주가 수집 소스: {source} (첫 종목 {ticker}으로 확인, 순서: NAVER → KRX → NAVER)")
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
_FDR_KRX_MARCAP_CACHE_BASE = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/"
    "refs/heads/master/data/listing/krx"
)


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


def _load_fdr_krx_marcap_cache_csv(preferred_date=None, lookback_days=15):
    """FDR GitHub KRX-MARCAP CSV를 preferred_date부터 거슬러가며 로드.

    Returns:
        (DataFrame, 사용된 날짜 datetime) 또는 실패 시 (None, None)
    """
    start = preferred_date
    if start is None:
        start = _get_krx_max_work_date()
    if start is None:
        start = datetime.now()
    start = pd.to_datetime(start).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)

    last_error = None
    for i in range(lookback_days + 1):
        day = start - timedelta(days=i)
        # 주말 CSV는 보통 없음 → 스킵해도 되지만 HEAD 비용이 작아 그대로 시도
        url = f"{_FDR_KRX_MARCAP_CACHE_BASE}/{day.strftime('%Y-%m-%d')}.csv"
        try:
            df = pd.read_csv(
                url,
                index_col=0,
                dtype={"Code": str, "Dept": str, "ChangeCode": str, "MarketId": str},
            )
            df = df.reset_index(drop=True)
            if df is not None and not df.empty and "Code" in df.columns:
                return df, day
        except Exception as e:
            last_error = e
            continue

    log_warning(
        f"FDR GitHub KRX-MARCAP CSV 폴백 실패 "
        f"(시작일={start.strftime('%Y-%m-%d')}, lookback={lookback_days}): {last_error}"
    )
    return None, None


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
        except Exception as e:
            primary_error = e
            log_warning(f"FDR StockListing(KRX-MARCAP) 실패, GitHub 일자 CSV 폴백 시도: {e}")

        if df_marcap is None or (hasattr(df_marcap, "empty") and df_marcap.empty):
            preferred = analysis_date if analysis_date is not None else None
            df_marcap, cache_fallback_date = _load_fdr_krx_marcap_cache_csv(preferred_date=preferred)
            if df_marcap is None or df_marcap.empty:
                raise primary_error or RuntimeError("KRX-MARCAP 캐시 CSV를 찾을 수 없습니다.")
            log_info(
                f"종목목록/시총 스냅샷 폴백 사용: FDR GitHub CSV "
                f"({cache_fallback_date.strftime('%Y-%m-%d')}, 오늘자 없음/StockListing 실패)"
            )
        
        # 스팩, 리츠 제외
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        
        # KONEX 제외 (KOSPI, KOSDAQ만 포함)
        if 'Market' in df_marcap.columns:
            df_marcap = df_marcap[df_marcap['Market'].isin(['KOSPI', 'KOSDAQ'])].copy()
            log_info(f"KONEX 제외 후 종목 수: {len(df_marcap)}개")
        
        # 상장주식수가 있는 경우만 필터링
        if 'Stocks' in df_marcap.columns:
            df_marcap = df_marcap[df_marcap['Stocks'] > 0]
        
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
        # Exclude_Rank(cond3: 시총 1000억 미만) 계산에 사용되므로
        # 시가총액(억) 값은 제외 규칙 평가 전에 먼저 세팅합니다.
        try:
            latest_data['시가총액'] = (reference_date_price * shares) / 1_0000_0000
        except Exception:
            latest_data['시가총액'] = np.nan
        
        # ATR 계산 (5일, 20일, 60일)
        atr_5 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=5)
        atr_20 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=20)
        atr_60 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=60)
        
        # ATRr_5 계산 (기준 - 1W): "최근 1주일 변동성 수준"
        if atr_5 is not None:
            latest_data['ATRr_5'] = (atr_5.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_5'] = np.nan
        
        # ATRr_20 계산 (기준 - 1M): "이 종목의 기초 체급은?"
        if atr_20 is not None:
            latest_data['ATRr_20'] = (atr_20.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_20'] = np.nan
        
        # ATRr_60 계산 (기준 - 3M): "중기 추세적 변동성"
        if atr_60 is not None:
            latest_data['ATRr_60'] = (atr_60.iloc[-1] / df_for_indicators['종가'].iloc[-1]) * 100
        else:
            latest_data['ATRr_60'] = np.nan
        
        df_for_indicators.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        
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
        
        # RSI_14 계산
        rsi_14 = df_for_indicators.ta.rsi(close='종가', length=14)
        
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
                
                # 시가총액 1000억 미만 종목 제외 (당일 조건, 억 단위 값 사용)
                # latest_data['시가총액']은 억 단위로 저장됨
                try:
                    market_cap_billion = latest_data.get('시가총액', None)
                    if pd.isna(market_cap_billion):
                        cond3 = False
                    else:
                        # 값이 문자열 등으로 들어오는 경우를 대비해 숫자로 강제 변환
                        market_cap_billion = float(market_cap_billion)
                        cond3 = bool(market_cap_billion < 1000)
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

def fetch_all_data(stock_list, selected_analysis_date, use_cache=True):
    """
    전체 데이터 수집 메인 함수
    
    주식 분석에 필요한 모든 데이터를 수집하고 처리합니다:
    1. 재무 데이터 수집 (PER, PBR, ROE 등)
    2. 주가 데이터 수집 및 기술적 지표 계산
    3. 거시경제 데이터 수집 (KOSPI, 환율, VIX 등)
    4. 데이터 정제 및 병합
    
    Args:
        stock_list: 분석할 종목 목록
        selected_analysis_date: 분석 기준일
        use_cache: 캐시 사용 여부
        
    Returns:
        tuple: (처리된 데이터프레임, 실제 분석일)
    """
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=450)).strftime('%Y-%m-%d')

    latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date, use_cache)
    # 재무데이터는 현재 피처/학습 파이프라인에서 사용하지 않으므로
    # 빈 DF라도 분석을 중단하지 않음

    if selected_analysis_date.date() < today.date():
        log_info(f"과거 분석(기준일={selected_analysis_date.strftime('%Y-%m-%d')}): 기준일의 시가총액 데이터를 수집합니다.")
        try:
            # 통일된 함수 사용
            df_marcap_past = fetch_stock_list_for_date(selected_analysis_date)
            
            stock_list = pd.merge(
                stock_list[['종목코드', '종목명']],
                df_marcap_past[['종목코드', '상장주식수']],
                on='종목코드',
                how='inner'
            )
            log_info(f"기준일({selected_analysis_date.strftime('%Y-%m-%d')})에 존재했던 {len(stock_list)}개 종목으로 필터링되었습니다.")
        except Exception as e:
            log_warning(f"기준일({selected_analysis_date.strftime('%Y-%m-%d')})의 시가총액 데이터를 가져오는 데 실패했습니다: {e}")
            log_info("최신 상장주식수 정보를 사용하여 분석을 계속합니다.")
            
    # 공통 함수를 사용하여 실제 거래일 확인
    actual_trading_date = get_actual_trading_date(selected_analysis_date)
    today = datetime.now().date()
    is_today_analysis = actual_trading_date == today
    
    # 실시간 거시경제 데이터 수집 (캐시 사용 안함)
    log_info("🔄 실시간 거시경제 데이터를 수집합니다")
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        log_warning("거시 경제 데이터 수집에 실패했지만 분석을 계속합니다.")
        # 거시경제 데이터 없이도 분석 계속
    all_feature_data, all_actual_dates = [], []
    stock_records = stock_list.to_dict('records')
    
    # 배치 단위로 처리하여 메모리 효율성 향상
    batch_size = 100
    total_batches = (len(stock_records) + batch_size - 1) // batch_size
    total_stocks = len(stock_records)
    
    log_info(f"주식 분석 시작: {total_stocks:,}개 종목을 {total_batches}개 그룹으로 처리 (예상 5-10분)")
    if stock_records:
        probe_stock_ohlcv_source(stock_records[0]['종목코드'], end_date=end_date_for_fetch)
    
    for i in range(0, len(stock_records), batch_size):
        batch = stock_records[i:i + batch_size]
        current_batch = i // batch_size + 1
        batch_start = i + 1
        batch_end = min(i + batch_size, total_stocks)
        
        # 첫 번째 그룹에서만 상세 설명 표시
        if current_batch == 1:
            log_info("가격, 거래량, 기술적 지표 계산 중...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 워커 수 감소
            future_to_stock = {executor.submit(fetch_and_process_ticker_data, r, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): r for r in batch}
            
            # 전체 진행률을 포함한 진행률 표시
            total_processed = (current_batch - 1) * batch_size
            total_remaining = total_stocks - total_processed
            
            # 통합된 진행률 표시
            completed_count = 0
            total_count = len(batch)
            
            # 배치 시작 로그
            log_info(f"그룹 {current_batch}/{total_batches} 처리 시작 ({total_count}개 종목)")
            
            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    result, analysis_date = future.result()
                    if result and analysis_date: 
                        all_feature_data.append(result)
                        all_actual_dates.append(analysis_date)
                    
                    completed_count += 1
                    progress_percent = (completed_count / total_count) * 100
                    
                    # 매번 진행률 업데이트 (같은 줄에서) - PROGRESS 접두사 유지
                    if completed_count % 10 == 0 or completed_count == total_count:
                        log_progress(f"그룹 {current_batch}/{total_batches} 처리 중", 
                                   completed_count, total_count,
                                   context={'batch': current_batch, 'total_batches': total_batches})
                    
                except Exception as e: 
                    completed_count += 1
                    log_warning(f"종목 처리 중 오류 발생: {e}", 
                               context={'batch': current_batch, 'stock': future_to_stock.get(future, {}).get('종목명', 'Unknown')})
                    continue
            
            # 배치 완료 로그
            log_info(f"그룹 {current_batch}/{total_batches} 처리 완료 ({completed_count}/{total_count}개 종목)")
        
        # 배치 간 메모리 정리
        gc.collect()
        time.sleep(0.1)  # API 부하 방지
        
        success_count = len(all_feature_data)
        progress_percent = (current_batch / total_batches) * 100
        log_info(f"   ✅ 그룹 {current_batch}/{total_batches} 완료! ({progress_percent:.1f}% 진행, {success_count:,}개 종목 수집)")
    
    log_info(f"🎉 종목 데이터 수집 완료! 총 {len(all_feature_data):,}개 종목")
    log_info("🔄 데이터 정제 및 거시경제 지표 병합 중...")
    
    if not all_feature_data: return pd.DataFrame(), None
    final_df = pd.DataFrame(all_feature_data)

    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    
    # 인덱스를 datetime으로 확실히 변환하고 dtype 통일 (나노초로 통일)
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)
    
    # 날짜 dtype 통일 (나노초로 통일하여 merge_asof 호환성 보장)
    final_df['date'] = pd.to_datetime(final_df['date']).astype('datetime64[ns]')
    macro_df.index = pd.to_datetime(macro_df.index).astype('datetime64[ns]')
    
    log_info("   🔗 거시경제 지표(KOSPI, USD/KRW, VIX)를 종목 데이터와 병합 중...")
    
    # merge_asof 대신 더 안전한 방법 사용
    try:
        final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    except Exception as e:
        log_warning(f"   ⚠️ merge_asof 실패, 일반 merge로 시도: {e}")
        # 일반 merge로 대체 (날짜 dtype 통일 보장)
        macro_df_reset = macro_df.reset_index()
        macro_df_reset.rename(columns={'index': 'date'}, inplace=True)
        # 날짜 dtype 통일
        macro_df_reset['date'] = pd.to_datetime(macro_df_reset['date']).astype('datetime64[ns]')
        final_df['date'] = pd.to_datetime(final_df['date']).astype('datetime64[ns]')
        final_df = pd.merge(final_df, macro_df_reset, on='date', how='left')
    
    # Relative_Strength_20 피처는 제거됨
    # 임시 컬럼 정리
    if '종목_수익률_20일_임시' in final_df.columns:
        final_df.drop(columns=['종목_수익률_20일_임시'], inplace=True, errors='ignore')
    
    log_info("   🧹 무한대 값 및 결측값 정제 중...")
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)
    
    if final_df.empty:
        log_error("피처 생성 후 유효한 데이터가 없습니다.")
        return pd.DataFrame(), None
        
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0])
    
    log_info("✅ 모든 피처 데이터 생성 완료!")
    log_info(f"   📊 최종 데이터: {len(final_df):,}개 종목, {len(final_df.columns)}개 피처")
    return final_df, actual_analysis_date_final