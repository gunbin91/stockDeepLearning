"""
주식 데이터 수집 모듈 (미국 주식: NYSE/NASDAQ/AMEX)
============================================

이 파일은 미국 주식 분석에 필요한 데이터를 수집합니다.
- 종목 목록 수집 (NYSE/NASDAQ/AMEX)
- (재무 데이터) 외부 재무 API 없이 파이프라인 호환 컬럼을 NaN으로 채움
- 주가 데이터 수집 (가격, 거래량, 기술적 지표)
- 거시경제 데이터 수집 (IXIC, VIX)

주요 기능:
- 실시간 데이터 수집
- 기술적 지표 자동 계산
- 데이터 품질 검증 및 정제
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import pandas_ta as ta
import concurrent.futures
import threading
from tqdm import tqdm
import os
from datetime import datetime, timedelta
import time
import gc
import locale
import re
import platform
from functools import lru_cache
import json
import urllib.request

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

# =================================================================
# 일봉 데이터 조회 유틸 (Yahoo 우선, 실패 시 FDR 폴백)
# - yfinance의 end는 exclusive이므로 +1일 보정
# - 컬럼은 Open/High/Low/Close/Volume로 정규화
# =================================================================
_YF_LOCK = threading.Lock()

def _normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
    except Exception:
        pass
    return df

def _to_yyyy_mm_dd(dt_like):
    try:
        return pd.to_datetime(dt_like).strftime('%Y-%m-%d')
    except Exception:
        return dt_like

def _is_krx_code(symbol: str) -> bool:
    try:
        return re.match(r'^\d{5}[0-9KLMN]$', str(symbol).strip()) is not None
    except Exception:
        return False

def _is_yahoo_only_symbol(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    return s in {'IXIC', '^IXIC', 'VIX', '^VIX', 'DJI', '^DJI', 'S&P500', 'US500', '^GSPC'}

def _extract_batch_ticker_df(batch_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return None
    if not isinstance(batch_df.columns, pd.MultiIndex):
        return batch_df.copy()
    # level 0: fields, level 1: tickers (기본)
    try:
        if 'Open' in batch_df.columns.levels[0]:
            df_t = batch_df.xs(ticker, level=1, axis=1)
        elif 'Open' in batch_df.columns.levels[1]:
            df_t = batch_df.xs(ticker, level=0, axis=1)
        else:
            df_t = None
    except Exception:
        df_t = None
    return df_t

def fetch_daily_ohlcv_batch(tickers, start=None, end=None) -> dict:
    """
    배치 다운로드용 일봉 OHLCV 조회 (Yahoo)
    Returns: {ticker: df}
    """
    result = {str(t).strip(): None for t in tickers}
    try:
        import yfinance as yf
        yf_start = _to_yyyy_mm_dd(start) if start is not None else None
        yf_end = None
        if end is not None:
            try:
                yf_end = (pd.to_datetime(end) + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception:
                yf_end = end

        batch_symbols = " ".join([str(t).strip() for t in tickers])
        with _YF_LOCK:
            batch_df = yf.download(
                batch_symbols,
                start=yf_start,
                end=yf_end,
                interval='1d',
                progress=False,
                auto_adjust=False,
                threads=False
            )
        if batch_df is None or batch_df.empty:
            return result

        for t in result.keys():
            df_t = _extract_batch_ticker_df(batch_df, t)
            if df_t is None or df_t.empty:
                continue
            df_t = df_t.copy()
            df_t.index = pd.to_datetime(df_t.index)
            result[t] = df_t
        return result
    except Exception as e:
        log_warning(f"[Yahoo] 배치 일봉 데이터 조회 실패: {e}")
        return result

def fetch_daily_ohlcv(symbol: str, start=None, end=None) -> pd.DataFrame:
    """
    일봉 OHLCV 데이터 조회 (Yahoo 우선, 실패 시 FDR 폴백)
    """
    yf_df = None
    try:
        import yfinance as yf
        yf_symbol = str(symbol).strip()
        if yf_symbol.upper() in ('IXIC', '^IXIC'):
            yf_symbol = '^IXIC'
        elif yf_symbol.upper() in ('VIX', '^VIX'):
            yf_symbol = '^VIX'

        yf_start = _to_yyyy_mm_dd(start) if start is not None else None
        yf_end = None
        if end is not None:
            try:
                yf_end = (pd.to_datetime(end) + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception:
                yf_end = end

        # yfinance는 멀티스레드 호출 시 결과가 섞일 수 있어 전역 락으로 직렬화
        with _YF_LOCK:
            yf_df = yf.download(
                yf_symbol,
                start=yf_start,
                end=yf_end,
                interval='1d',
                progress=False,
                auto_adjust=False,
                threads=False
            )
        yf_df = _normalize_yfinance_columns(yf_df)
    except Exception as e:
        log_warning(f"[Yahoo] 일봉 데이터 조회 실패 ({symbol}): {e}")
        yf_df = None

    if yf_df is not None and not yf_df.empty:
        try:
            yf_df.index = pd.to_datetime(yf_df.index)
        except Exception:
            pass
        return yf_df

    # Yahoo 전용 심볼/미국 티커는 FDR도 Yahoo 경로를 타므로 폴백 실익이 적음
    if (not _is_krx_code(symbol)) or _is_yahoo_only_symbol(symbol):
        log_warning(f"[Yahoo] 일봉 데이터 조회 실패, FDR 폴백 생략 ({symbol})")
        return None

    try:
        fdr_df = fdr.DataReader(symbol, start, end)
        if fdr_df is not None and not fdr_df.empty:
            fdr_df.index = pd.to_datetime(fdr_df.index)
        return fdr_df
    except Exception as e:
        log_warning(f"[Fallback] 일봉 데이터 조회 실패 ({symbol}): {e}")
        return None

    return None

# =================================================================
# 미국 시가총액 통일 소스: Nasdaq Screener API
# - Yahoo(yfinance)는 환경에 따라 401(Unauthorized/crumb)로 막힐 수 있어 기본 경로에서 배제
# - 한 번 호출로 다수 심볼의 marketCap(USD)을 가져올 수 있어 대량 티커에도 안정적
# =================================================================
@lru_cache(maxsize=1)
def _get_nasdaq_screener_marketcap_map() -> dict:
    """
    NASDAQ Screener API에서 symbol -> marketCap(USD) 매핑을 가져옵니다.
    Returns:
        dict: { 'AAPL': 4004539426530.0, ... }
    """
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8", "replace"))
    rows = (data.get("data") or {}).get("rows") or []

    def _to_float_marketcap(v):
        if v is None:
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.upper() in ("N/A", "NA", "NULL", "NONE", "-"):
            return np.nan
        s = s.replace("$", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return np.nan

    m = {}
    for r in rows:
        sym = str(r.get("symbol") or "").strip()
        if not sym:
            continue
        mc = _to_float_marketcap(r.get("marketCap"))
        if np.isfinite(mc) and mc > 0:
            m[sym] = mc
    return m

# =================================================================
# 실시간 시가총액/상장주식수 조회 (NASDAQ): yfinance 사용 (파일 캐시 없음)
# =================================================================

@lru_cache(maxsize=10000)
def _get_realtime_marketcap_and_shares_yf(ticker: str):
    """
    yfinance로 실시간 marketCap / sharesOutstanding을 조회합니다.
    - 파일 캐시는 사용하지 않음(요청대로)
    - 동일 실행 내 중복 호출만 LRU로 줄임(네트워크 폭발 방지)

    Returns:
        (marketcap_usd, shares_outstanding)  둘 다 실패하면 (np.nan, np.nan)
    """
    try:
        import yfinance as yf
    except Exception:
        return (np.nan, np.nan)

    t = str(ticker).strip()
    # Yahoo Finance 티커 관례 보정:
    # - 클래스주/특수 티커는 '.' 대신 '-'를 사용하는 경우가 많음 (예: BRK.B -> BRK-B)
    t = t.replace(".", "-")
    if not t:
        return (np.nan, np.nan)

    marketcap = np.nan
    shares = np.nan
    try:
        tk = yf.Ticker(t)

        # 1) fast_info 우선 (가볍고 빠름)
        try:
            fi = getattr(tk, "fast_info", None)
            if fi:
                # yfinance 버전에 따라 key명이 다를 수 있어 방어적으로 처리
                mc = fi.get("market_cap") or fi.get("marketCap")
                sh = fi.get("shares") or fi.get("sharesOutstanding")
                if mc is not None and np.isfinite(float(mc)) and float(mc) > 0:
                    marketcap = float(mc)
                if sh is not None and np.isfinite(float(sh)) and float(sh) > 0:
                    shares = float(sh)
        except Exception:
            pass

        # 2) info fallback (무겁지만 더 많은 티커에서 동작)
        if not (np.isfinite(marketcap) or np.isfinite(shares)):
            try:
                info = tk.get_info()
                mc = info.get("marketCap")
                sh = info.get("sharesOutstanding")
                if mc is not None and np.isfinite(float(mc)) and float(mc) > 0:
                    marketcap = float(mc)
                if sh is not None and np.isfinite(float(sh)) and float(sh) > 0:
                    shares = float(sh)
            except Exception:
                pass

        return (marketcap, shares)
    except Exception:
        return (np.nan, np.nan)
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
    # NASDAQ 전용: 로컬/WSL 시간대가 KST여도 일자 판단이 흔들리지 않도록 뉴욕 시간 기준 사용
    try:
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        ny_now = datetime.utcnow()
    today = ny_now.date()
    selected_date = selected_analysis_date.date()
    
    # AAPL로 실제 거래일 확인 (NASDAQ/US 기준)
    try:
        sample_fetch_start = (ny_now - timedelta(days=10)).strftime('%Y-%m-%d')
        sample_df = fetch_daily_ohlcv('AAPL', sample_fetch_start, ny_now.strftime('%Y-%m-%d'))
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
    # 항상 실시간 재무데이터 수집 (캐시 사용 안함)
    log_step("실시간 재무데이터 수집", "START", {"모드": "실시간 수집"})
    return _fetch_realtime_financial_data(stock_list, selected_analysis_date)

def _fetch_realtime_financial_data(stock_list, selected_analysis_date):
    """실시간 재무데이터 수집 (NASDAQ 버전: 외부 재무 API 미사용, NaN 채움)"""
    try:
        log_info("📊 재무데이터 수집: NASDAQ 버전은 재무 데이터 없이 진행합니다(PER/PBR 등은 NaN).")

        if stock_list is None or stock_list.empty:
            return pd.DataFrame()

        result_df = stock_list[['종목코드']].copy()
        # 학습/분석 파이프라인 호환 컬럼 (필요시 추가)
        for col in ['PER', 'PBR', 'ROE', 'EPS', 'BPS', 'DIV', 'DPS']:
            result_df[col] = np.nan
        result_df['date'] = pd.to_datetime(get_actual_trading_date(selected_analysis_date))

        return result_df
    except Exception as e:
        log_error(f"재무데이터 수집 실패: {e}")
        # NASDAQ 버전에서는 재시도 루프를 돌지 않고 빈 데이터로 폴백
        if stock_list is None or stock_list.empty:
            return pd.DataFrame()
        result_df = stock_list[['종목코드']].copy()
        for col in ['PER', 'PBR', 'ROE', 'EPS', 'BPS', 'DIV', 'DPS']:
            result_df[col] = np.nan
        result_df['date'] = pd.to_datetime(get_actual_trading_date(selected_analysis_date))
        return result_df



def _get_stock_list_from_marcap(analysis_date=None):
    """미국 주식 목록 가져오기 (NYSE/NASDAQ/AMEX 통일)"""
    try:
        exchanges = ["NASDAQ", "NYSE", "AMEX"]
        stock_lists = []

        for exchange in exchanges:
            log_info(f"FinanceDataReader를 통해 {exchange} 종목 리스트 수집...")
            df_list = fdr.StockListing(exchange)
            if df_list is None or df_list.empty:
                log_warning(f"{exchange} 종목 리스트가 비어있습니다. (계속 진행)")
                continue

            # FDR 버전에 따라 컬럼명이 다를 수 있어 방어적으로 처리
            symbol_col = 'Symbol' if 'Symbol' in df_list.columns else ('Code' if 'Code' in df_list.columns else None)
            name_col = 'Name' if 'Name' in df_list.columns else ('Security Name' if 'Security Name' in df_list.columns else None)
            if symbol_col is None:
                log_warning(f"{exchange} 리스트에서 심볼 컬럼을 찾을 수 없습니다. columns={list(df_list.columns)}")
                continue
            if name_col is None:
                # 종목명 컬럼이 없는 경우에도 파이프라인은 동작하도록 심볼을 이름으로 사용
                name_col = symbol_col

            stock_list = df_list[[symbol_col, name_col]].copy()
            stock_list.rename(columns={symbol_col: '종목코드', name_col: '종목명'}, inplace=True)
            stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.strip()
            stock_list['종목명'] = stock_list['종목명'].astype(str).str.strip()
            stock_list['시장구분'] = exchange

            # 시가총액(기준일) 힌트
            if 'MarketCap' in df_list.columns:
                stock_list['시가총액_기준일'] = pd.to_numeric(df_list['MarketCap'], errors='coerce')
            elif 'Marcap' in df_list.columns:
                stock_list['시가총액_기준일'] = pd.to_numeric(df_list['Marcap'], errors='coerce')
            else:
                stock_list['시가총액_기준일'] = np.nan

            stock_lists.append(stock_list)

        if not stock_lists:
            raise DataFetchError("미국(NYSE/NASDAQ/AMEX) 종목 리스트가 비어있습니다.", source="FinanceDataReader")

        stock_list = pd.concat(stock_lists, ignore_index=True)

        # 중복 티커 제거 (시장 중복 상장 대비)
        before_dedup = len(stock_list)
        stock_list = stock_list.drop_duplicates(subset=['종목코드'], keep='first')
        dup_count = before_dedup - len(stock_list)
        if dup_count > 0:
            log_warning(f"중복 티커 {dup_count}개 제거됨 (종목코드 기준)")

        stock_list['상장주식수'] = np.nan

        # Screener API marketCap 우선 적용 (리스트 값이 부정확한 경우 보정)
        try:
            screener_map = _get_nasdaq_screener_marketcap_map()
            tickers = stock_list['종목코드'].astype(str).str.strip()
            mapped = tickers.map(screener_map)
            mapped2 = tickers.str.replace('.', '-', regex=False).map(screener_map)
            combined = mapped.combine_first(mapped2)
            valid_mask = pd.to_numeric(combined, errors='coerce') > 0
            if valid_mask.any():
                stock_list.loc[valid_mask, '시가총액_기준일'] = combined[valid_mask].values
                log_info(f"Screener 시가총액 우선 적용: {int(valid_mask.sum()):,}개")
        except Exception as e:
            log_warning(f"Screener 시가총액 보정 실패(무시하고 진행): {e}")

        # 시가총액 조회 불가 종목 제외
        mktcap = pd.to_numeric(stock_list['시가총액_기준일'], errors='coerce')
        before_filter = len(stock_list)
        stock_list = stock_list[mktcap > 0].copy()
        removed = before_filter - len(stock_list)
        if removed > 0:
            log_info(f"시가총액 미확인 종목 {removed}개 제외")

        log_info(f"총 {len(stock_list)}개 미국(NYSE/NASDAQ/AMEX) 종목을 찾았습니다.")
        return stock_list
        
    except Exception as e:
        error_msg = f"FinanceDataReader API 통신 실패 (NASDAQ StockListing): {e}"
        log_error(error_msg)
        raise DataFetchError(error_msg, source="FinanceDataReader")

def fetch_stock_list():
    """현재 날짜 기준 주식 목록 가져오기 (캐시 없이 실시간 수집)"""
    return _get_stock_list_from_marcap()

def fetch_stock_list_for_date(analysis_date):
    """특정 날짜 기준 주식 목록 가져오기 (캐시 없음)"""
    return _get_stock_list_from_marcap(analysis_date)

def _fetch_macro_data(start_date, end_date):
    log_info(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
    try:
        # 날짜 형식을 명시적으로 변환
        start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        
        # 각 데이터 소스별로 개별 처리 (오류 방지)
        macro_data = {}
        
        try:
            ixic = fetch_daily_ohlcv('IXIC', start_date_str, end_date_str)
            if not ixic.empty:
                ixic_copy = ixic.copy()
                ixic_copy.index = pd.to_datetime(ixic_copy.index, format='mixed', errors='coerce')
                macro_data['IXIC'] = ixic_copy['Close']
        except Exception as e:
            log_warning(f"IXIC 데이터 수집 실패: {e}")
        
        try:
            vix = fetch_daily_ohlcv('^VIX', start_date_str, end_date_str)
            if not vix.empty:
                # 날짜 인덱스를 안전하게 처리 (format='mixed' 사용)
                vix_copy = vix.copy()
                vix_copy.index = pd.to_datetime(vix_copy.index, format='mixed', errors='coerce')
                macro_data['VIX'] = vix_copy['Close']
        except Exception as e:
            log_warning(f"VIX 데이터 수집 실패: {e}")
            
        if macro_data:
            macro_df = pd.concat(macro_data.values(), axis=1, keys=macro_data.keys()).ffill()
            # pct_1d는 IXIC만 생성 (VIX는 생성하지 않음)
            if 'IXIC' in macro_df.columns:
                macro_df['IXIC_pct_1d'] = macro_df['IXIC'].pct_change(1)
            if 'VIX' in macro_df.columns:
                # VIX_pct_1d는 생성하지 않음 (삭제 요청)
                pass
            
            # IXIC 이격도 및 MA20_Slope 계산
            if 'IXIC' in macro_df.columns:
                ixic_close = macro_df['IXIC']
                
                # 이격도 계산 (종목 데이터와 동일한 방식)
                # 이격도 = (현재가 / 이동평균) * 100
                for period in [20]:
                    ma = ixic_close.rolling(window=period).mean()
                    macro_df[f'IXIC_disparity_{period}'] = (ixic_close / ma) * 100
                
                # (NASDAQ 전용) 국내 지표 계산은 사용하지 않음
                # try:
                #     kospi_std_20 = kospi_close.rolling(window=20).std()
                #     kospi_mean_20 = kospi_close.rolling(window=20).mean()
                #     macro_df['KOSPI_변동성(1M)'] = kospi_std_20 / kospi_mean_20
                # except Exception as e:
                #     log_warning(f"KOSPI 변동성(1M) 계산 실패: {e}")
                #     macro_df['KOSPI_변동성(1M)'] = np.nan
                
                # IXIC_MA20_Slope 계산 (IXIC 20일 이동평균선 기울기)
                try:
                    from data_processor import calculate_normalized_linear_regression_slope
                    ixic_ma20 = ixic_close.rolling(window=20).mean()
                    macro_df['IXIC_MA20_Slope'] = calculate_normalized_linear_regression_slope(ixic_ma20, window=20)
                except Exception as e:
                    log_warning(f"IXIC_MA20_Slope 계산 실패: {e}")
                    macro_df['IXIC_MA20_Slope'] = np.nan

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

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df, df_price_full=None):
    ticker = stock_info['종목코드']
    shares = stock_info.get('상장주식수', np.nan)
    try:
        fetch_start = (pd.to_datetime(start_date_for_fetch) - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # NASDAQ 일봉 주가 데이터 수집 (티커 그대로)
        if df_price_full is None:
            try:
                df_price_full = fetch_daily_ohlcv(ticker, fetch_start, end_date_for_fetch)
            except:
                df_price_full = None
        
        if df_price_full is None or df_price_full.empty or len(df_price_full) < 251 + 60: return None, None
        df_price_full.rename(columns={'Open':'시가', 'Close':'종가', 'High':'고가', 'Low':'저가', 'Volume':'거래량'}, inplace=True)

        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)

        # -----------------------------------------------------------------
        # NASDAQ 버전 거래/가격 보정 (코스피 버전의 "삼성전자 거래일 보정"과 동일한 철학)
        # - 선택 기준일 <= 구간에서 "종가가 유효한 마지막 날짜"를 실제 기준일로 사용
        # - 최신 현재가도 "종가가 유효한 마지막 값"을 사용 (NaN이면 N/A로 떨어지는 문제 방지)
        # - 전날종가도 "기준일 이전의 마지막 유효 종가"를 사용
        # -----------------------------------------------------------------
        df_upto = df_price_full[df_price_full.index <= selected_analysis_date_ts].copy()
        if df_upto.empty:
            return None, None

        # 종가 유효값 기준으로 실제 분석일 찾기
        if '종가' not in df_upto.columns:
            return None, None

        df_upto_valid = df_upto[df_upto['종가'].notna()].copy()
        if df_upto_valid.empty:
            # 해당 기간에 유효 종가가 없으면 스킵
            return None, None

        actual_analysis_date = df_upto_valid.index.max()
        reference_date_price = float(df_upto_valid.loc[actual_analysis_date]['종가'])

        # 최신 현재가: 전체 구간에서 종가 유효 마지막 값 사용
        df_full_valid = df_price_full[df_price_full['종가'].notna()]
        latest_current_price = float(df_full_valid.iloc[-1]['종가']) if not df_full_valid.empty else float('nan')

        # 전날종가: 기준일 이전 유효 종가
        df_prev_valid = df_upto_valid[df_upto_valid.index < actual_analysis_date]
        prev_close = float(df_prev_valid.iloc[-1]['종가']) if not df_prev_valid.empty else latest_current_price
        
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        fs_data = pd.DataFrame()
        if latest_fs_df is not None and not latest_fs_df.empty and '종목코드' in latest_fs_df.columns:
            fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        
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
        
        # RVOL(1W): 5일 평균 거래량 / 20일 평균 거래량
        try:
            거래량_5일_평균 = df_for_indicators['거래량'].rolling(window=5).mean().iloc[-1]
            거래량_20일_평균 = df_for_indicators['거래량'].rolling(window=20).mean().iloc[-1]
            if pd.notna(거래량_5일_평균) and pd.notna(거래량_20일_평균) and 거래량_20일_평균 > 0:
                latest_data['RVOL(1W)'] = 거래량_5일_평균 / 거래량_20일_평균
            else:
                latest_data['RVOL(1W)'] = np.nan
        except Exception as e:
            log_warning(f"RVOL(1W) 계산 실패 ({ticker}): {e}")
            latest_data['RVOL(1W)'] = np.nan
        
        # --------------------------------------------------------------
        # 시가총액 통일 (NASDAQ Screener API 기반 marketCap)
        # - Yahoo(yfinance)는 현재 환경에서 401로 막힐 수 있어 기본 경로에서 사용하지 않음
        # - 시총 피처(log_mktcap/시총 회전율) 목적에는 marketCap(USD) 자체가 가장 안정적
        # --------------------------------------------------------------
        marketcap_usd_ref = np.nan
        marketcap_hint = stock_info.get('시가총액_기준일', np.nan)
        try:
            if pd.notna(marketcap_hint) and float(marketcap_hint) > 0:
                marketcap_usd_ref = float(marketcap_hint)
            # fallback: shares가 유효한 경우에만 기준일가*shares로 계산 (향후 확장용)
            elif pd.notna(shares) and shares > 0 and pd.notna(reference_date_price) and reference_date_price > 0:
                marketcap_usd_ref = float(reference_date_price) * float(shares)
        except Exception:
            marketcap_usd_ref = np.nan

        latest_data['시가총액'] = marketcap_usd_ref

        # 시총 회전율 계산
        try:
            시가총액 = marketcap_usd_ref
            if pd.notna(시가총액) and 시가총액 > 0:
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
        
        # 시가총액: NASDAQ Screener 기반 힌트(또는 위에서 계산된 marketcap_usd_ref)로 log 변환
        if pd.notna(marketcap_hint) and marketcap_hint > 0:
            latest_data['log_mktcap'] = np.log(marketcap_hint)
        elif pd.notna(marketcap_usd_ref) and marketcap_usd_ref > 0:
            latest_data['log_mktcap'] = np.log(marketcap_usd_ref)
        else:
            latest_data['log_mktcap'] = np.nan

        # 재무 피처는 NASDAQ 버전에서 기본 NaN (필요 시 추후 yfinance/SEC 등으로 확장)
        latest_data['이익수익률'] = np.nan

        # NASDAQ 전용 정리:
        # - 수익률(1M/3M) 피처는 과거에 제거된 피처이며 현재 파이프라인에서 사용하지 않음
        # - 불필요한 계산/컬럼 생성을 제거하여 성능 및 로그 노이즈를 줄임
        
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
        
        # ATRr_14는 기존 호환성을 위해 유지 (다른 곳에서 사용할 수 있음)
        df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        
        df_for_indicators.ta.obv(close='종가', volume='거래량', append=True)
        
        # OBV는 계산하지만 OBV_Slope 피처는 제거됨
        
        df_for_indicators.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        
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
        # - MA120/MA240의 상호 순서는 무관
        # - MA5가 MA120, MA240 둘 다 아래에 있고,
        # - MA5 기울기(각도)가 "하방 5도 이하"인 경우 제외
        #
        # 조건:
        #   (MA120 > MA5) & (MA240 > MA5) & (MA5_Angle_Deg <= 0)
        #
        # MA5_Angle_Deg 정의:
        #   ma5 = SMA(5)
        #   delta = (ma5_t - ma5_{t-1}) / ma5_{t-1}
        #   angle_deg = arctan(delta) * 180/pi
        #   (상승이면 +, 하락이면 -)
        # =================================================================
        try:
            ma5_series = df_for_indicators['종가'].rolling(window=5).mean()
            ma120_lvl = df_for_indicators['종가'].rolling(window=120).mean()
            ma240_lvl = df_for_indicators['종가'].rolling(window=240).mean()

            # MA5_Angle_Deg (정의 A: MA5 전일 대비 %변화 각도)
            if len(ma5_series) >= 2 and pd.notna(ma5_series.iloc[-1]) and pd.notna(ma5_series.iloc[-2]) and ma5_series.iloc[-2] != 0:
                delta = float((ma5_series.iloc[-1] - ma5_series.iloc[-2]) / ma5_series.iloc[-2])
                latest_data['MA5_Angle_Deg'] = float(np.degrees(np.arctan(delta)))
            else:
                latest_data['MA5_Angle_Deg'] = np.nan

            # Exclude_Rank (일자별/시점별 동적 평가)
            if len(ma120_lvl) and len(ma240_lvl) and len(ma5_series):
                ma5_last = ma5_series.iloc[-1]
                ma120_last = ma120_lvl.iloc[-1]
                ma240_last = ma240_lvl.iloc[-1]
                angle = latest_data.get('MA5_Angle_Deg', np.nan)
                latest_data['Exclude_Rank'] = bool(
                    pd.notna(ma240_last) and pd.notna(ma120_last) and pd.notna(ma5_last)
                    and (ma120_last > ma5_last)
                    and (ma240_last > ma5_last)
                    and (pd.notna(angle) and angle <= 0)
                )
            else:
                latest_data['Exclude_Rank'] = False
        except Exception:
            # 계산 실패 시에도 파이프라인은 계속 진행 (제외는 적용하지 않음)
            latest_data['MA5_Angle_Deg'] = latest_data.get('MA5_Angle_Deg', np.nan)
            latest_data['Exclude_Rank'] = False

        latest_data['52주_신고가_비율'] = (df_for_indicators['종가'] / df_for_indicators['종가'].rolling(250).max()).iloc[-1]
        
        # Relative_Strength_20, RVOL 피처는 제거됨
        
        # Z_Score_20 계산 (표준화 이격)
        mean_20 = df_for_indicators['종가'].rolling(20).mean().iloc[-1]
        std_20 = df_for_indicators['종가'].rolling(20).std().iloc[-1]
        if pd.notna(mean_20) and pd.notna(std_20) and std_20 != 0:
            latest_data['Z_Score_20'] = (df_for_indicators['종가'].iloc[-1] - mean_20) / std_20
        else:
            latest_data['Z_Score_20'] = np.nan
        
        # Position_Range_60 계산 (Donchian)
        high_60 = df_for_indicators['고가'].rolling(60).max().iloc[-1]
        low_60 = df_for_indicators['저가'].rolling(60).min().iloc[-1]
        if pd.notna(high_60) and pd.notna(low_60) and (high_60 - low_60) != 0:
            latest_data['Position_Range_60'] = (df_for_indicators['종가'].iloc[-1] - low_60) / (high_60 - low_60)
            latest_data['Position_Range_60'] = max(0, min(1, latest_data['Position_Range_60']))
        else:
            latest_data['Position_Range_60'] = 0.5
        
        # 변동성(1W), 변동성(3M) 피처 제거됨 (2024년 12월)
        
        # Eff_Ratio_10 계산 (효율성 비율) - 2024년 12월 제거
        # try:
        #     change = df_for_indicators['종가'].diff(10).abs()
        #     volatility = df_for_indicators['종가'].diff(1).abs().rolling(10).sum()
        #     if pd.notna(change.iloc[-1]) and pd.notna(volatility.iloc[-1]):
        #         latest_data['Eff_Ratio_10'] = change.iloc[-1] / (volatility.iloc[-1] + 1e-9)
        #     else:
        #         latest_data['Eff_Ratio_10'] = np.nan
        # except Exception as e:
        #     log_warning(f"Eff_Ratio_10 계산 실패 ({ticker}): {e}")
        #     latest_data['Eff_Ratio_10'] = np.nan
        
        # OBV는 OBV_Slope 계산에만 사용되므로 피처 리스트에서 제외
        # RSI_14는 RSI_Signal_Oscillator 계산에만 사용되므로 피처 리스트에서 제외
        technical_features_to_add = ['ATRr_14', 'ADX_14']
        for feature in technical_features_to_add:
            if feature in df_for_indicators.columns:
                 latest_data[feature] = df_for_indicators[feature].iloc[-1]

        if fs_data is not None and not fs_data.empty:
            latest_data.update(fs_data.iloc[0].to_dict())
        else:
            # 파이프라인 호환을 위해 기본 컬럼을 NaN으로 채움
            for col in ['PER', 'PBR', 'ROE', 'EPS', 'BPS', 'DIV', 'DPS']:
                latest_data[col] = np.nan
        
        # PBR_log 계산 (PBR 로그 변환) - 2024년 12월 제거
        # if 'PBR' in latest_data and pd.notna(latest_data['PBR']) and latest_data['PBR'] > 0:
        #     latest_data['PBR_log'] = np.log(latest_data['PBR'])
        # else:
        #     latest_data['PBR_log'] = np.nan
            
        # [정리] Log_Return_20도 현재 피처 세트에서 사용하지 않으므로 계산을 생략합니다.
            
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
            
        # MA120_Slope 계산 (회귀 윈도우 20)
        try:
            if len(df_for_indicators) >= 120:
                ma120_series = df_for_indicators['종가'].rolling(window=120).mean()
                latest_data['MA120_Slope'] = calculate_normalized_linear_regression_slope_latest(ma120_series, window=5)
            else:
                latest_data['MA120_Slope'] = np.nan
        except Exception as e:
            log_warning(f"MA120_Slope 계산 실패 ({ticker}): {e}")
            latest_data['MA120_Slope'] = np.nan
        
        # MA240_Slope 계산 (회귀 윈도우 20)
        try:
            if len(df_for_indicators) >= 240:
                ma240_series = df_for_indicators['종가'].rolling(window=240).mean()
                latest_data['MA240_Slope'] = calculate_normalized_linear_regression_slope_latest(ma240_series, window=5)
            else:
                latest_data['MA240_Slope'] = np.nan
        except Exception as e:
            log_warning(f"MA240_Slope 계산 실패 ({ticker}): {e}")
            latest_data['MA240_Slope'] = np.nan
        
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price
        latest_data['기준일가'] = reference_date_price
        latest_data['전날종가'] = prev_close  # 전날 종가(유효값) 추가
        if pd.notna(marketcap_hint) and marketcap_hint > 0:
            latest_data['시가총액'] = marketcap_hint
        elif pd.notna(shares) and shares > 0:
            latest_data['시가총액'] = reference_date_price * shares
        else:
            latest_data['시가총액'] = np.nan
        
        
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
    3. 거시경제 데이터 수집 (IXIC, VIX 등)
    4. 데이터 정제 및 병합
    
    Args:
        stock_list: 분석할 종목 목록
        selected_analysis_date: 분석 기준일
        use_cache: 캐시 사용 여부
        
    Returns:
        tuple: (처리된 데이터프레임, 실제 분석일)
    """
    # NASDAQ 전용: 날짜 경계(미국 시간) 이슈를 줄이기 위해 뉴욕 시간 기준으로 수집 구간 설정
    try:
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        ny_now = datetime.utcnow()
    today_date = ny_now.date()
    end_date_for_fetch = ny_now.strftime('%Y-%m-%d')
    start_date_for_fetch = (ny_now - timedelta(days=450)).strftime('%Y-%m-%d')

    latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date, use_cache)

    # NASDAQ 버전: 재무데이터 없이도 파이프라인이 진행되어야 함
    if latest_fs_df is None or latest_fs_df.empty or latest_fs_df.dropna(how='all').empty:
        log_warning("재무데이터가 없어도 분석을 계속 진행합니다. (PER/PBR 등은 NaN으로 처리)")
        latest_fs_df = stock_list[['종목코드']].copy()
        for col in ['PER', 'PBR', 'ROE', 'EPS', 'BPS', 'DIV', 'DPS']:
            latest_fs_df[col] = np.nan

    if selected_analysis_date.date() < today_date:
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
    is_today_analysis = actual_trading_date == today_date
    
    # 실시간 거시경제 데이터 수집 (캐시 사용 안함)
    log_info("🔄 실시간 거시경제 데이터를 수집합니다")
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        log_warning("거시 경제 데이터 수집에 실패했지만 분석을 계속합니다.")
        # 거시경제 데이터 없이도 분석 계속
    all_feature_data, all_actual_dates = [], []
    stock_records = stock_list.to_dict('records')
    
    # 배치 단위로 처리하여 메모리 효율성 향상
    batch_size = 50
    total_batches = (len(stock_records) + batch_size - 1) // batch_size
    total_stocks = len(stock_records)
    
    log_info(f"주식 분석 시작: {total_stocks:,}개 종목을 {total_batches}개 그룹으로 처리 (예상 5-10분)")
    
    # NOTE:
    # - 배치마다 ThreadPoolExecutor를 생성/종료하면(특히 Windows) 종료(shutdown/join)에서 간헐적으로 스톨하는 사례가 있음
    # - executor를 전체 루프 동안 재사용하고, 마지막에 한 번만 종료하여 스톨 가능성을 크게 낮춤
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # 워커 수 감소
    try:
        for i in range(0, len(stock_records), batch_size):
            batch = stock_records[i:i + batch_size]
            current_batch = i // batch_size + 1
            batch_start = i + 1
            batch_end = min(i + batch_size, total_stocks)
            
            # 첫 번째 그룹에서만 상세 설명 표시
            if current_batch == 1:
                log_info("가격, 거래량, 기술적 지표 계산 중...")
            
            # 10개 티커 배치 다운로드 (Yahoo) → 티커별 DataFrame 분리
            try:
                batch_tickers = [str(r.get('종목코드', '')).strip() for r in batch]
                batch_fetch_start = (pd.to_datetime(start_date_for_fetch) - timedelta(days=60)).strftime('%Y-%m-%d')
                batch_price_map = fetch_daily_ohlcv_batch(batch_tickers, batch_fetch_start, end_date_for_fetch)
            except Exception as e:
                log_warning(f"배치 일봉 다운로드 실패(단건 모드로 폴백): {e}")
                batch_price_map = {}

            future_to_stock = {
                executor.submit(
                    fetch_and_process_ticker_data,
                    r,
                    start_date_for_fetch,
                    end_date_for_fetch,
                    selected_analysis_date,
                    latest_fs_df,
                    batch_price_map.get(str(r.get('종목코드', '')).strip())
                ): r for r in batch
            }
            
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
                    
                    # 매번 진행률 업데이트 (같은 줄에서) - PROGRESS 접두사 유지
                    if completed_count % 10 == 0 or completed_count == total_count:
                        log_progress(
                            f"그룹 {current_batch}/{total_batches} 처리 중",
                            completed_count,
                            total_count,
                            context={'batch': current_batch, 'total_batches': total_batches}
                        )
                    
                except Exception as e: 
                    completed_count += 1
                    log_warning(
                        f"종목 처리 중 오류 발생: {e}",
                        context={
                            'batch': current_batch,
                            'stock': future_to_stock.get(future, {}).get('종목명', 'Unknown')
                        }
                    )
                    continue
            
            # 배치 완료 로그
            log_info(f"그룹 {current_batch}/{total_batches} 처리 완료 ({completed_count}/{total_count}개 종목)")

            # future dict 참조 정리(메모리 압박 완화)
            try:
                del future_to_stock
            except Exception:
                pass
        
            # 배치 간 메모리 정리
            # - Windows 환경에서 매 배치마다 full GC(gc.collect())는 장시간 스톨처럼 보일 수 있음
            # - 기본은 가벼운 0세대 GC만 수행하고, full GC는 간헐적으로만 수행
            try:
                gc_start = time.time()
                log_info("🧹 배치 후 메모리 정리 시작", context={
                    "batch": current_batch,
                    "total_batches": total_batches,
                    "success_count_so_far": len(all_feature_data)
                })

                # 가벼운 수집(0세대) - 대부분의 경우 이걸로 충분
                collected_gen0 = gc.collect(0)

                # full GC는 간헐적으로만 (기본: 5배치마다 1회)
                collected_full = None
                if current_batch % 5 == 0:
                    collected_full = gc.collect()

                elapsed_ms = int((time.time() - gc_start) * 1000)
                log_info("🧹 배치 후 메모리 정리 완료", context={
                    "batch": current_batch,
                    "collected_gen0": collected_gen0,
                    "collected_full": collected_full,
                    "elapsed_ms": elapsed_ms
                })
            except Exception as e:
                log_warning("배치 후 메모리 정리 중 오류(계속 진행)", context={
                    "batch": current_batch,
                    "error": str(e)
                })

            time.sleep(0.1)  # API 부하 방지
            
            success_count = len(all_feature_data)
            progress_percent = (current_batch / total_batches) * 100
            log_info(f"   ✅ 그룹 {current_batch}/{total_batches} 완료! ({progress_percent:.1f}% 진행, {success_count:,}개 종목 수집)")
    finally:
        # executor 종료 지점에서 스톨이 발생하는지 진단 로그 추가
        # NOTE:
        # - 일부 환경(특히 WSL/Windows)에서 executor.shutdown(wait=True)에서 간헐적으로 스톨 사례가 있음
        # - 종료를 블로킹하지 않고(wait=False) 종료 요청을 보낸 뒤, 스레드 종료 상태만 짧게 확인
        try:
            # 내부 스레드 상태(진단용) - private API이므로 방어적으로 처리
            thread_cnt = None
            alive_cnt = None
            try:
                threads = list(getattr(executor, "_threads", []))
                thread_cnt = len(threads)
                alive_cnt = sum(1 for t in threads if getattr(t, "is_alive", lambda: False)())
            except Exception:
                pass

            log_info("🧵 ThreadPoolExecutor 종료 요청", context={
                "thread_cnt": thread_cnt,
                "alive_cnt": alive_cnt
            })

            # 우선 종료 신호만 보내고 블로킹하지 않음
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python 버전 호환: cancel_futures 미지원
                executor.shutdown(wait=False)

            # 짧게만 상태 확인(무한 대기 금지)
            t0 = time.time()
            timeout_sec = 10.0
            while True:
                try:
                    threads = list(getattr(executor, "_threads", []))
                    alive_cnt = sum(1 for t in threads if getattr(t, "is_alive", lambda: False)())
                except Exception:
                    alive_cnt = None
                    break

                if alive_cnt == 0:
                    break
                if (time.time() - t0) >= timeout_sec:
                    break
                time.sleep(0.1)

            elapsed_ms = int((time.time() - t0) * 1000)
            if alive_cnt == 0:
                log_info("🧵 ThreadPoolExecutor 종료 완료", context={"elapsed_ms": elapsed_ms})
            else:
                log_warning("🧵 ThreadPoolExecutor 종료 지연(계속 진행)", context={
                    "alive_cnt": alive_cnt,
                    "elapsed_ms": elapsed_ms,
                    "timeout_sec": timeout_sec
                })
        except Exception as e:
            log_warning("ThreadPoolExecutor 종료 처리 중 오류(계속 진행)", context={"error": str(e)})
    
    log_info(f"🎉 종목 데이터 수집 완료! 총 {len(all_feature_data):,}개 종목")
    log_info("🔄 데이터 정제 및 거시경제 지표 병합 중...")
    
    if not all_feature_data: return pd.DataFrame(), None
    final_df = pd.DataFrame(all_feature_data)

    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    
    # 인덱스를 datetime으로 확실히 변환
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)
    
    log_info("   🔗 거시경제 지표(IXIC, VIX)를 종목 데이터와 병합 중...")
    
    # merge_asof 대신 더 안전한 방법 사용
    try:
        final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    except Exception as e:
        log_warning(f"   ⚠️ merge_asof 실패, 일반 merge로 시도: {e}")
        # 일반 merge로 대체
        macro_df_reset = macro_df.reset_index()
        macro_df_reset.rename(columns={'index': 'date'}, inplace=True)
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