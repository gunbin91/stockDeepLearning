"""
실시간 데이터 처리 시스템 (미국 주식: NYSE/NASDAQ/AMEX)
================================================

이 파일은 미국 종목을 대상으로 주식 분석 데이터를 수집/전처리합니다.
대용량 데이터를 효율적으로 처리하기 위해 병렬 처리와 메모리 최적화를 사용합니다.

주요 기능:
- 종목 목록 수집 (NYSE/NASDAQ/AMEX)
- (선택) 정적 시가총액(가능 시) 활용
- 거시경제 데이터 수집 (IXIC, VIX 등)
- 기술적 지표 계산
- 데이터 품질 검증 및 정제

주의:
- 본 프로젝트는 미국 주식 전용이며 국내(KRX) 기반 로직은 사용하지 않습니다.
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas_ta as ta
import concurrent.futures
import threading
from tqdm import tqdm
import os
import gc
import locale
import re
import platform
import multiprocessing
import time
import json
import urllib.request
import warnings

# 가장 강력한 경고 무시: 환경 변수로 Python 레벨에서 차단
os.environ['PYTHONWARNINGS'] = 'ignore'

# yfinance의 pandas deprecated API 경고 무시 (yfinance 라이브러리 자체 문제)
# Pandas4Warning을 직접 import하여 필터링
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except ImportError:
    # pandas 버전에 따라 Pandas4Warning이 없을 수 있음
    pass
# 추가로 FutureWarning도 필터링
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*", category=FutureWarning)
# 더 포괄적으로 모든 Pandas4Warning 무시
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)

# WSL2/Linux 환경에서 multiprocessing 최적화
# fork 방식 사용 (spawn보다 빠르고 메모리 효율적)
if platform.system() != 'Windows':
    try:
        # Linux/WSL2에서는 fork 방식이 기본값이지만 명시적으로 설정
        multiprocessing.set_start_method('fork', force=False)
    except RuntimeError:
        # 이미 설정된 경우 무시
        pass

# Windows 환경에서 로케일 설정 (FinanceDataReader 내부 오류 방지)
if platform.system() == 'Windows':
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 로케일 설정 실패 시 기본값 유지
        pass

from scoring import calculate_factor_scores
from path_manager import path_manager
from logger import log_info, log_critical, log_error, log_warning, log_progress

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
    # yfinance 경고 억제 (함수 전체를 감싸서)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from pandas.errors import Pandas4Warning
            warnings.simplefilter("ignore", Pandas4Warning)
        except (ImportError, AttributeError):
            pass
        
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
                    auto_adjust=True,
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
    # yfinance 경고 억제 (함수 전체를 감싸서)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from pandas.errors import Pandas4Warning
            warnings.simplefilter("ignore", Pandas4Warning)
        except (ImportError, AttributeError):
            pass
        
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
                    auto_adjust=True,
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
# 유틸리티 함수: 정규화된 선형회귀기울기 계산
# =================================================================

def calculate_normalized_linear_regression_slope(series, window=5):
    """
    정규화된 선형회귀기울기 계산 (NumPy 벡터화)
    
    Args:
        series: pandas Series (예: 변동성 값)
        window: 선형회귀에 사용할 기간
    
    Returns:
        정규화된 기울기 시계열 (백분율, %)
    
    계산식:
        기울기 = LINEARREG_SLOPE(series, window) / series * 100
    """
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    values = series.values
    n = len(values)
    slopes = np.full(n, np.nan, dtype=np.float64)
    
    # 시간 인덱스 (고정, 재사용)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_centered_sq_sum = np.sum(x_centered ** 2)
    
    # 슬라이딩 윈도우 계산
    for i in range(window - 1, n):
        # 현재 윈도우의 값
        y = values[i - window + 1:i + 1]
        current_value = values[i]
        
        # 안전성 검사: 0으로 나누기 및 NaN 방지
        if current_value == 0 or np.isnan(current_value):
            continue
        if np.isnan(y).any():
            continue
        
        # 선형회귀 기울기 계산
        y_mean = np.nanmean(y)
        y_centered = y - y_mean
        numerator = np.sum(x_centered * y_centered)
        
        if x_centered_sq_sum == 0:
            continue
        
        # 절대 기울기
        abs_slope = numerator / x_centered_sq_sum
        
        # 정규화: 현재 값으로 나누고 100 곱하기 (백분율)
        normalized_slope = (abs_slope / current_value) * 100
        slopes[i] = normalized_slope
    
    return pd.Series(slopes, index=series.index)

# 통일된 경로 사용
PROJECT_ROOT = str(path_manager.project_root)
DATA_DIR = str(path_manager.data_dir)

# =================================================================
# 전역 변수: 피처 계산 시 공유 데이터 (fork 방식에서 Copy-on-Write로 효율적)
# =================================================================
# WSL2/Linux에서 fork 방식을 사용할 때, 이 전역 변수들은 각 프로세스에서
# Copy-on-Write로 공유되어 메모리 효율적이고 빠르게 접근 가능합니다.
_global_marcap_data = None
_global_financial_data = None

def set_global_feature_data(marcap_data, financial_data):
    """
    전역 피처 데이터 설정 함수
    
    ProcessPoolExecutor 사용 전에 메인 프로세스에서 호출하여
    전역 변수에 데이터를 설정합니다. fork 방식에서는 이 데이터가
    각 프로세스에 Copy-on-Write로 공유됩니다.
    
    Args:
        marcap_data: 시가총액 데이터 (pandas.DataFrame)
        financial_data: 재무 데이터 (pandas.DataFrame)
    """
    global _global_marcap_data, _global_financial_data
    _global_marcap_data = marcap_data
    _global_financial_data = financial_data

def clear_global_feature_data():
    """
    전역 피처 데이터 초기화 함수
    
    사용 후 메모리 해제를 위해 호출합니다.
    """
    global _global_marcap_data, _global_financial_data
    _global_marcap_data = None
    _global_financial_data = None
    gc.collect()

def fetch_stock_list():
    """
    주식 목록 수집 함수
    
    미국 상장 종목 목록을 수집합니다. (NYSE/NASDAQ/AMEX)
    
    Returns:
        pandas.DataFrame: 종목코드, 종목명, 시장구분이 포함된 데이터프레임
    """
    try:
        exchanges = ["NASDAQ", "NYSE"]
        stock_lists = []

        for exchange in exchanges:
            log_info(f"FinanceDataReader를 통해 {exchange} 종목 리스트 수집...")
            df_list = fdr.StockListing(exchange)
            if df_list is None or df_list.empty:
                log_warning(f"{exchange} 종목 리스트를 가져올 수 없습니다. (계속 진행)")
                continue

            symbol_col = 'Symbol' if 'Symbol' in df_list.columns else ('Code' if 'Code' in df_list.columns else None)
            name_col = 'Name' if 'Name' in df_list.columns else ('Security Name' if 'Security Name' in df_list.columns else None)
            if symbol_col is None:
                log_warning(f"{exchange} 리스트에서 심볼 컬럼을 찾을 수 없습니다. columns={list(df_list.columns)}")
                continue
            if name_col is None:
                name_col = symbol_col

            stock_list = df_list[[symbol_col, name_col]].copy()
            stock_list.rename(columns={symbol_col: '종목코드', name_col: '종목명'}, inplace=True)
            stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.strip()
            stock_list['종목명'] = stock_list['종목명'].astype(str).str.strip()
            stock_list['시장구분'] = exchange

            # 정적 시가총액(가능 시)
            if 'MarketCap' in df_list.columns:
                stock_list['시가총액'] = pd.to_numeric(df_list['MarketCap'], errors='coerce')
            elif 'Marcap' in df_list.columns:
                stock_list['시가총액'] = pd.to_numeric(df_list['Marcap'], errors='coerce')
            else:
                stock_list['시가총액'] = np.nan

            stock_lists.append(stock_list)

        if not stock_lists:
            log_error("미국(NYSE/NASDAQ) 종목 리스트를 가져올 수 없습니다.")
            return pd.DataFrame()

        stock_list = pd.concat(stock_lists, ignore_index=True)

        # 중복 티커 제거 (시장 중복 상장 대비)
        before_dedup = len(stock_list)
        stock_list = stock_list.drop_duplicates(subset=['종목코드'], keep='first')
        dup_count = before_dedup - len(stock_list)
        if dup_count > 0:
            log_warning(f"중복 티커 {dup_count}개 제거됨 (종목코드 기준)")

        # 시가총액 관련 필터링 제거 (이후 병렬로 get_shares_full 수집하여 처리)
        log_info(f"주식 목록 수집 완료: {len(stock_list)}개 미국(NYSE/NASDAQ) 종목")
        return stock_list

    except Exception as e:
        log_error(f"주식 목록 수집 실패: {e}")
        return pd.DataFrame()

def _fetch_financial_data(start_date, end_date):
    """월초 재무데이터 수집 및 일별 분배 (NASDAQ 버전: 재무데이터 수집 생략)"""
    try:
        log_info("📊 NASDAQ 버전: 재무데이터 수집을 생략합니다.")
        return pd.DataFrame()
        
    except Exception as e:
        log_error(f"월초 재무데이터 수집 실패: {e}")
        return pd.DataFrame()

def _fetch_monthly_financial_data(date):
    """
    NASDAQ 전용 프로젝트에서는 월초 재무데이터(국내 기반)를 사용하지 않습니다.
    하위 호환을 위해 빈 DataFrame을 반환합니다.
    """
    try:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def distribute_monthly_financial_data_to_daily(monthly_data, start_date, end_date):
    """월초 재무데이터를 일별로 분배"""
    try:
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        all_dates = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
        trading_dates = all_dates[all_dates.weekday < 5]
        
        distributed_data = []
        
        for date in trading_dates:
            month_start = date.replace(day=1)
            month_dates = pd.date_range(start=month_start, end=month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1), freq='D')
            month_trading_dates = month_dates[month_dates.weekday < 5]
            
            if not month_trading_dates.empty:
                month_first_trading_day = month_trading_dates[0]
                
                monthly_data_for_date = monthly_data[monthly_data['date'] == month_first_trading_day].copy()
                if not monthly_data_for_date.empty:
                    monthly_data_for_date['date'] = date
                    distributed_data.append(monthly_data_for_date)
        
        if not distributed_data:
            log_warning("분배할 월초 재무데이터가 없습니다.")
            return monthly_data
        
        result_df = pd.concat(distributed_data, ignore_index=True)
        result_df.sort_values(by=['Code', 'date'], inplace=True)
        
        log_info(f"월초 재무데이터 일별 분배 완료: {len(result_df)}개 레코드")
        return result_df
        
    except Exception as e:
        log_error(f"월초 재무데이터 일별 분배 실패: {e}")
        return monthly_data

def _fetch_macro_data(start_date, end_date):
    """거시경제 데이터 수집 - data_fetcher.py의 구현 사용"""
    try:
        log_info(f"거시경제 데이터 수집 중: {start_date} ~ {end_date}")
        
        start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        
        macro_data = {}
        
        def _log_df_meta(tag, df):
            try:
                cols = list(df.columns) if df is not None else []
                idx_name = getattr(df.index, 'name', None) if df is not None else None
                idx_type = type(df.index).__name__ if df is not None else None
                log_info(f"{tag} 메타 | shape={getattr(df, 'shape', None)} | cols={cols} | index=({idx_type}, name={idx_name})")
            except Exception:
                pass

        try:
            ixic = fetch_daily_ohlcv('IXIC', start_date_str, end_date_str)
            _log_df_meta("IXIC 원본", ixic)
            if not ixic.empty:
                ixic_copy = ixic.copy()
                ixic_copy.index = pd.to_datetime(ixic_copy.index, format='mixed', errors='coerce')
                _log_df_meta("IXIC 정규화", ixic_copy)
                macro_data['IXIC'] = ixic_copy['Close']
        except Exception as e:
            log_warning(f"IXIC 데이터 수집 실패: {e}")
        
        try:
            vix = fetch_daily_ohlcv('^VIX', start_date_str, end_date_str)
            _log_df_meta("VIX 원본", vix)
            if not vix.empty:
                vix_copy = vix.copy()
                vix_copy.index = pd.to_datetime(vix_copy.index, format='mixed', errors='coerce')
                _log_df_meta("VIX 정규화", vix_copy)
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
                
                # KOSPI 변동성 1M 계산 (20일 기준) - 2024년 12월 제거
                # try:
                #     kospi_std_20 = kospi_close.rolling(window=20).std()
                #     kospi_mean_20 = kospi_close.rolling(window=20).mean()
                #     macro_df['KOSPI_변동성(1M)'] = kospi_std_20 / kospi_mean_20
                # except Exception as e:
                #     log_warning(f"KOSPI 변동성(1M) 계산 실패: {e}")
                #     macro_df['KOSPI_변동성(1M)'] = np.nan
                
                # IXIC_MA20_Slope 계산 (IXIC 20일 이동평균선 기울기)
                try:
                    ixic_ma20 = ixic_close.rolling(window=20).mean()
                    macro_df['IXIC_MA20_Slope'] = calculate_normalized_linear_regression_slope(ixic_ma20, window=5)
                except Exception as e:
                    log_warning(f"IXIC_MA20_Slope 계산 실패: {e}")
                    macro_df['IXIC_MA20_Slope'] = np.nan
            
            macro_df.reset_index(inplace=True)
            if 'date' not in macro_df.columns:
                if 'index' in macro_df.columns:
                    macro_df.rename(columns={'index': 'date'}, inplace=True)
                elif 'Date' in macro_df.columns:
                    macro_df.rename(columns={'Date': 'date'}, inplace=True)
                else:
                    datetime_cols = [c for c in macro_df.columns if np.issubdtype(macro_df[c].dtype, np.datetime64)]
                    if datetime_cols:
                        macro_df.rename(columns={datetime_cols[0]: 'date'}, inplace=True)
                    else:
                        macro_df['date'] = pd.to_datetime(macro_df.index, errors='coerce')
            _log_df_meta("거시경제 최종", macro_df)
            log_info("✅ 거시경제 데이터 수집 완료.")
            return macro_df
        else:
            log_warning("거시경제 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        log_error(f"거시경제 데이터 수집 실패: {e}")
        return pd.DataFrame()

def _prepare_price_df(ticker: str, df_price: pd.DataFrame):
    if df_price is None or df_price.empty or len(df_price) < 251 + 60:
        return (ticker, None)
    if 'Close' not in df_price.columns:
        return (ticker, None)
    # 실데이터가 거의 없으면 실패로 처리 (상장폐지/무효 티커 방지)
    valid_close = df_price['Close'].notna().sum()
    if valid_close < 251 + 60:
        return (ticker, None)
    df_price = df_price.copy()
    df_price.rename(columns={'Open':'시가', 'Close':'종가', 'High': '고가', 'Low': '저가', 'Volume':'거래량'}, inplace=True)
    df = df_price[['시가', '종가', '고가', '저가', '거래량']].copy()
    df.sort_index(inplace=True)
    return (ticker, df)

def fetch_ticker_price_data(stock_info, start_date, end_date, df_price=None):
    """
    단일 종목 주가 데이터 다운로드 함수 (I/O 작업)
    
    ThreadPoolExecutor로 병렬 처리되는 I/O 작업만 수행합니다.
    외부 future.result(timeout=120)로 타임아웃이 관리됩니다.
    
    Args:
        stock_info: 종목 정보 (종목코드, 종목명 등)
        start_date: 데이터 수집 시작일
        end_date: 데이터 수집 종료일
        
    Returns:
        tuple: (ticker, df_price) 또는 (ticker, None) - 실패 시
    """
    ticker = stock_info['종목코드']
    try:
        # NASDAQ 일봉 데이터 수집 (티커 그대로)
        if df_price is None:
            try:
                df_price = fetch_daily_ohlcv(ticker, start_date, end_date)
            except:
                df_price = None

        ticker, df = _prepare_price_df(ticker, df_price)
        if df is None:
            return (ticker, None)
        
        # 메모리 최적화: 원본 데이터프레임 해제
        del df_price
        gc.collect()
        
        return (ticker, df)
        
    except Exception as e:
        log_error(f"종목 {ticker} 데이터 다운로드 중 오류: {e}")
        return (ticker, None)



def calculate_ticker_features(ticker, df_price, stock_name=None, df_marcap_ticker=None):
    """
    단일 종목 피처 계산 함수 (CPU 작업)
    
    ProcessPoolExecutor로 병렬 처리되는 CPU 작업만 수행합니다.
    전역 변수(_global_financial_data)를 사용하여
    fork 방식에서 Copy-on-Write로 효율적으로 데이터에 접근합니다.
    
    Args:
        ticker: 종목코드
        df_price: 다운로드된 주가 데이터 (시가, 종가, 고가, 저가, 거래량)
        df_marcap_ticker: 해당 종목의 주식수(Shares) 이력 데이터
        
    Returns:
        pandas.DataFrame: 처리된 종목 데이터
    """
    # yfinance 경고 억제 (함수 전체를 감싸서)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from pandas.errors import Pandas4Warning
            warnings.simplefilter("ignore", Pandas4Warning)
        except (ImportError, AttributeError):
            pass
        
        global _global_financial_data
        
        try:
            df = df_price.copy()
            
            # 과거 주식 수(Shares) 데이터 병합 및 시가총액 계산
            if df_marcap_ticker is None or df_marcap_ticker.empty or 'Shares' not in df_marcap_ticker.columns:
                df['시가총액'] = np.nan
            else:
                df_marcap_ticker = df_marcap_ticker.copy()
                df_marcap_ticker.sort_values(by='date', inplace=True)
                df_marcap_ticker['date'] = pd.to_datetime(df_marcap_ticker['date']).astype('datetime64[ns]')
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index).astype('datetime64[ns]')
                try:
                    df = pd.merge_asof(left=df, right=df_marcap_ticker[['date', 'Shares']], left_index=True, right_on='date', direction='backward')
                except Exception as e:
                    log_warning(f"⚠️ {ticker} Shares merge_asof 실패, 일반 merge로 시도: {e}")
                    df = df.reset_index()
                    df = pd.merge(df, df_marcap_ticker[['date', 'Shares']], on='date', how='left')
                    df = df.set_index('date')
                
                # 빈 날짜는 과거 주식 수로 채우기
                df['Shares'] = df['Shares'].ffill()
                
                # 시가총액 = 종가 * 주식 수
                if '종가' in df.columns:
                    df['시가총액'] = df['종가'] * df['Shares']
                else:
                    df['시가총액'] = np.nan

            try:
                del df_marcap_ticker
            except:
                pass
            gc.collect()
            
            # 재무데이터 병합 (백업 프로젝트와 동일한 방식, 전역 변수 사용)
            if _global_financial_data is not None and not _global_financial_data.empty:
                df_financial_ticker = _global_financial_data[_global_financial_data['Code'] == ticker].copy()
                if not df_financial_ticker.empty:
                    df_financial_ticker.sort_values(by='date', inplace=True)
                    # 날짜 dtype 통일 (나노초로 통일하여 merge_asof 호환성 보장)
                    df_financial_ticker['date'] = pd.to_datetime(df_financial_ticker['date']).astype('datetime64[ns]')
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index).astype('datetime64[ns]')
                    try:
                        df = pd.merge_asof(left=df, right=df_financial_ticker[['date', 'PER', 'PBR', 'ROE', 'EPS', 'BPS']], 
                                           left_index=True, right_on='date', direction='backward')
                    except Exception as e:
                        # merge_asof 실패 시 일반 merge로 대체
                        log_warning(f"⚠️ {ticker} 재무데이터 merge_asof 실패, 일반 merge로 시도: {e}")
                        df = df.reset_index()
                        df = pd.merge(df, df_financial_ticker[['date', 'PER', 'PBR', 'ROE', 'EPS', 'BPS']], on='date', how='left')
                        df = df.set_index('date')
                    
                    # 재무데이터가 없는 경우 기본값 설정
                    if 'PER' not in df.columns or df['PER'].isnull().all():
                        df['PER'] = np.nan
                        df['PBR'] = np.nan
                        df['ROE'] = np.nan
                        df['EPS'] = np.nan
                        df['BPS'] = np.nan
                    
                    # 재무데이터 메모리 해제
                    del df_financial_ticker
                    gc.collect()
                else:
                    # 재무데이터가 없는 경우 기본값 설정
                    df['PER'] = np.nan
                    df['PBR'] = np.nan
                    df['ROE'] = np.nan
                    df['EPS'] = np.nan
                    df['BPS'] = np.nan
            else:
                # 재무데이터가 없는 경우 기본값 설정
                df['PER'] = np.nan
                df['PBR'] = np.nan
                df['ROE'] = np.nan
                df['EPS'] = np.nan
                df['BPS'] = np.nan
            
            # ATR 계산 (5일, 20일, 60일)
            try:
                atr_5 = df.ta.atr(high='고가', low='저가', close='종가', length=5)
                atr_20 = df.ta.atr(high='고가', low='저가', close='종가', length=20)
                atr_60 = df.ta.atr(high='고가', low='저가', close='종가', length=60)
                
                # ATRr_5 계산 (기준 - 1W): "최근 1주일 변동성 수준"
                if atr_5 is not None:
                    df['ATRr_5'] = (atr_5 / df['종가']) * 100
                else:
                    df['ATRr_5'] = np.nan
                
                # ATRr_20 계산 (기준 - 1M)
                if atr_20 is not None:
                    df['ATRr_20'] = (atr_20 / df['종가']) * 100
                else:
                    df['ATRr_20'] = np.nan
                
                # ATRr_60 계산 (기준 - 3M)
                if atr_60 is not None:
                    df['ATRr_60'] = (atr_60 / df['종가']) * 100
                else:
                    df['ATRr_60'] = np.nan
            except Exception as e:
                log_warning(f"ATR 계산 실패 ({ticker}): {e}")
                df['ATRr_5'] = np.nan
                df['ATRr_20'] = np.nan
                df['ATRr_60'] = np.nan
        
            # ATRr_14는 기존 호환성을 위해 유지 (다른 곳에서 사용할 수 있음)
            try:
                df.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
            except Exception as e:
                log_warning(f"ATRr_14 계산 실패 ({ticker}): {e}")
                df['ATRr_14'] = np.nan
            
            try:
                df.ta.obv(close='종가', volume='거래량', append=True)
            except Exception as e:
                log_warning(f"OBV 계산 실패 ({ticker}): {e}")
                df['OBV'] = np.nan
            
            # OBV는 계산하지만 OBV_Slope 피처는 제거됨
            
            try:
                df.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
            except Exception as e:
                log_warning(f"ADX 계산 실패 ({ticker}): {e}")
                df['ADX_14'] = np.nan
            
            # RSI_14 계산
            try:
                rsi_14 = df.ta.rsi(close='종가', length=14)
                
                # RSI_Signal_Oscillator 계산: RSI_14 - RSI_14.rolling(9).mean()
                # MACD 원리를 RSI에 적용한 것으로, 양수면 RSI가 평균을 뚫고 올라가는 중(골든크로스)
                if rsi_14 is not None and len(rsi_14) >= 9:
                    rsi_14_ma9 = rsi_14.rolling(window=9).mean()
                    df['RSI_Signal_Oscillator'] = rsi_14 - rsi_14_ma9
                else:
                    df['RSI_Signal_Oscillator'] = np.nan
            except Exception as e:
                log_warning(f"RSI 계산 실패 ({ticker}): {e}")
                df['RSI_14'] = np.nan
                df['RSI_Signal_Oscillator'] = np.nan
            
            # 기존 방식과 동일한 기본 지표들만 사용 (과도한 기술적 지표 제거)
            
            # 거래대금 계산
            df['거래대금'] = df['종가'] * df['거래량']
            
            # RVOL (상대 거래량) 계산
            try:
                거래량_20일_평균 = df['거래량'].rolling(window=20).mean()
                df['RVOL'] = df['거래량'] / 거래량_20일_평균
                # 무한대 값 처리
                df['RVOL'] = df['RVOL'].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                log_warning(f"RVOL 계산 실패 ({ticker}): {e}")
                df['RVOL'] = np.nan
            
            # 시총 회전율 계산
            try:
                # 시총 회전율(1W): 5일 평균 거래대금 / 시가총액 * 100
                거래대금_5일_평균 = df['거래대금'].rolling(window=5).mean()
                df['시총 회전율(1W)'] = (거래대금_5일_평균 / df['시가총액']) * 100
                df['시총 회전율(1W)'] = df['시총 회전율(1W)'].replace([np.inf, -np.inf], np.nan)
                
                # 시총 회전율(3M): 60일 평균 거래대금 / 시가총액 * 100
                거래대금_60일_평균 = df['거래대금'].rolling(window=60).mean()
                df['시총 회전율(3M)'] = (거래대금_60일_평균 / df['시가총액']) * 100
                df['시총 회전율(3M)'] = df['시총 회전율(3M)'].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                log_warning(f"시총 회전율 계산 실패 ({ticker}): {e}")
                df['시총 회전율(1W)'] = np.nan
                df['시총 회전율(3M)'] = np.nan
            
            # Z_Score_20 계산 (내부용) 및 Trend_Pullback_Score 생성
            try:
                mean_20 = df['종가'].rolling(20).mean()
                std_20 = df['종가'].rolling(20).std()
                
                # std_20이 0인 경우 처리 (변동성이 없으면 z_score를 0으로 설정)
                z_score_20 = (df['종가'] - mean_20) / std_20.replace(0, np.nan)
                z_score_20 = z_score_20.fillna(0)  # std가 0인 경우 z_score를 0으로 설정
                
                # MA20_Slope 내부 계산용
                ma20 = df['종가'].rolling(window=20).mean()
                ma20_slope = calculate_normalized_linear_regression_slope(ma20, window=5)
                
                # Trend_Pullback_Score 계산
                # 의미: 추세 강도와 눌림 정도를 결합한 점수
                # - 상승 추세(ma20_slope > 0) + 눌림(z_score_20 < 0): 높은 양수 점수
                # - 상승 추세 + 과열(z_score_20 > 0): 낮은 양수 또는 0
                # - 하락 추세(ma20_slope < 0): 음수 또는 0
                # 기본 공식: abs(z_score_20) * ma20_slope (단, 조건에 따라 가중치 조정)
                
                # NaN 값 처리
                ma20_slope_clean = ma20_slope.fillna(0)
                z_score_20_clean = z_score_20.fillna(0)
                
                # 기본 점수 계산: abs(z_score) * ma20_slope
                base_score = np.abs(z_score_20_clean) * ma20_slope_clean
                
                # 조건별 가중치 적용
                # 1. 상승 추세 + 눌림: 가중치 1.0 (가장 높은 점수)
                # 2. 상승 추세 + 과열: 가중치 0.3 (낮은 점수)
                # 3. 하락 추세: 가중치 0.1 또는 음수 (매우 낮은 점수)
                
                condition_up_pullback = (ma20_slope_clean > 0) & (z_score_20_clean < 0)  # 상승 추세 + 눌림
                condition_up_overheat = (ma20_slope_clean > 0) & (z_score_20_clean >= 0)  # 상승 추세 + 과열
                condition_down = (ma20_slope_clean <= 0)  # 하락 추세
                
                df['Trend_Pullback_Score'] = np.where(
                    condition_up_pullback,
                    base_score * 1.0,  # 상승 추세 + 눌림: 최고 점수
                    np.where(
                        condition_up_overheat,
                        base_score * 0.3,  # 상승 추세 + 과열: 낮은 점수
                        np.where(
                            condition_down,
                            base_score * 0.1,  # 하락 추세: 매우 낮은 점수
                            0
                        )
                    )
                )
                
                # 원본 데이터에 NaN이 있던 위치는 NaN으로 복원
                nan_mask = ma20_slope.isna() | z_score_20.isna()
                df.loc[nan_mask, 'Trend_Pullback_Score'] = np.nan
                    
            except Exception as e:
                log_warning(f"Trend_Pullback_Score 계산 실패 ({ticker}): {e}")
                df['Trend_Pullback_Score'] = np.nan
        
            # Position_Range_60 계산 (Donchian)
            try:
                high_60 = df['고가'].rolling(60).max()
                low_60 = df['저가'].rolling(60).min()
                range_60 = high_60 - low_60
                df['Position_Range_60'] = np.where(range_60 != 0, (df['종가'] - low_60) / range_60, 0.5)
                df['Position_Range_60'] = df['Position_Range_60'].clip(0, 1)
            except Exception as e:
                log_warning(f"Position_Range_60 계산 실패 ({ticker}): {e}")
                df['Position_Range_60'] = np.nan

            # Max_Drawdown_20 계산 (최근 20일 최대 낙폭, %)
            # roll_max = 고가.rolling(20).max()
            # daily_dd = (저가 / roll_max) - 1
            # Max_Drawdown_20 = daily_dd.rolling(20).min() * 100
            try:
                roll_max_20 = df['고가'].rolling(window=20).max()
                daily_dd_20 = (df['저가'] / roll_max_20) - 1
                df['Max_Drawdown_20'] = daily_dd_20.rolling(window=20).min() * 100
            except Exception as e:
                log_warning(f"Max_Drawdown_20 계산 실패 ({ticker}): {e}")
                df['Max_Drawdown_20'] = np.nan
            
            # 변동성(1W), 변동성(3M) 피처 제거됨 (2024년 12월)
            
            # Eff_Ratio_10 계산 (효율성 비율) - 2024년 12월 제거
            # try:
            #     change = df['종가'].diff(10).abs()
            #     volatility = df['종가'].diff(1).abs().rolling(10).sum()
            #     df['Eff_Ratio_10'] = change / (volatility + 1e-9)
            #     # 무한대 값 처리
            #     df['Eff_Ratio_10'] = df['Eff_Ratio_10'].replace([np.inf, -np.inf], np.nan)
            # except Exception as e:
            #     log_warning(f"Eff_Ratio_10 계산 실패 ({ticker}): {e}")
            #     df['Eff_Ratio_10'] = np.nan
            
            # 재무데이터 관련 지표 (백업 프로젝트와 동일)
            if 'PER' in df.columns and not df['PER'].isnull().all():
                df['이익수익률'] = 1 / df['PER']  # 이익수익률 = 1/PER
            else:
                df['이익수익률'] = np.nan
            
            # 시가총액 관련 지표 (재무데이터가 없는 경우에만 간단한 계산)
            if 'PER' not in df.columns or df['PER'].isnull().all():
                df['PER'] = df['종가'] / (df['거래대금'] / df['거래량'])  # 간단한 PER 계산
            if 'PBR' not in df.columns or df['PBR'].isnull().all():
                df['PBR'] = df['종가'] / (df['시가총액'] / df['거래량'])  # 간단한 PBR 계산
        
            # 핵심 피처 추가
            # 1. log_mktcap (시가총액 로그 변환)
            # 시가총액이 0보다 큰 경우에만 로그 적용 (경고 방지)
            df['log_mktcap'] = np.nan  # float 타입으로 초기화
            mask = df['시가총액'] > 0
            df.loc[mask, 'log_mktcap'] = np.log(df.loc[mask, '시가총액'])
            
            # 2. PBR_log (PBR 로그 변환) - 2024년 12월 제거
            # PBR이 0보다 큰 경우에만 로그 적용 (경고 방지)
            # df['PBR_log'] = np.nan  # float 타입으로 초기화
            # if 'PBR' in df.columns:
            #     pbr_mask = df['PBR'] > 0
            #     df.loc[pbr_mask, 'PBR_log'] = np.log(df.loc[pbr_mask, 'PBR'])
            # else:
            #     df['PBR_log'] = np.nan
            
            # 2-2. [신규 추가] HV변동성(1M) (Historical Volatility 1M)
            # 일별 로그 수익률의 20일 이동 표준편차
            try:
                log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
                df['HV_Volatility_20'] = log_ret_1d.rolling(window=20).std()
            except Exception as e:
                log_warning(f"HV_Volatility_20 계산 실패 ({ticker}): {e}")
                df['HV_Volatility_20'] = np.nan
            
            # 2-2-0. [신규 추가] HV변동성(1W) (Historical Volatility 1W)
            try:
                if 'log_ret_1d' not in locals():
                    log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
                df['HV_Volatility_5'] = log_ret_1d.rolling(window=5).std()
            except Exception as e:
                log_warning(f"HV_Volatility_5 계산 실패 ({ticker}): {e}")
                df['HV_Volatility_5'] = np.nan
            
            # 2-4. [신규 추가] HV변동성(3M) (Historical Volatility 3M)
            # 일별 로그 수익률의 60일 이동 표준편차
            try:
                if 'log_ret_1d' not in locals():
                    log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
                df['HV_Volatility_60'] = log_ret_1d.rolling(window=60).std()
            except Exception as e:
                log_warning(f"HV_Volatility_60 계산 실패 ({ticker}): {e}")
                df['HV_Volatility_60'] = np.nan
            
            # 2-3. [신규 추가] VWAP Disparity(1W) (VWAP 괴리율 1주)
            # 최근 5일 거래대금 가중 평균 가격 대비 현재가 비율
            try:
                tp = (df['고가'] + df['저가'] + df['종가']) / 3
                money = tp * df['거래량']
                
                sum_money_5 = money.rolling(window=5).sum()
                sum_vol_5 = df['거래량'].rolling(window=5).sum()
                
                vwap_5 = sum_money_5 / (sum_vol_5 + 1e-9)
                df['VWAP_Disparity_5'] = (df['종가'] / vwap_5 - 1) * 100
            except Exception as e:
                log_warning(f"VWAP_Disparity_5 계산 실패 ({ticker}): {e}")
                df['VWAP_Disparity_5'] = np.nan
            
            # 3. 이격도 계산 (120일, 240일) - disparity_20 제거
            for p in [120, 240]:
                ma = df['종가'].rolling(window=p).mean()
                df[f'disparity_{p}'] = (df['종가'] / ma) * 100
            # disparity_20 추가
            try:
                ma20 = df['종가'].rolling(window=20).mean()
                df['disparity_20'] = (df['종가'] / ma20) * 100
            except Exception as e:
                log_warning(f"disparity_20 계산 실패 ({ticker}): {e}")
                df['disparity_20'] = np.nan
            
            # MA120_Slope 계산 (120일 이동평균선 기울기)
            try:
                ma120 = df['종가'].rolling(window=120).mean()
                df['MA120_Slope'] = calculate_normalized_linear_regression_slope(ma120, window=5)
            except Exception as e:
                log_warning(f"MA120_Slope 계산 실패 ({ticker}): {e}")
                df['MA120_Slope'] = np.nan

            # MA20_Slope 계산 (20일 이동평균선 기울기) - MA120/MA240과 동일한 방식
            try:
                ma20 = df['종가'].rolling(window=20).mean()
                df['MA20_Slope'] = calculate_normalized_linear_regression_slope(ma20, window=5)
            except Exception as e:
                log_warning(f"MA20_Slope 계산 실패 ({ticker}): {e}")
                df['MA20_Slope'] = np.nan
            
            # MA240_Slope 계산 (240일 이동평균선 기울기)
            try:
                ma240 = df['종가'].rolling(window=240).mean()
                df['MA240_Slope'] = calculate_normalized_linear_regression_slope(ma240, window=5)
            except Exception as e:
                log_warning(f"MA240_Slope 계산 실패 ({ticker}): {e}")
                df['MA240_Slope'] = np.nan

            # =================================================================
            # 랭킹 제외 규칙 (KrStockTmp 기준 반영)
            # - MA20이 MA120, MA240 아래에 있고 종가가 MA60 아래에 있으면 제외
            # - 14거래일 연속 MA20 하락 중이며 종가가 MA20 아래에 있으면 제외
            # =================================================================
            try:
                ma5 = df['종가'].rolling(window=5).mean()
                ma20_lvl = df['종가'].rolling(window=20).mean()
                ma60_lvl = df['종가'].rolling(window=60).mean()
                ma120_lvl = df['종가'].rolling(window=120).mean()
                ma240_lvl = df['종가'].rolling(window=240).mean()
                
                # 단기 모멘텀 (MA5 기울기)
                ma5_prev = ma5.shift(1)
                delta = (ma5 - ma5_prev) / ma5_prev.replace(0, np.nan)
                df['MA5_Angle_Deg'] = np.degrees(np.arctan(delta.astype(float)))

                # 14거래일 연속 MA20 하락 확인
                ma20_diff = ma20_lvl.diff()
                ma20_down = (ma20_diff < 0).astype(int)
                ma20_down_14_days = ma20_down.rolling(window=14).sum() == 14

                cond1 = (ma20_lvl < ma120_lvl) & (ma20_lvl < ma240_lvl) & (df['종가'] < ma60_lvl)
                cond2 = ma20_down_14_days & (df['종가'] < ma20_lvl)

                df['Exclude_Rank'] = cond1 | cond2
            except Exception:
                # 계산 실패 시에도 기존 파이프라인은 유지
                df['MA5_Angle_Deg'] = np.nan
                df['Exclude_Rank'] = False
        
            # 3. 52주 신고가 비율
            df['52주_최고가'] = df['종가'].rolling(250).max()
            df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
            
            # target 변수 생성:
            # - 향후 14거래일 동안 최저가가 현재가 대비 -10% 이하로 내려가지 않고 (>= 0.90)
            # - 향후 14거래일 동안 최고가가 현재가 대비 +8% 이상 한 번이라도 상승하면 (>= 1.08)
            # => 1, 아니면 0
            min_price_14d = df['종가'].rolling(window=14, min_periods=1).min().shift(-14)
            max_price_14d = df['종가'].rolling(window=14, min_periods=1).max().shift(-14)
            # 조건: 최소값 >= 현재가격 * 0.90 AND 최대값 >= 현재가격 * 1.08
            df['target'] = ((min_price_14d / df['종가'] >= 0.90) & (max_price_14d / df['종가'] >= 1.08)).astype(int)
            
            # 중간 변수 삭제 (메모리 최적화)
            del min_price_14d, max_price_14d

            # =================================================================
            # 학습 타겟 제외 규칙 (요청사항)
            # - Exclude_Rank(True)인 샘플은 학습 데이터에서 제외(drop)하기 위해 target을 NaN 처리합니다.
            # - 학습 스크립트(train_gpu_main.py / train_lgbm_gpu_main.py)는
            #   full_df['target'].notna()로 먼저 필터링하므로 정상 동작합니다.
            # =================================================================
            try:
                if 'Exclude_Rank' in df.columns:
                    exclude_mask = df['Exclude_Rank'].fillna(False)
                    if exclude_mask.any():
                        df.loc[exclude_mask, 'target'] = np.nan
            except Exception:
                # 제외 규칙 적용 실패 시에도 기존 target은 유지
                pass

            df['종목코드'] = ticker
            # 종목명 추가 (있는 경우만)
            if stock_name is not None:
                df['종목명'] = stock_name
            
            # 데이터 구조 설정
            # merge_asof 후 date 컬럼이 제거되므로 다시 추가
            df['date'] = df.index
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            log_error(f"종목 {ticker} 피처 계산 중 오류: {e}")
            # 오류 발생 시에도 메모리 정리
            try:
                gc.collect()
            except:
                pass
            return None



def _fetch_and_prepare_data(start_date, end_date, skip_factor_scores=False):
    """실시간 데이터 수집 및 전처리 (파이프라인 최적화 적용)"""
    log_info(f"실시간 데이터 수집 시작 ({start_date} ~ {end_date})...")
    
    stock_list = fetch_stock_list()
    if stock_list.empty: 
        raise ValueError("종목 리스트를 가져올 수 없습니다.")
    
    import yfinance as yf
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # 429 차단 우회를 위한 스마트 리트라이 세션 구성
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    def _fetch_shares_for_ticker(ticker_sym):
        try:
            t = str(ticker_sym).replace(".", "-").strip()
            tk = yf.Ticker(t, session=session)
            start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            # sleep 없이 최고 속도로 수집 (세션에서 자동 재시도)
            shares = tk.get_shares_full(start=start_str, end=end_str)
            if shares is not None and not shares.empty:
                df_s = pd.DataFrame(shares, columns=['Shares'])
                df_s.index = pd.to_datetime(df_s.index).tz_localize(None)
                df_s = df_s.reset_index().rename(columns={'index': 'date'})
                df_s['Code'] = ticker_sym
                return df_s
        except Exception:
            pass
        return None
    
    # -------------------------------------------------------------------------
    # 재무데이터 수집 (기존 동일)
    # -------------------------------------------------------------------------
    try:
        log_info("📊 재무데이터 수집 시작...")
        df_financial_long = _fetch_financial_data(start_date, end_date)
        
        if df_financial_long.empty:
            log_warning("재무데이터 수집에 실패했지만 분석을 계속합니다.")
        else:
            log_info(f"✅ 재무데이터 수집 완료: {len(df_financial_long)}개 레코드")
            
    except Exception as e:
        log_warning(f"재무데이터 수집 실패: {e}. 분석을 계속합니다.")
        df_financial_long = pd.DataFrame()
    
    # 전역 변수에 재무 데이터 설정
    set_global_feature_data(None, df_financial_long)
    
    all_data = []
    stock_records = stock_list.to_dict('records')
    total_count = len(stock_records)
    
    log_info(f"🚀 개별 종목 파이프라인 수집/연산 시작: 총 {total_count}개 종목")
    
    download_failed = []
    failed_count = 0
    completed_count = 0
    
    batch_size = 50
    total_batches = (total_count + batch_size - 1) // batch_size
    
    cpu_workers = max(2, int((os.cpu_count() or 8) * 2 / 3))
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as cpu_executor:
            futures_to_ticker = {}
            
            for batch_idx in range(0, total_count, batch_size):
                batch = stock_records[batch_idx:batch_idx + batch_size]
                current_batch = batch_idx // batch_size + 1
                batch_tickers = [str(row.get('종목코드', '')).strip() for row in batch]
                
                # [1] 주가 데이터 다운로드 (Bulk)
                batch_price_map = fetch_daily_ohlcv_batch(batch_tickers, start_date, end_date)
                
                # [2] 주식수 데이터 다운로드 (ThreadPool 20개 극대화)
                batch_shares_map = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as io_executor:
                    io_futures = {io_executor.submit(_fetch_shares_for_ticker, t): t for t in batch_tickers}
                    for f in concurrent.futures.as_completed(io_futures):
                        t = io_futures[f]
                        try:
                            batch_shares_map[t] = f.result()
                        except:
                            batch_shares_map[t] = None
                
                # [3] CPU 풀에 즉시 투입
                batch_submitted = 0
                for row in batch:
                    ticker = str(row.get('종목코드', '')).strip()
                    df_price = batch_price_map.get(ticker)
                    
                    _, df_prepared = _prepare_price_df(ticker, df_price)
                    if df_prepared is not None:
                        stock_name = row.get('종목명')
                        df_marcap_ticker = batch_shares_map.get(ticker)
                        
                        fut = cpu_executor.submit(
                            calculate_ticker_features, 
                            ticker, df_prepared, stock_name, df_marcap_ticker
                        )
                        futures_to_ticker[fut] = ticker
                        batch_submitted += 1
                    else:
                        download_failed.append(ticker)
                
                log_info(f"배치 {current_batch}/{total_batches} 파이프라인 투입 완료 ({batch_submitted}/{len(batch)}개 종목)")
                gc.collect()
                
            # [4] 연산 결과 취합
            log_info("⚙️ 파이프라인 연산 완료 대기 중...")
            for future in concurrent.futures.as_completed(futures_to_ticker):
                ticker = futures_to_ticker[future]
                try:
                    result_df = future.result(timeout=300)
                    if result_df is not None and isinstance(result_df, pd.DataFrame) and not result_df.empty and '종목코드' in result_df.columns:
                        all_data.append(result_df)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    log_error(f"❌ 종목 {ticker} 연산 중 오류: {e}")
                    
                completed_count += 1
                log_progress("파이프라인 진행", completed_count, len(futures_to_ticker))
                if completed_count % 50 == 0:
                    gc.collect()

    finally:
        clear_global_feature_data()
        gc.collect()

    if not all_data: 
        log_error("처리된 데이터가 없습니다.")
        raise ValueError("처리된 데이터가 없습니다.")
    
    # 데이터 검증 로직 추가
    success_count = len(all_data)
    total_attempted = total_count  # 전체 시도한 종목 수
    download_fail_count = len(download_failed)  # 다운로드 실패 수
    calc_fail_count = failed_count  # 피처 계산 실패 수
    total_fail_count = download_fail_count + calc_fail_count
    
    success_rate = (success_count / total_attempted) * 100 if total_attempted > 0 else 0
    log_info(f"✅ 개별 종목 피처 데이터 생성 완료: {success_count}/{total_attempted}개 종목 처리됨 (성공률: {success_rate:.1f}%)")
    log_info(f"   - 다운로드 실패: {download_fail_count}개, 피처 계산 실패: {calc_fail_count}개")
    
    # 성공률이 너무 낮으면 경고
    if success_rate < 50:
        log_warning(f"⚠️ 종목 처리 성공률이 낮습니다: {success_rate:.1f}% (권장: 80% 이상)")
    elif success_rate < 80:
        log_warning(f"⚠️ 종목 처리 성공률이 다소 낮습니다: {success_rate:.1f}% (권장: 80% 이상)")
    
    # 동일한 방식으로 처리
    raw_feature_df = pd.concat(all_data)
    # 어떤 경우에도 date 컬럼을 보장
    raw_feature_df['date'] = raw_feature_df.index
    raw_feature_df = raw_feature_df.reset_index(drop=True)
    
    raw_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_feature_df.dropna(subset=['date', '종목코드'], inplace=True)
    raw_feature_df['date'] = pd.to_datetime(raw_feature_df['date'])
    raw_feature_df.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)
    
    # 개별 데이터 리스트 메모리 해제
    del all_data
    gc.collect()
    
    # 거시경제 데이터 추가
    macro_df = _fetch_macro_data(start_date, end_date)
    if not macro_df.empty and 'date' in macro_df.columns and 'date' in raw_feature_df.columns:
        raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')
    else:
        log_warning("거시경제 데이터 병합 스킵: date 컬럼이 없습니다.")
    
    # Relative_Strength_20 피처는 제거됨
    
    raw_feature_df.sort_values(by=['date', '종목코드'], inplace=True)

    # 팩터 점수 계산 (백테스팅/분석용, 학습용 데이터 생성 시에는 불필요)
    if skip_factor_scores:
        log_info("   ℹ️ 학습용 데이터 생성: 팩터 점수 계산을 건너뜁니다.")
        final_df = raw_feature_df.copy()
    else:
        log_info("일별 팩터 점수 계산 중...")
        final_df = raw_feature_df.groupby('date', group_keys=False).apply(calculate_factor_scores).reset_index(drop=True)
    
    # 최종 데이터 검증
    if final_df.empty:
        log_error("최종 데이터가 비어있습니다.")
        raise ValueError("최종 데이터가 비어있습니다.")
    
    # 데이터 품질 검증
    total_rows = len(final_df)
    valid_rows = len(final_df.dropna(subset=['종목코드', 'date']))
    data_quality = valid_rows / total_rows * 100 if total_rows > 0 else 0
    
    log_info(f"📊 최종 데이터 품질: {valid_rows:,}/{total_rows:,}개 유효 행 ({data_quality:.1f}%)")
    
    if data_quality < 70:
        log_warning(f"⚠️ 데이터 품질이 낮습니다: {data_quality:.1f}% (권장: 90% 이상)")
    elif data_quality < 90:
        log_warning(f"⚠️ 데이터 품질이 다소 낮습니다: {data_quality:.1f}% (권장: 90% 이상)")
    
    # 원본 데이터 메모리 해제
    del raw_feature_df
    gc.collect()
    
    return final_df

def get_preprocessed_data(start_date, end_date, skip_factor_scores=False):
    """실시간 데이터 전처리 함수
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        skip_factor_scores: True일 경우 팩터 점수 계산을 건너뜀 (학습용 데이터 생성 시 사용)
    """
    try:
        log_info("🔄 실시간 데이터 수집 시작", context={
            "start_date": start_date,
            "end_date": end_date,
            "mode": "realtime"
        })
        
        return _fetch_and_prepare_data(start_date, end_date, skip_factor_scores=skip_factor_scores)
        
    except Exception as e:
        log_critical("실시간 데이터 수집 중 오류", exception=e, context={
            "start_date": start_date,
            "end_date": end_date
        })
        return pd.DataFrame()
