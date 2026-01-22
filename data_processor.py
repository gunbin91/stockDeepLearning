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
from tqdm import tqdm
import os
import gc
import locale
import platform
import multiprocessing
import time
import json
import urllib.request

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
# 미국 시가총액 보강용 (Yahoo 차단 대비): Nasdaq Screener API
# - 한 번 호출로 다수 심볼의 marketCap(USD)을 확보할 수 있어 대량 티커에도 안정적
# - 표준 라이브러리(urllib)만 사용하여 환경 의존 최소화
# =================================================================
def _fetch_nasdaq_screener_marketcap_map() -> dict:
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
        exchanges = ["NASDAQ", "NYSE", "AMEX"]
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
            log_error("미국(NYSE/NASDAQ/AMEX) 종목 리스트를 가져올 수 없습니다.")
            return pd.DataFrame()

        stock_list = pd.concat(stock_lists, ignore_index=True)

        # 중복 티커 제거 (시장 중복 상장 대비)
        before_dedup = len(stock_list)
        stock_list = stock_list.drop_duplicates(subset=['종목코드'], keep='first')
        dup_count = before_dedup - len(stock_list)
        if dup_count > 0:
            log_warning(f"중복 티커 {dup_count}개 제거됨 (종목코드 기준)")

        # --------------------------------------------------------------
        # 학습/전처리 경로용 시가총액 보강 (Nasdaq Screener API)
        # - Yahoo(yfinance/quote)는 환경에 따라 401(crumb/unauthorized)로 막힐 수 있어 배제
        # - Screener API는 1회 호출로 다수 심볼 marketCap을 제공 → 대규모 티커에도 안정적
        # - 원본 훼손 최소: NaN(또는 0 이하)인 티커만 보강, 실패 시 NaN 유지
        # --------------------------------------------------------------
        try:
            missing_mask = stock_list['시가총액'].isna() | (pd.to_numeric(stock_list['시가총액'], errors='coerce') <= 0)
            missing_tickers = stock_list.loc[missing_mask, '종목코드'].astype(str).str.strip().tolist()
            if missing_tickers:
                log_info(f"📌 시가총액 보강 필요: {len(missing_tickers):,}개 티커 (screener marketCap 조회)")
                screener_map = _fetch_nasdaq_screener_marketcap_map()

                ok = 0
                # 티커 관례 차이 보정: '.' -> '-' (예: BRK.B -> BRK-B)
                for t in missing_tickers:
                    mc = screener_map.get(t)
                    if mc is None:
                        mc = screener_map.get(t.replace(".", "-"))
                    if mc is not None and np.isfinite(float(mc)) and float(mc) > 0:
                        stock_list.loc[stock_list['종목코드'] == t, '시가총액'] = float(mc)
                        ok += 1

                still_missing = int(stock_list['시가총액'].isna().sum())
                log_info(f"✅ 시가총액 보강 완료: 성공 {ok:,}개 | 남은 NaN {still_missing:,}개")
        except Exception as e:
            log_warning(f"시가총액 보강 중 오류(무시하고 진행): {e}")

        # 시가총액 조회 불가 종목 제외
        mktcap = pd.to_numeric(stock_list['시가총액'], errors='coerce')
        before_filter = len(stock_list)
        stock_list = stock_list[mktcap > 0].copy()
        removed = before_filter - len(stock_list)
        if removed > 0:
            log_info(f"시가총액 미확인 종목 {removed}개 제외")

        log_info(f"주식 목록 수집 완료: {len(stock_list)}개 미국(NYSE/NASDAQ/AMEX) 종목")
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
        
        try:
            ixic = fdr.DataReader('IXIC', start_date_str, end_date_str)
            if not ixic.empty:
                ixic_copy = ixic.copy()
                ixic_copy.index = pd.to_datetime(ixic_copy.index, format='mixed', errors='coerce')
                macro_data['IXIC'] = ixic_copy['Close']
        except Exception as e:
            log_warning(f"IXIC 데이터 수집 실패: {e}")
        
        try:
            vix = fdr.DataReader('^VIX', start_date_str, end_date_str)
            if not vix.empty:
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
            macro_df.rename(columns={'index': 'date'}, inplace=True)
            log_info("✅ 거시경제 데이터 수집 완료.")
            return macro_df
        else:
            log_warning("거시경제 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        log_error(f"거시경제 데이터 수집 실패: {e}")
        return pd.DataFrame()

def fetch_ticker_price_data(stock_info, start_date, end_date):
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
        df_price = None
        
        try:
            df_price = fdr.DataReader(ticker, start_date, end_date)
        except:
            df_price = None
        
        # 데이터 품질 검증: 충분한 데이터가 있는지 확인
        # 251일(1년) + 60일(추가 버퍼) = 311일 이상의 데이터 필요
        if df_price is None or df_price.empty or len(df_price) < 251 + 60: 
            return (ticker, None)
            
        df_price.rename(columns={'Open':'시가', 'Close':'종가', 'High': '고가', 'Low': '저가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['시가', '종가', '고가', '저가', '거래량']].copy()
        df.sort_index(inplace=True)
        
        # 메모리 최적화: 원본 데이터프레임 해제
        del df_price
        gc.collect()
        
        return (ticker, df)
        
    except Exception as e:
        log_error(f"종목 {ticker} 데이터 다운로드 중 오류: {e}")
        return (ticker, None)



def calculate_ticker_features(ticker, df_price, stock_name=None):
    """
    단일 종목 피처 계산 함수 (CPU 작업)
    
    ProcessPoolExecutor로 병렬 처리되는 CPU 작업만 수행합니다.
    전역 변수(_global_marcap_data, _global_financial_data)를 사용하여
    fork 방식에서 Copy-on-Write로 효율적으로 데이터에 접근합니다.
    
    Args:
        ticker: 종목코드
        df_price: 다운로드된 주가 데이터 (시가, 종가, 고가, 저가, 거래량)
        
    Returns:
        pandas.DataFrame: 처리된 종목 데이터
    """
    global _global_marcap_data, _global_financial_data
    
    try:
        df = df_price.copy()
        
        # 전역 변수 검증
        if _global_marcap_data is None:
            log_error(f"종목 {ticker} 피처 계산 중 오류: 전역 시가총액 데이터가 설정되지 않았습니다.")
            return None
        
        # 시가총액 데이터 주입 (NASDAQ 버전: 정적 MarketCap 기반, 없으면 NaN)
        df_marcap_ticker = _global_marcap_data[_global_marcap_data['Code'] == ticker].copy()
        if df_marcap_ticker.empty:
            df['시가총액'] = np.nan
        else:
            if 'Marcap' in df_marcap_ticker.columns and 'date' in df_marcap_ticker.columns:
                df_marcap_ticker.sort_values(by='date', inplace=True)
                df = pd.merge_asof(left=df, right=df_marcap_ticker[['date', 'Marcap']], left_index=True, right_on='date', direction='backward')
                df.rename(columns={'Marcap': '시가총액'}, inplace=True)
            elif 'MarketCap' in df_marcap_ticker.columns:
                mc = pd.to_numeric(df_marcap_ticker['MarketCap'].iloc[0], errors='coerce')
                df['시가총액'] = mc
            elif '시가총액' in df_marcap_ticker.columns:
                mc = pd.to_numeric(df_marcap_ticker['시가총액'].iloc[0], errors='coerce')
                df['시가총액'] = mc
            else:
                df['시가총액'] = np.nan

        del df_marcap_ticker
        gc.collect()
        
        # 재무데이터 병합 (백업 프로젝트와 동일한 방식, 전역 변수 사용)
        if _global_financial_data is not None and not _global_financial_data.empty:
            df_financial_ticker = _global_financial_data[_global_financial_data['Code'] == ticker].copy()
            if not df_financial_ticker.empty:
                df_financial_ticker.sort_values(by='date', inplace=True)
                df = pd.merge_asof(left=df, right=df_financial_ticker[['date', 'PER', 'PBR', 'ROE', 'EPS', 'BPS']], 
                                   left_index=True, right_on='date', direction='backward')
                
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
            ma5 = df['종가'].rolling(window=5).mean()
            ma120_lvl = df['종가'].rolling(window=120).mean()
            ma240_lvl = df['종가'].rolling(window=240).mean()

            ma5_prev = ma5.shift(1)
            delta = (ma5 - ma5_prev) / ma5_prev.replace(0, np.nan)
            df['MA5_Angle_Deg'] = np.degrees(np.arctan(delta.astype(float)))

            df['Exclude_Rank'] = (ma120_lvl > ma5) & (ma240_lvl > ma5) & (df['MA5_Angle_Deg'] <= 0)
        except Exception:
            # 계산 실패 시에도 기존 파이프라인은 유지
            df['MA5_Angle_Deg'] = np.nan
            df['Exclude_Rank'] = False
        
        # 3. 52주 신고가 비율
        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        # target 변수 생성:
        # - 향후 10거래일 동안 최저가가 현재가 대비 -5% 이하로 내려가지 않고 (>= 0.95)
        # - 향후 10거래일 동안 최고가가 현재가 대비 +8% 이상 한 번이라도 상승하면 (>= 1.08)
        # => 1, 아니면 0
        min_price_10d = df['종가'].shift(-10).rolling(window=10, min_periods=1).min()
        max_price_10d = df['종가'].shift(-10).rolling(window=10, min_periods=1).max()
        # 조건: 최소값 >= 현재가격 * 0.95 AND 최대값 >= 현재가격 * 1.08
        df['target'] = ((min_price_10d / df['종가'] >= 0.95) & (max_price_10d / df['종가'] >= 1.08)).astype(int)
        
        # 중간 변수 삭제 (메모리 최적화)
        del min_price_10d, max_price_10d

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
    """실시간 데이터 수집 및 전처리
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        skip_factor_scores: True일 경우 팩터 점수 계산을 건너뜀 (학습용 데이터 생성 시 사용)
    """
    log_info(f"실시간 데이터 수집 시작 ({start_date} ~ {end_date})...")
    
    stock_list = fetch_stock_list()
    if stock_list.empty: 
        raise ValueError("종목 리스트를 가져올 수 없습니다.")
    
    # 미국 버전: KRX-MARCAP 대신 정적 시가총액(가능 시)을 사용
    df_marcap_long = stock_list[['종목코드']].copy()
    df_marcap_long.rename(columns={'종목코드': 'Code'}, inplace=True)
    if '시가총액' in stock_list.columns:
        df_marcap_long['MarketCap'] = pd.to_numeric(stock_list['시가총액'], errors='coerce')
    else:
        df_marcap_long['MarketCap'] = np.nan
    log_info("✅ 미국 정적 시가총액(가능 시) 준비 완료")
    
    # 재무데이터 수집 추가
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
    
    all_data = []
    stock_records = stock_list.to_dict('records')
    total_count = len(stock_records)  # 전체 종목 수 저장
    
    log_info(f"개별 종목 피처 데이터 생성 시작: {total_count}개 종목")
    
    # =================================================================
    # 1단계: ThreadPoolExecutor로 데이터 다운로드 (I/O 작업)
    # 배치 처리로 멈춘 future 문제 해결
    # =================================================================
    log_info("📥 1단계: 주가 데이터 다운로드 중...")
    downloaded_data = {}  # {ticker: df_price}
    download_failed = []
    
    # 배치 단위로 처리하여 멈춘 future 문제 해결
    batch_size = 100
    total_batches = (total_count + batch_size - 1) // batch_size
    download_completed = 0
    
    log_info(f"총 {total_count}개 종목을 {total_batches}개 배치로 처리합니다.")
    
    for batch_idx in range(0, total_count, batch_size):
        batch = stock_records[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        thread_workers = min(12, len(batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_workers) as executor:
            future_to_stock = {executor.submit(fetch_ticker_price_data, row, start_date, end_date): row for row in batch}
            
            # 배치 내 완료된 future 처리
            batch_completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    ticker, df_price = future.result(timeout=120)  # 2분 타임아웃
                    if df_price is not None:
                        downloaded_data[ticker] = df_price
                    else:
                        download_failed.append(ticker)
                except concurrent.futures.TimeoutError:
                    stock_info = future_to_stock.get(future, {})
                    ticker = stock_info.get('종목코드', 'Unknown')
                    download_failed.append(ticker)
                    log_warning(f"⏱️ 종목 {ticker} 전체 작업 타임아웃 (2분 초과) - 스킵하고 다음 종목으로 진행")
                except Exception as e:
                    stock_info = future_to_stock.get(future, {})
                    ticker = stock_info.get('종목코드', 'Unknown')
                    download_failed.append(ticker)
                    log_error(f"❌ 종목 {ticker} 데이터 다운로드 중 오류: {e}")
                
                batch_completed += 1
                download_completed += 1
                log_progress("주가 데이터 다운로드", download_completed, total_count)
            
            # 배치 완료 로그
            log_info(f"배치 {current_batch}/{total_batches} 완료 ({batch_completed}/{len(batch)}개 종목)")
        
        # 배치 간 메모리 정리 및 API 부하 방지
        gc.collect()
        if current_batch < total_batches:
            time.sleep(0.5)  # 배치 간 0.5초 대기
    
    log_info(f"✅ 데이터 다운로드 완료: {len(downloaded_data)}/{total_count}개 성공 (실패: {len(download_failed)}개)")
    
    if not downloaded_data:
        log_error("다운로드된 데이터가 없습니다.")
        raise ValueError("다운로드된 데이터가 없습니다.")
    
    # 전역 변수에 데이터 설정 (fork 방식에서 Copy-on-Write로 효율적으로 공유됨)
    set_global_feature_data(df_marcap_long, df_financial_long)
    
    # =================================================================
    # 2단계: ProcessPoolExecutor로 피처 계산 (CPU 작업)
    # =================================================================
    log_info("⚙️ 2단계: 피처 계산 중...")
    failed_count = 0
    
    try:
        # 논리 프로세서 수의 2/3 사용 (최소 2개)
        cpu_workers = max(2, int((os.cpu_count() or 8) * 2 / 3))
        
        # ProcessPoolExecutor 사용 (CPU 작업 병렬 처리)
        # WSL2 환경에서는 fork 방식으로 효율적으로 동작
        # 전역 변수를 사용하므로 큰 데이터를 인자로 전달하지 않아도 됨
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            # 전역 변수를 사용하므로 df_marcap_long, df_financial_long 인자 제거
            # 종목명 정보도 함께 전달
            stock_name_map = {row['종목코드']: row.get('종목명', None) for row in stock_records}
            future_to_ticker = {executor.submit(calculate_ticker_features, ticker, df_price, stock_name_map.get(ticker)): ticker 
                               for ticker, df_price in downloaded_data.items()}
        completed_count = 0
        total_calc_count = len(downloaded_data)
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                result_df = future.result(timeout=300)  # 5분 타임아웃
                if result_df is not None and isinstance(result_df, pd.DataFrame):
                    # 데이터 무결성 검증
                    if not result_df.empty and '종목코드' in result_df.columns:
                        all_data.append(result_df)
                    else:
                        failed_count += 1
                        ticker = future_to_ticker.get(future, 'Unknown')
                        log_warning(f"⚠️ 종목 {ticker} 데이터 무결성 검증 실패: 빈 데이터프레임 또는 필수 컬럼 누락")
                else:
                    failed_count += 1
                    ticker = future_to_ticker.get(future, 'Unknown')
            except concurrent.futures.TimeoutError:
                failed_count += 1
                ticker = future_to_ticker.get(future, 'Unknown')
                log_error(f"⏱️ 종목 {ticker} 피처 계산 타임아웃 (5분 초과)")
            except Exception as e:
                failed_count += 1
                ticker = future_to_ticker.get(future, 'Unknown')
                log_error(f"❌ 종목 {ticker} 피처 계산 중 오류: {e}")
            
            completed_count += 1
            # 진행률 로그 메시지 (PROGRESS 접두사 자동 추가됨) - 매번 출력하되 같은 줄에서 덮어쓰기
            log_progress("피처 계산", completed_count, total_calc_count)
            # 주기적 메모리 정리 (10개마다)
            if completed_count % 10 == 0:
                gc.collect()
    finally:
        # 전역 변수 초기화 (메모리 해제)
        clear_global_feature_data()
        
        # 다운로드된 데이터 메모리 해제
        del downloaded_data
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
    raw_feature_df = pd.concat(all_data).reset_index()
    
    raw_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_feature_df.dropna(subset=['date', '종목코드'], inplace=True)
    raw_feature_df['date'] = pd.to_datetime(raw_feature_df['date'])
    raw_feature_df.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)
    
    # 개별 데이터 리스트 메모리 해제
    del all_data
    gc.collect()
    
    # 거시경제 데이터 추가
    macro_df = _fetch_macro_data(start_date, end_date)
    if not macro_df.empty:
        raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')
    
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
