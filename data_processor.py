"""
실시간 데이터 처리 시스템
========================

이 파일은 주식 분석을 위한 데이터를 실시간으로 수집하고 처리합니다.
대용량 데이터를 효율적으로 처리하기 위해 병렬 처리와 메모리 최적화를 사용합니다.

주요 기능:
- 종목 목록 수집 (KOSPI, KOSDAQ)
- 시가총액 데이터 수집 및 일별 분배
- 재무 데이터 수집 (월초 데이터를 일별로 분배)
- 거시경제 데이터 수집
- 기술적 지표 계산
- 데이터 품질 검증 및 정제
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas_ta as ta
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from tqdm import tqdm
import os
import gc
import locale
import platform
import multiprocessing
import traceback
from collections import deque

# Windows 환경에서 multiprocessing 지원
if platform.system() == 'Windows':
    # Windows에서 multiprocessing을 사용할 때 필요
    multiprocessing.freeze_support()
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 로케일 설정 실패 시 기본값 유지
        pass

from scoring import calculate_factor_scores as calculate_factor_scores_func
from path_manager import path_manager
from logger import log_info, log_critical, log_error, log_warning, log_progress

# 통일된 경로 사용
PROJECT_ROOT = str(path_manager.project_root)
DATA_DIR = str(path_manager.data_dir)

# =================================================================
# 유틸리티 함수: 정규화된 선형회귀기울기 계산 (gpuStock 피처 호환)
# =================================================================
def calculate_normalized_linear_regression_slope(series: pd.Series, window: int = 5) -> pd.Series:
    """
    정규화된 선형회귀 기울기(%)를 계산합니다.
    - window 길이의 선형회귀 slope를 구한 뒤, 현재 값으로 나누고 100을 곱합니다.
    - MA120/MA240/KOSPI_MA20 기울기 피처 계산에 사용합니다.
    """
    if series is None or len(series) < window:
        return pd.Series([np.nan] * (len(series) if series is not None else 0), index=(series.index if series is not None else None))

    values = series.values
    n = len(values)
    slopes = np.full(n, np.nan, dtype=np.float64)

    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_centered_sq_sum = np.sum(x_centered ** 2)
    if x_centered_sq_sum == 0:
        return pd.Series(slopes, index=series.index)

    for i in range(window - 1, n):
        y = values[i - window + 1:i + 1]
        current_value = values[i]

        if current_value == 0 or np.isnan(current_value) or np.isnan(y).any():
            continue

        y_mean = np.nanmean(y)
        y_centered = y - y_mean
        numerator = np.sum(x_centered * y_centered)
        abs_slope = numerator / x_centered_sq_sum
        slopes[i] = (abs_slope / current_value) * 100

    return pd.Series(slopes, index=series.index)

def fetch_stock_list():
    """
    주식 목록 수집 함수
    
    KOSPI와 KOSDAQ에 상장된 모든 종목의 목록을 수집합니다.
    스팩, 리츠 등은 제외하고 일반 주식만 수집합니다.
    
    Returns:
        pandas.DataFrame: 종목코드, 종목명, 시장구분이 포함된 데이터프레임
    """
    try:
        # KRX에서 주식 목록 가져오기
        stock_list = fdr.StockListing('KRX')
        if not stock_list.empty:
            # 필요한 컬럼만 선택
            stock_list = stock_list[['Code', 'Name', 'Market']].copy()
            stock_list.columns = ['종목코드', '종목명', '시장구분']
            
            # KOSPI, KOSDAQ만 필터링
            stock_list = stock_list[stock_list['시장구분'].isin(['KOSPI', 'KOSDAQ'])]
            
            log_info(f"주식 목록 수집 완료: {len(stock_list)}개 종목")
            return stock_list
        else:
            log_error("주식 목록을 가져올 수 없습니다")
            return pd.DataFrame()
            
    except Exception as e:
        log_error(f"주식 목록 수집 실패: {e}")
        return pd.DataFrame()

def _fetch_financial_data(start_date, end_date):
    """월초 재무데이터 수집 및 일별 분배 (삼성전자 거래일 기준)"""
    try:
        log_info(f"월초 재무데이터 수집 시작: {start_date} ~ {end_date}")
        
        # 삼성전자 주가 데이터로 실제 거래일 확인
        try:
            from pykrx import stock
            trading_days = pd.to_datetime(stock.get_market_ohlcv(start_date, end_date, "005930").index).strftime('%Y%m%d').tolist()
            log_info(f"삼성전자 기준 실제 거래일: {len(trading_days)}개")
        except Exception as e:
            log_warning(f"삼성전자 거래일 확인 실패, 기본 로직 사용: {e}")
            # 폴백: 기본 거래일 생성
            start_date_obj = pd.to_datetime(start_date)
            end_date_obj = pd.to_datetime(end_date)
            all_dates = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
            trading_days = all_dates[all_dates.weekday < 5].strftime('%Y%m%d').tolist()
        
        # 월초 거래일 찾기 (삼성전자 거래일 기준)
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        monthly_first_dates = []
        current_date = start_date_obj
        
        while current_date <= end_date_obj:
            month_start = current_date.replace(day=1)
            
            # 월초 거래일 찾기 (1일~3일 중 실제 거래일)
            month_first_trading_day = None
            for day in range(1, 4):  # 1일~3일까지 확인
                target_date = month_start.replace(day=day)
                if target_date.strftime('%Y%m%d') in trading_days:
                    month_first_trading_day = target_date
                    break
            
            # 폴백: 과거 가까운 거래일 찾기
            if month_first_trading_day is None:
                for days_back in range(1, 10):  # 최대 10일 전까지 확인
                    target_date = month_start - pd.Timedelta(days=days_back)
                    if target_date.strftime('%Y%m%d') in trading_days:
                        month_first_trading_day = target_date
                        log_info(f"월초 거래일 폴백: {month_start.strftime('%Y-%m')} → {target_date.strftime('%Y-%m-%d')}")
                        break
            
            if month_first_trading_day is not None:
                monthly_first_dates.append(month_first_trading_day)
            else:
                log_warning(f"월초 거래일을 찾을 수 없음: {month_start.strftime('%Y-%m')}")
            
            current_date = month_start + pd.DateOffset(months=1)
        
        log_info(f"월초 재무데이터 수집 날짜: {len(monthly_first_dates)}개")
        
        # 월초 재무데이터 수집 (강화된 오류 처리)
        financial_dfs = []
        completed_count = 0
        failed_count = 0
        total_dates = len(monthly_first_dates)
        
        if total_dates == 0:
            log_warning("수집할 월초 거래일이 없습니다.")
            return pd.DataFrame()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(_fetch_monthly_financial_data, date): date for date in monthly_first_dates}
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty: 
                        result_df['date'] = pd.to_datetime(date_str)
                        financial_dfs.append(result_df)
                        completed_count += 1
                    else:
                        failed_count += 1
                        log_warning(f"재무데이터 수집 실패: {date_str.strftime('%Y-%m-%d')} ({failed_count}번째 실패)")
                except Exception as e:
                    failed_count += 1
                    log_error(f"재무데이터 수집 오류 ({date_str.strftime('%Y-%m-%d')}): {e}")
                
                # 진행률 로그 메시지 - 매번 출력하되 같은 줄에서 덮어쓰기
                log_progress("월초 재무데이터 수집", completed_count + failed_count, total_dates)
        
        # 수집 결과 검증
        success_rate = (completed_count / total_dates) * 100 if total_dates > 0 else 0
        log_info(f"재무데이터 수집 완료: {completed_count}/{total_dates} ({success_rate:.1f}%)")
        
        if not financial_dfs: 
            log_warning("수집된 재무데이터가 없습니다.")
            log_warning("재무데이터 수집에 실패했지만 분석을 계속합니다.")
            return pd.DataFrame()
        
        if success_rate < 50:
            log_warning(f"재무데이터 수집 성공률이 낮습니다: {success_rate:.1f}%")
            
        df_financial_long = pd.concat(financial_dfs, ignore_index=True)
        df_financial_long.sort_values(by=['Code', 'date'], inplace=True)
        
        # 월초 데이터를 일별로 분배 (데이터 정합성 검증 포함)
        df_financial_long = distribute_monthly_financial_data_to_daily(df_financial_long, start_date, end_date)
        
        # 데이터 정합성 검증
        if not df_financial_long.empty:
            total_records = len(df_financial_long)
            unique_codes = df_financial_long['Code'].nunique()
            date_range = df_financial_long['date'].nunique()
            
            log_info(f"✅ 월초 재무데이터 수집 및 일별 분배 완료: {total_records}개 레코드")
            log_info(f"데이터 정합성 검증: 종목수 {unique_codes}개, 거래일수 {date_range}개")
        else:
            log_warning("분배된 재무데이터가 없습니다.")
        
        return df_financial_long
        
    except Exception as e:
        log_error(f"월초 재무데이터 수집 실패: {e}")
        return pd.DataFrame()

def _fetch_monthly_financial_data(date):
    """특정 날짜의 재무데이터 수집"""
    try:
        # pykrx를 사용한 재무데이터 수집
        from pykrx import stock
        
        date_str = date.strftime('%Y%m%d')
        
        try:
            # 전체 시장 일괄 호출 방식
            df_fundamental = stock.get_market_fundamental(date_str, market="ALL")
            
            if df_fundamental.empty:
                log_warning(f"재무데이터가 비어있음 ({date_str})")
                return pd.DataFrame()
            
            # 데이터 처리
            if 'PBR' in df_fundamental.columns:
                df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
                
                if not df_fundamental.empty:
                    df_fundamental.reset_index(inplace=True)
                    df_fundamental.rename(columns={'티커': 'Code'}, inplace=True)  # Code 컬럼명으로 통일
                    
                    # 컬럼 선택
                    required_cols = ['Code', 'PBR', 'PER', 'EPS', 'BPS', 'DIV', 'DPS']
                    available_cols = [col for col in required_cols if col in df_fundamental.columns]
                    df_fundamental = df_fundamental[available_cols]
                    
                    # ROE 계산
                    if 'PBR' in df_fundamental.columns and 'PER' in df_fundamental.columns:
                        df_fundamental['ROE'] = np.where(df_fundamental['PER'] != 0, 
                                                       df_fundamental['PBR'] / df_fundamental['PER'], 
                                                       np.nan)
                    
                    return df_fundamental
                else:
                    log_warning(f"PBR > 0 조건을 만족하는 데이터가 없음 ({date_str})")
                    return pd.DataFrame()
            else:
                log_warning(f"PBR 컬럼이 없음 ({date_str})")
                return pd.DataFrame()
                
        except Exception as e:
            log_warning(f"재무데이터 API 호출 실패 ({date_str}): {e}")
            return pd.DataFrame()
        
    except Exception as e:
        log_error(f"재무데이터 수집 실패 ({date}): {e}")
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
            kospi = fdr.DataReader('KS11', start_date_str, end_date_str)
            if not kospi.empty:
                kospi_copy = kospi.copy()
                kospi_copy.index = pd.to_datetime(kospi_copy.index, format='mixed', errors='coerce')
                macro_data['KOSPI'] = kospi_copy['Close']
        except Exception as e:
            log_warning(f"KOSPI 데이터 수집 실패: {e}")
        
        try:
            usdkrw = fdr.DataReader('USD/KRW', start_date_str, end_date_str)
            if not usdkrw.empty:
                usdkrw_copy = usdkrw.copy()
                usdkrw_copy.index = pd.to_datetime(usdkrw_copy.index, format='mixed', errors='coerce')
                macro_data['USDKRW'] = usdkrw_copy['Close']
        except Exception as e:
            log_warning(f"USD/KRW 데이터 수집 실패: {e}")
        
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
            for col in macro_df.columns:
                macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
                macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)

            # gpuStock 피처 호환(원복): KOSPI_disparity_20, KOSPI_MA20_Slope, KOSPI_변동성(1M)
            try:
                if 'KOSPI' in macro_df.columns:
                    kospi_close = macro_df['KOSPI']

                    # 1) KOSPI_disparity_20
                    try:
                        kospi_ma20 = kospi_close.rolling(window=20).mean()
                        macro_df['KOSPI_disparity_20'] = (kospi_close / kospi_ma20) * 100
                    except Exception as e:
                        log_warning(f"KOSPI_disparity_20 계산 실패: {e}")
                        macro_df['KOSPI_disparity_20'] = np.nan

                    # 2) KOSPI_변동성(1M): std/mean (20일)
                    try:
                        kospi_std_20 = kospi_close.rolling(window=20).std()
                        kospi_mean_20 = kospi_close.rolling(window=20).mean()
                        macro_df['KOSPI_변동성(1M)'] = kospi_std_20 / kospi_mean_20.replace(0, np.nan)
                    except Exception as e:
                        log_warning(f"KOSPI_변동성(1M) 계산 실패: {e}")
                        macro_df['KOSPI_변동성(1M)'] = np.nan

                    # 3) KOSPI_MA20_Slope
                    try:
                        # 종목 MA20_Slope와 동일하게 window=5로 slope 계산
                        from data_processor import calculate_normalized_linear_regression_slope
                        kospi_ma20 = kospi_close.rolling(window=20).mean()
                        macro_df['KOSPI_MA20_Slope'] = calculate_normalized_linear_regression_slope(kospi_ma20, window=5)
                    except Exception as e:
                        log_warning(f"KOSPI_MA20_Slope 계산 실패: {e}")
                        macro_df['KOSPI_MA20_Slope'] = np.nan
            except Exception as e:
                log_warning(f"gpuStock 거시 피처 계산 실패: {e}")

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

def download_ticker_data(stock_info, start_date, end_date):
    """
    1단계: API 다운로드 함수 (I/O 바운드 작업)
    
    하나의 종목에 대해 주가 데이터만 수집합니다.
    
    Args:
        stock_info: 종목 정보 (종목코드, 종목명 등)
        start_date: 데이터 수집 시작일
        end_date: 데이터 수집 종료일
        
    Returns:
        tuple: (ticker, pandas.DataFrame) 또는 (ticker, None) - 주가 데이터만 포함된 데이터프레임
    """
    ticker = stock_info['종목코드']
    try:
        # =================================================================
        # 하이브리드 데이터 수집 방식 (3단계 폴백 시스템)
        # =================================================================
        # 1단계: Yahoo Finance (가장 빠르고 안정적)
        # 2단계: KRX (한국 거래소 공식 데이터)
        # 3단계: NAVER (최후의 수단)
        # 이렇게 하는 이유: 일부 종목은 특정 데이터 소스에서만 제공되기 때문
        df_price = None
        try:
            # 1차 시도: Yahoo Finance (가장 빠름)
            df_price = fdr.DataReader(ticker, start_date, end_date)
        except:
            try:
                # 2차 시도: KRX (한국 거래소 공식 데이터)
                df_price = fdr.DataReader(f'KRX:{ticker}', start_date, end_date)
            except:
                try:
                    # 3차 시도: NAVER (최후의 수단)
                    df_price = fdr.DataReader(f'NAVER:{ticker}', start_date, end_date)
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
        
        # 주가 데이터만 반환 (병합 및 피처 계산은 별도 함수에서 수행)
        return (ticker, df)
        
    except Exception as e:
        log_error(f"종목 {ticker} 데이터 다운로드 중 오류: {e}")
        # 오류 발생 시에도 메모리 정리
        try:
            gc.collect()
        except:
            pass
        return (ticker, None)


def merge_and_calculate_features(args):
    """
    2단계: 데이터 병합 및 피처 계산 함수 (CPU 바운드 작업)
    
    다운로드된 주가 데이터에 대해 다음 작업을 수행합니다:
    1. 시가총액 데이터 병합
    2. 재무 데이터 병합
    3. 기술적 지표 계산 (ATR, OBV, ADX, 볼린저 밴드 등)
    4. 수익률 및 변동성 계산
    5. 타겟 변수 생성 (향후 10거래일: -5% 이상 하락 없이, +8% 이상 상승 1회라도)
    
    Args:
        args: (ticker, df, df_marcap_ticker, df_financial_ticker) 튜플
            - ticker: 종목코드
            - df: 주가 데이터프레임
            - df_marcap_ticker: 시가총액 데이터 (해당 종목만, 이미 필터링됨)
            - df_financial_ticker: 재무 데이터 (해당 종목만, 이미 필터링됨)
        
    Returns:
        pandas.DataFrame: 완전한 피처 데이터프레임 또는 None
    """
    ticker, df, df_marcap_ticker, df_financial_ticker = args
    if df is None or df.empty:
        return None
    
    try:
        # =================================================================
        # 데이터 병합 (CPU 작업)
        # =================================================================
        
        # 시가총액 데이터 병합 (이미 필터링된 데이터 사용)
        if df_marcap_ticker.empty:
            return None
            
        df_marcap_ticker.sort_values(by='date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['date', 'Marcap']], left_index=True, right_on='date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        
        # 시가총액 데이터 메모리 해제
        del df_marcap_ticker
        gc.collect()
        
        # 재무데이터 병합 (이미 필터링된 데이터 사용)
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
        
        # =================================================================
        # 피처 계산 (CPU 작업)
        # =================================================================
        
        # 기존 방식과 동일한 최소한의 기술적 지표만 사용 (강화된 오류 처리)
        try:
            df.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        except Exception as e:
            df['ATRr_14'] = np.nan
        
        try:
            df.ta.obv(close='종가', volume='거래량', append=True)
        except Exception as e:
            df['OBV'] = np.nan
        
        try:
            df.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        except Exception as e:
            df['ADX_14'] = np.nan
        
        # 볼린저 밴드 계산
        try:
            bbands = df.ta.bbands(close='종가', length=20, std=2)
            if bbands is not None and not bbands.empty:
                # 새로운 컬럼명 형식 확인 (pandas-ta 최신 버전)
                if all(col in bbands.columns for col in ['BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBM_20_2.0_2.0']):
                    df['BBW_20_2'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']
                    # BB_Position 계산: (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
                    current_price = df['종가']
                    bb_lower = bbands['BBL_20_2.0_2.0']
                    bb_upper = bbands['BBU_20_2.0_2.0']
                    # 0으로 나누기 방지
                    bb_range = bb_upper - bb_lower
                    df['BB_Position'] = np.where(bb_range != 0, (current_price - bb_lower) / bb_range, 0.5)
                    # 0~1 범위로 제한
                    df['BB_Position'] = df['BB_Position'].clip(0, 1)
                # 기존 컬럼명 형식 확인 (pandas-ta 이전 버전)
                elif all(col in bbands.columns for col in ['BBL_20_2.0', 'BBU_20_2.0', 'BBM_20_2.0']):
                    df['BBW_20_2'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']
                    current_price = df['종가']
                    bb_lower = bbands['BBL_20_2.0']
                    bb_upper = bbands['BBU_20_2.0']
                    # 0으로 나누기 방지
                    bb_range = bb_upper - bb_lower
                    df['BB_Position'] = np.where(bb_range != 0, (current_price - bb_lower) / bb_range, 0.5)
                    # 0~1 범위로 제한
                    df['BB_Position'] = df['BB_Position'].clip(0, 1)
                else:
                    # 컬럼명을 찾을 수 없는 경우 기본값 설정
                    df['BBW_20_2'] = np.nan
                    df['BB_Position'] = np.nan
            else:
                # 볼린저 밴드 계산 실패 시 기본값 설정
                df['BBW_20_2'] = np.nan
                df['BB_Position'] = np.nan
        except Exception as e:
            df['BBW_20_2'] = np.nan
            df['BB_Position'] = np.nan
        
        # 볼린저 밴드 데이터 메모리 해제 (안전하게)
        try:
            if 'bbands' in locals():
                del bbands
        except:
            pass
        gc.collect()
        
        # 수익률 계산 (기존 유지)
        df['수익률(1M)'] = df['종가'].pct_change(20)
        df['수익률(3M)'] = df['종가'].pct_change(60)
        
        # 변동성 계산 (기존 유지: 점수 계산/기존 호환)
        df['변동성(1W)'] = df['종가'].rolling(5).std() / df['종가'].rolling(5).mean()
        df['변동성(1M)'] = df['종가'].rolling(20).std() / df['종가'].rolling(20).mean()
        df['변동성(3M)'] = df['종가'].rolling(60).std() / df['종가'].rolling(60).mean()

        # =================================================================
        # 최신 gpuStock 피처(정확한 컬럼명/계산식) 동기화
        # =================================================================
        
        # 거래대금 계산
        df['거래대금'] = df['종가'] * df['거래량']
        df['거래대금_MA5'] = df['거래대금'].rolling(5).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(20).mean()

        # (참고) 최신 gpuStock은 거래대금_log 계열을 학습 피처로 사용하지 않음
        
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
        
        # =================================================================
        # gpuStock 피처 세트 추가(기존 컬럼은 유지하고 "추가"만 함)
        # =================================================================

        # 1) ATRr_5/20/60 (ATR / 종가 * 100)
        try:
            atr_5 = df.ta.atr(high='고가', low='저가', close='종가', length=5)
            atr_20 = df.ta.atr(high='고가', low='저가', close='종가', length=20)
            atr_60 = df.ta.atr(high='고가', low='저가', close='종가', length=60)
            df['ATRr_5'] = (atr_5 / df['종가']) * 100 if atr_5 is not None else np.nan
            df['ATRr_20'] = (atr_20 / df['종가']) * 100 if atr_20 is not None else np.nan
            df['ATRr_60'] = (atr_60 / df['종가']) * 100 if atr_60 is not None else np.nan
        except Exception as e:
            log_warning(f"ATRr_5/20/60 계산 실패 ({ticker}): {e}")
            df['ATRr_5'] = np.nan
            df['ATRr_20'] = np.nan
            df['ATRr_60'] = np.nan

        # 2) RVOL (거래량 / 20일 평균 거래량)
        try:
            vol_ma20 = df['거래량'].rolling(window=20).mean()
            df['RVOL'] = (df['거래량'] / vol_ma20).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_warning(f"RVOL 계산 실패 ({ticker}): {e}")
            df['RVOL'] = np.nan

        # 3) 시총 회전율 (거래대금 롤링평균 / 시가총액 * 100)
        try:
            traded_value_ma5 = df['거래대금'].rolling(window=5).mean()
            traded_value_ma60 = df['거래대금'].rolling(window=60).mean()
            df['시총 회전율(1W)'] = (traded_value_ma5 / df['시가총액'] * 100).replace([np.inf, -np.inf], np.nan)
            df['시총 회전율(3M)'] = (traded_value_ma60 / df['시가총액'] * 100).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_warning(f"시총 회전율 계산 실패 ({ticker}): {e}")
            df['시총 회전율(1W)'] = np.nan
            df['시총 회전율(3M)'] = np.nan

        # 4) RSI_Signal_Oscillator (RSI_14 - RSI_14.rolling(9).mean())
        try:
            rsi_14 = df.ta.rsi(close='종가', length=14)
            if rsi_14 is not None and len(rsi_14) >= 9:
                rsi_14_ma9 = rsi_14.rolling(window=9).mean()
                df['RSI_Signal_Oscillator'] = rsi_14 - rsi_14_ma9
            else:
                df['RSI_Signal_Oscillator'] = np.nan
        except Exception as e:
            log_warning(f"RSI_Signal_Oscillator 계산 실패 ({ticker}): {e}")
            df['RSI_Signal_Oscillator'] = np.nan

        # 5) Z_Score_20
        try:
            mean_20 = df['종가'].rolling(window=20).mean()
            std_20 = df['종가'].rolling(window=20).std()
            df['Z_Score_20'] = (df['종가'] - mean_20) / std_20
        except Exception as e:
            log_warning(f"Z_Score_20 계산 실패 ({ticker}): {e}")
            df['Z_Score_20'] = np.nan

        # 6) Position_Range_60 (Donchian 위치)
        try:
            high_60 = df['고가'].rolling(window=60).max()
            low_60 = df['저가'].rolling(window=60).min()
            range_60 = high_60 - low_60
            df['Position_Range_60'] = np.where(range_60 != 0, (df['종가'] - low_60) / range_60, 0.5)
            df['Position_Range_60'] = pd.Series(df['Position_Range_60'], index=df.index).clip(0, 1)
        except Exception as e:
            log_warning(f"Position_Range_60 계산 실패 ({ticker}): {e}")
            df['Position_Range_60'] = np.nan

        # 7) Eff_Ratio_10
        try:
            change = df['종가'].diff(10).abs()
            volatility = df['종가'].diff(1).abs().rolling(10).sum()
            df['Eff_Ratio_10'] = (change / (volatility + 1e-9)).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_warning(f"Eff_Ratio_10 계산 실패 ({ticker}): {e}")
            df['Eff_Ratio_10'] = np.nan

        # 8) log_mktcap (시가총액 로그 변환: 안전하게)
        df['log_mktcap'] = np.nan
        try:
            m_mask = df['시가총액'] > 0
            df.loc[m_mask, 'log_mktcap'] = np.log(df.loc[m_mask, '시가총액'])
        except Exception:
            pass

        # 9) PBR_log
        df['PBR_log'] = np.nan
        try:
            if 'PBR' in df.columns:
                pbr_mask = df['PBR'] > 0
                df.loc[pbr_mask, 'PBR_log'] = np.log(df.loc[pbr_mask, 'PBR'])
        except Exception:
            pass

        # 10) MA120_Slope / MA240_Slope
        try:
            ma120 = df['종가'].rolling(window=120).mean()
            df['MA120_Slope'] = calculate_normalized_linear_regression_slope(ma120, window=5)
        except Exception as e:
            log_warning(f"MA120_Slope 계산 실패 ({ticker}): {e}")
            df['MA120_Slope'] = np.nan
        try:
            ma240 = df['종가'].rolling(window=240).mean()
            df['MA240_Slope'] = calculate_normalized_linear_regression_slope(ma240, window=5)
        except Exception as e:
            log_warning(f"MA240_Slope 계산 실패 ({ticker}): {e}")
            df['MA240_Slope'] = np.nan
        
        # 2. 이격도 계산 (gpuStock: 120/240 + disparity_20 추가)
        for p in [120, 240]:
            ma = df['종가'].rolling(window=p).mean()
            df[f'disparity_{p}'] = (df['종가'] / ma) * 100
        try:
            ma20 = df['종가'].rolling(window=20).mean()
            df['disparity_20'] = (df['종가'] / ma20) * 100
        except Exception as e:
            log_warning(f"disparity_20 계산 실패 ({ticker}): {e}")
            df['disparity_20'] = np.nan

        # 2-1) MA20_Slope (20일 이동평균선 기울기)
        try:
            ma20 = df['종가'].rolling(window=20).mean()
            df['MA20_Slope'] = calculate_normalized_linear_regression_slope(ma20, window=5)
        except Exception as e:
            log_warning(f"MA20_Slope 계산 실패 ({ticker}): {e}")
            df['MA20_Slope'] = np.nan

        # 2-2) Trend_Pullback_Score (내부 z_score_20 + ma20_slope 기반)
        try:
            mean_20 = df['종가'].rolling(20).mean()
            std_20 = df['종가'].rolling(20).std()
            z_score_20 = (df['종가'] - mean_20) / std_20.replace(0, np.nan)
            z_score_20 = z_score_20.fillna(0)

            ma20 = df['종가'].rolling(window=20).mean()
            ma20_slope = calculate_normalized_linear_regression_slope(ma20, window=5)

            ma20_slope_clean = ma20_slope.fillna(0)
            z_score_20_clean = z_score_20.fillna(0)

            base_score = np.abs(z_score_20_clean) * ma20_slope_clean
            condition_up_pullback = (ma20_slope_clean > 0) & (z_score_20_clean < 0)
            condition_up_overheat = (ma20_slope_clean > 0) & (z_score_20_clean >= 0)
            condition_down = (ma20_slope_clean <= 0)

            df['Trend_Pullback_Score'] = np.where(
                condition_up_pullback,
                base_score * 1.0,
                np.where(
                    condition_up_overheat,
                    base_score * 0.3,
                    np.where(condition_down, base_score * 0.1, 0),
                ),
            )

            nan_mask = ma20_slope.isna() | z_score_20.isna()
            df.loc[nan_mask, 'Trend_Pullback_Score'] = np.nan
        except Exception as e:
            log_warning(f"Trend_Pullback_Score 계산 실패 ({ticker}): {e}")
            df['Trend_Pullback_Score'] = np.nan

        # 2-3) RVOL(1W): 5일 평균 거래량 / 20일 평균 거래량
        try:
            vol_ma5 = df['거래량'].rolling(window=5).mean()
            vol_ma20 = df['거래량'].rolling(window=20).mean()
            df['RVOL(1W)'] = (vol_ma5 / vol_ma20).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_warning(f"RVOL(1W) 계산 실패 ({ticker}): {e}")
            df['RVOL(1W)'] = np.nan

        # 2-4) Log_Return_20: log(종가 / 20영업일 전 종가)
        try:
            df['Log_Return_20'] = np.log(df['종가'] / df['종가'].shift(20))
        except Exception as e:
            log_warning(f"Log_Return_20 계산 실패 ({ticker}): {e}")
            df['Log_Return_20'] = np.nan

        # 2-5) HV_Volatility_5/20/60: 1일 로그수익률의 rolling std (원복)
        try:
            log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
            df['HV_Volatility_5'] = log_ret_1d.rolling(window=5).std()
            df['HV_Volatility_20'] = log_ret_1d.rolling(window=20).std()
            df['HV_Volatility_60'] = log_ret_1d.rolling(window=60).std()
        except Exception as e:
            log_warning(f"HV_Volatility 계산 실패 ({ticker}): {e}")
            df['HV_Volatility_5'] = np.nan
            df['HV_Volatility_20'] = np.nan
            df['HV_Volatility_60'] = np.nan

        # 2-6) VWAP_Disparity_5: (종가 / 5일 VWAP - 1) * 100
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

        # 2-7) Max_Drawdown_20: 최근 20일 최대 낙폭(%)
        # roll_max_20 = 고가.rolling(20).max()
        # daily_dd_20 = (저가 / roll_max_20) - 1
        # Max_Drawdown_20 = daily_dd_20.rolling(20).min() * 100
        try:
            roll_max_20 = df['고가'].rolling(window=20).max()
            daily_dd_20 = (df['저가'] / roll_max_20) - 1
            df['Max_Drawdown_20'] = daily_dd_20.rolling(window=20).min() * 100
        except Exception as e:
            log_warning(f"Max_Drawdown_20 계산 실패 ({ticker}): {e}")
            df['Max_Drawdown_20'] = np.nan
        
        # 3. 52주 신고가 비율
        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        # =================================================================
        # 타겟 변수 생성 (요구사항)
        # - 향후 10거래일 동안:
        #   1) 최저 종가가 현재 종가 대비 -5% 이상 빠지지 않고 (future_min / now >= 0.95)
        #   2) 최고 종가가 현재 종가 대비 +8% 이상을 한 번이라도 찍으면 (future_max / now >= 1.08)
        # - 위 조건을 모두 만족하면 1, 아니면 0
        #
        # 주의:
        # - "향후 10거래일"을 정확히 보장하기 위해 min_periods=10로 계산합니다.
        # - 마지막 10거래일 구간은 미래 데이터가 부족하므로 target을 NaN으로 둡니다(학습 시 자동 제외).
        # =================================================================
        try:
            future_prices = df['종가'].shift(-1)
            future_max_10d = future_prices[::-1].rolling(window=10, min_periods=10).max()[::-1]
            future_min_10d = future_prices[::-1].rolling(window=10, min_periods=10).min()[::-1]

            cond = (future_min_10d / df['종가'] >= 0.95) & (future_max_10d / df['종가'] >= 1.08)
            valid_mask = future_max_10d.notna() & future_min_10d.notna() & df['종가'].notna()
            df['target'] = np.where(valid_mask, cond.astype(int), np.nan)
        except Exception as e:
            log_warning(f"target(10일/-5%방어/+8%상승) 생성 실패 ({ticker}): {e}")
            try:
                # 예외 상황에서는 min_periods=1로 완화하여 최대한 계산을 시도 (단, 마지막 구간 왜곡 가능)
                future_prices = df['종가'].shift(-1)
                future_max_10d = future_prices[::-1].rolling(window=10, min_periods=1).max()[::-1]
                future_min_10d = future_prices[::-1].rolling(window=10, min_periods=1).min()[::-1]
                cond = (future_min_10d / df['종가'] >= 0.95) & (future_max_10d / df['종가'] >= 1.08)
                valid_mask = future_max_10d.notna() & future_min_10d.notna() & df['종가'].notna()
                df['target'] = np.where(valid_mask, cond.astype(int), np.nan)
            except Exception:
                df['target'] = np.nan

        # =================================================================
        # 학습 타겟 제외 규칙 (사용자 요청)
        # - 조건: MA5 < MA120 AND MA5 < MA240 (120/240의 대소관계는 보지 않음)
        # - 그리고 MA5 당일 기울기(정의 A, 전일 대비 %변화 각도) <= 10도
        # => 위 조건을 모두 만족하는 샘플은 학습 대상에서 제외(drop)하기 위해 target을 NaN 처리합니다.
        #
        # 정의 A:
        #   ma5 = SMA(5)
        #   delta = (ma5_t - ma5_{t-1}) / ma5_{t-1}
        #   angle_deg = arctan(delta) * 180/pi
        # =================================================================
        try:
            ma5 = df['종가'].rolling(window=5).mean()
            ma120_lvl = df['종가'].rolling(window=120).mean()
            ma240_lvl = df['종가'].rolling(window=240).mean()

            ma5_prev = ma5.shift(1)
            delta = (ma5 - ma5_prev) / ma5_prev.replace(0, np.nan)
            ma5_angle_deg = np.degrees(np.arctan(delta.astype(float)))

            # 피처로도 사용 가능하도록 컬럼으로 저장
            df['MA5_Angle_Deg'] = ma5_angle_deg

            # 실시간/백테스팅에서 최종순위 산정 시 제외 플래그 (일자별로 동적 평가)
            df['Exclude_Rank'] = (ma5 < ma120_lvl) & (ma5 < ma240_lvl) & (ma5_angle_deg <= 10)

            exclude_mask = (ma5 < ma120_lvl) & (ma5 < ma240_lvl) & (ma5_angle_deg <= 10)
            if exclude_mask.any():
                df.loc[exclude_mask, 'target'] = np.nan
        except Exception:
            # 제외 규칙 계산 실패 시에도 기존 타겟 로직은 유지
            pass
        df['종목코드'] = ticker
        
        # 데이터 구조 설정
        # merge_asof 후 date 컬럼이 제거되므로 다시 추가
        df['date'] = df.index
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        # 오류 발생 시에도 메모리 정리
        try:
            gc.collect()
        except:
            pass
        return None


def process_single_ticker_data(stock_info, start_date, end_date, df_marcap_long, df_financial_long, pbar_lock):
    """
    단일 종목 데이터 처리 함수 (기존 호환성 유지용)
    
    이 함수는 download_ticker_data와 merge_and_calculate_features를 순차적으로 호출합니다.
    기존 코드와의 호환성을 위해 유지됩니다.
    """
    ticker, df = download_ticker_data(stock_info, start_date, end_date)
    if df is None:
        return None
    return merge_and_calculate_features((ticker, df, df_marcap_long, df_financial_long))



def _fetch_and_prepare_data(start_date, end_date, calculate_factor_scores=True):
    """
    실시간 데이터 수집 및 전처리
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        calculate_factor_scores: 팩터 점수 계산 여부 (기본값: True)
    """
    log_info(f"실시간 데이터 수집 시작 ({start_date} ~ {end_date})...")
    
    stock_list = fetch_stock_list()
    if stock_list.empty: 
        raise ValueError("종목 리스트를 가져올 수 없습니다.")
    
    try:
        # 월초 거래일만 수집하여 효율성 극대화
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        # 월별 첫 거래일만 수집
        monthly_first_dates = []
        current_date = start_date_obj
        
        while current_date <= end_date_obj:
            # 해당 월의 첫 거래일 찾기
            month_start = current_date.replace(day=1)
            month_dates = pd.date_range(start=month_start, end=month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1), freq='D')
            trading_dates = month_dates[month_dates.weekday < 5]
            
            if not trading_dates.empty:
                monthly_first_dates.append(trading_dates[0])
            
            # 다음 달로 이동
            current_date = month_start + pd.DateOffset(months=1)
        
        # 문자열로 변환
        marcap_dates = [date.strftime('%Y%m%d') for date in monthly_first_dates]
        
        if not marcap_dates:
            raise Exception("수집할 시가총액 데이터 날짜가 없습니다.")
        
        marcap_dfs = []
        completed_count = 0
        total_dates = len(marcap_dates)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in marcap_dates}
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty: 
                        result_df['date'] = pd.to_datetime(date_str)
                        marcap_dfs.append(result_df)
                except Exception: 
                    continue
                
                completed_count += 1
                # PROGRESS 접두사로 진행률 로그 출력 - 매번 출력하되 같은 줄에서 덮어쓰기
                log_progress("시가총액 데이터 수집", completed_count, total_dates)
        
        if not marcap_dfs: 
            raise Exception("수집된 시가총액 데이터가 없습니다.")
            
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'date'], inplace=True)
        
        # KONEX 제외 (KOSPI, KOSDAQ만 포함)
        if 'Market' in df_marcap_long.columns:
            before_count = len(df_marcap_long)
            df_marcap_long = df_marcap_long[df_marcap_long['Market'].isin(['KOSPI', 'KOSDAQ'])].copy()
            excluded_count = before_count - len(df_marcap_long)
            if excluded_count > 0:
                log_info(f"   🚫 KONEX 제외: {excluded_count}개 레코드 제외 (KOSPI/KOSDAQ만 포함)")
        
        # 시가총액 100억 미만 제외 (100억 = 10,000,000,000원)
        if 'Marcap' in df_marcap_long.columns:
            min_marcap = 10_000_000_000  # 100억원
            before_count = len(df_marcap_long)
            df_marcap_long = df_marcap_long[df_marcap_long['Marcap'] >= min_marcap].copy()
            excluded_count = before_count - len(df_marcap_long)
            if excluded_count > 0:
                log_info(f"   🚫 시가총액 100억 미만 제외: {excluded_count}개 레코드 제외 (시가총액 100억 이상만 포함)")
        
        # 원본 시가총액 데이터 메모리 해제
        del marcap_dfs
        import gc
        gc.collect()
        
        # 월초 데이터를 일별로 분배
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        all_dates = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
        trading_dates = all_dates[all_dates.weekday < 5]
        
        distributed_data = []
        
        for date in trading_dates:
            # 해당 월의 첫 거래일 찾기
            month_start = date.replace(day=1)
            month_dates = pd.date_range(start=month_start, end=month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1), freq='D')
            month_trading_dates = month_dates[month_dates.weekday < 5]
            
            if not month_trading_dates.empty:
                month_first_trading_day = month_trading_dates[0]
                
                # 해당 월의 첫 거래일 데이터를 현재 날짜에 복사
                monthly_data_for_date = df_marcap_long[df_marcap_long['date'] == month_first_trading_day].copy()
                if not monthly_data_for_date.empty:
                    monthly_data_for_date['date'] = date
                    distributed_data.append(monthly_data_for_date)
        
        if not distributed_data:
            log_warning("분배할 월초 데이터가 없습니다.")
        else:
            df_marcap_long = pd.concat(distributed_data, ignore_index=True)
            df_marcap_long.sort_values(by=['Code', 'date'], inplace=True)
        
        log_info(f"✅ 시가총액 데이터 수집 및 일별 분배 완료: {len(df_marcap_long)}개 레코드")
        
        # 필터링된 종목 리스트 생성 (시가총액 데이터 기준)
        # 시가총액 데이터에 있는 종목만 사용 (KONEX 제외, 시가총액 100억 미만 제외)
        valid_tickers = df_marcap_long['Code'].unique()
        before_count = len(stock_list)
        stock_list = stock_list[stock_list['종목코드'].isin(valid_tickers)].copy()
        excluded_count = before_count - len(stock_list)
        
        if excluded_count > 0:
            log_info(f"📋 종목 리스트 필터링 완료: {excluded_count}개 종목 제외 (KONEX 및 시가총액 100억 미만)")
            log_info(f"   ✅ 최종 분석 대상: {len(stock_list)}개 종목 (KOSPI/KOSDAQ, 시가총액 100억 이상)")
        else:
            log_info(f"📋 종목 리스트: {len(stock_list)}개 종목 (모두 유효)")
        
    except Exception as e:
        raise ConnectionError(f"시가총액 데이터 수집 실패: {e}")
    
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
    
    # 종목 리스트가 비어있으면 오류 발생
    if stock_list.empty:
        raise ValueError("필터링 후 유효한 종목이 없습니다. (KONEX 및 시가총액 100억 미만 제외)")
    
    all_data = []
    stock_records = stock_list.to_dict('records')
    
    log_info(f"📊 개별 종목 피처 데이터 생성 시작: {len(stock_records)}개 종목 (KONEX 제외, 시가총액 100억 이상)")
    
    # =================================================================
    # 1단계: API 다운로드 (I/O 바운드 - 스레드 사용)
    # =================================================================
    log_info("1단계: API 다운로드 중...")
    downloaded_data = {}
    completed_count = 0
    total_count = len(stock_records)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_stock = {executor.submit(download_ticker_data, row, start_date, end_date): row for row in stock_records}
        for future in concurrent.futures.as_completed(future_to_stock):
            try:
                ticker, result_df = future.result()
                if result_df is not None:
                    downloaded_data[ticker] = result_df
            except Exception: 
                pass
            
            completed_count += 1
            log_progress("API 다운로드", completed_count, total_count)
            # 주기적 메모리 정리 (10개마다)
            if completed_count % 10 == 0:
                gc.collect()
    
    log_info(f"✅ 1단계 완료: {len(downloaded_data)}개 종목 데이터 다운로드 완료")
    
    if not downloaded_data:
        log_error("다운로드된 데이터가 없습니다.")
        raise ValueError("다운로드된 데이터가 없습니다.")
    
    # =================================================================
    # 2단계: 데이터 병합 및 피처 계산 (CPU 바운드 - 멀티프로세싱 사용)
    # =================================================================
    log_info("2단계: 데이터 병합 및 피처 계산 중...")
    
    # 논리프로세서 수의 2/3 계산
    logical_cores = multiprocessing.cpu_count()
    process_workers = max(1, int(logical_cores * 2 / 3))
    log_info(f"   멀티프로세싱 워커 수: {process_workers}개 (논리프로세서: {logical_cores}개)")
    
    # 다운로드된 데이터를 튜플 리스트로 변환
    # Windows에서 multiprocessing을 사용할 때는 데이터 전달 최적화 필요
    # 각 프로세스에 필요한 데이터만 전달 (전체 데이터프레임 대신 필요한 부분만)
    merge_and_calc_args = []
    for ticker, df in downloaded_data.items():
        # 각 종목에 필요한 시가총액/재무데이터만 필터링하여 전달 (메모리 및 전송 오버헤드 감소)
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy() if not df_marcap_long.empty else pd.DataFrame()
        df_financial_ticker = df_financial_long[df_financial_long['Code'] == ticker].copy() if not df_financial_long.empty else pd.DataFrame()
        merge_and_calc_args.append((ticker, df, df_marcap_ticker, df_financial_ticker))
    
    completed_count = 0
    total_count = len(merge_and_calc_args)
    
    # Windows에서 ProcessPoolExecutor 사용 시 초기화 확인
    is_windows = (platform.system() == 'Windows')
    if is_windows:
        # Windows에서는 spawn 방식을 사용하므로 프로세스 생성 오버헤드가 큼
        log_info(f"   Windows 환경: spawn 방식 사용 (프로세스 생성 오버헤드 있음)")
        log_info(f"   데이터 전달 최적화: 각 종목별 필요한 데이터만 전달")

        # ✅ 안정성 우선: Windows spawn 환경에서는 워커 수를 과도하게 키우면
        # - IPC 파이프/큐 적체
        # - 프로세스 생성/종료 오버헤드
        # - 메모리 압력
        # 로 인해 BrokenProcessPool/Timeout/통신 오류가 발생하기 쉬움
        process_workers = min(process_workers, 6)
        log_info(f"   Windows 안정화: 워커 수 상한 적용 → {process_workers}개")

    # 폴백 기준(안전 우선)
    max_errors_before_fallback = max(5, int(total_count * 0.01))  # 1% 또는 최소 5개
    consecutive_errors_before_fallback = 5
    error_count = 0
    consecutive_errors = 0
    use_fallback_sequential = False

    # 대량 submit 방지: in-flight 제한(메모리/IPC 안정화)
    in_flight_limit = max(10, process_workers * 2)
    remaining = deque(merge_and_calc_args)
    
    # Windows spawn 환경 안정화를 위해 mp_context/max_tasks_per_child 사용
    # - max_tasks_per_child: 워커가 일정 작업 후 재시작되어 메모리 누수/파편화 누적 완화
    mp_ctx = multiprocessing.get_context('spawn') if is_windows else None
    max_tasks_per_child = 100 if is_windows else None

    try:
        with ProcessPoolExecutor(
            max_workers=process_workers,
            mp_context=mp_ctx,
            max_tasks_per_child=max_tasks_per_child
        ) as executor:
            future_to_args = {}

            # 초기 in-flight 채우기
            while remaining and len(future_to_args) < in_flight_limit:
                args = remaining.popleft()
                future = executor.submit(merge_and_calculate_features, args)
                future_to_args[future] = args

            log_info(f"   총 {total_count}개 작업 처리 시작 (in-flight 제한: {in_flight_limit}, 워커: {process_workers})")
            log_info(f"   💡 작업 관리자에서 Python 프로세스가 {process_workers}개 실행되는지 확인하세요")

            while future_to_args:
                done, _ = concurrent.futures.wait(
                    future_to_args.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    args = future_to_args.pop(future, None)
                    ticker = args[0] if args else "알 수 없음"
                    try:
                        result_df = future.result()
                        if result_df is not None:
                            all_data.append(result_df)
                        consecutive_errors = 0
                    except BrokenProcessPool as e:
                        # ✅ 풀 자체가 깨진 경우: 즉시 폴백
                        error_count += 1
                        consecutive_errors += 1
                        log_warning(
                            f"   종목 {ticker} 처리 중 BrokenProcessPool 발생 → 남은 작업은 순차 폴백합니다. "
                            f"({type(e).__name__}: {e})"
                        )
                        use_fallback_sequential = True
                        # 남은 args는 remaining에 유지되며, 아래에서 순차 처리
                        break
                    except Exception as e:
                        error_count += 1
                        consecutive_errors += 1
                        # 상세 원인 노출(Windows라도 문자열 로그는 가능)
                        err_type = type(e).__name__
                        err_msg = str(e)
                        log_warning(f"   종목 {ticker} 처리 중 오류 ({err_type}: {err_msg})")

                        # 에러가 반복되면 안정성 위해 폴백
                        if (error_count >= max_errors_before_fallback) or (consecutive_errors >= consecutive_errors_before_fallback):
                            log_warning(
                                f"   오류 누적({error_count}/{max_errors_before_fallback}) 또는 연속 오류({consecutive_errors})가 감지되어 "
                                f"남은 작업은 순차 폴백합니다."
                            )
                            use_fallback_sequential = True
                            break

                    completed_count += 1
                    log_progress("데이터 병합 및 피처 계산", completed_count, total_count)
                    if completed_count % 10 == 0:
                        gc.collect()

                if use_fallback_sequential:
                    # 더 이상 병렬로 진행하지 않음
                    break

                # in-flight 보충
                while remaining and len(future_to_args) < in_flight_limit:
                    args = remaining.popleft()
                    future = executor.submit(merge_and_calculate_features, args)
                    future_to_args[future] = args

    except BrokenProcessPool as e:
        # 풀 생성/운영 중 붕괴 시 폴백
        log_warning(f"   ProcessPoolExecutor 붕괴(BrokenProcessPool) 감지 → 순차 폴백합니다. ({e})")
        use_fallback_sequential = True
    except Exception as e:
        log_warning(f"   멀티프로세싱 실행 중 예외 → 순차 폴백합니다. ({type(e).__name__}: {e})")
        use_fallback_sequential = True

    # ✅ 폴백: 남은 작업(또는 전체)을 순차 처리하여 최대한 데이터 확보
    if use_fallback_sequential:
        log_info("   🔄 순차 폴백 모드로 남은 종목을 처리합니다(안정성 우선).")
        # 이미 완료된 개수는 유지, remaining에 남은 작업 처리
        while remaining:
            args = remaining.popleft()
            ticker = args[0]
            try:
                result_df = merge_and_calculate_features(args)
                if result_df is not None:
                    all_data.append(result_df)
            except Exception as e:
                # 순차 폴백에서도 예외는 계속 진행
                log_warning(f"   [폴백] 종목 {ticker} 처리 중 오류 ({type(e).__name__}: {e})")
                # 필요하면 상세 traceback도 남길 수 있음(로그 과다 방지 차원에서 1줄만)
            completed_count += 1
            log_progress("데이터 병합 및 피처 계산(폴백)", completed_count, total_count)
            if completed_count % 10 == 0:
                gc.collect()
    
    log_info(f"✅ 2단계 완료: {len(all_data)}개 종목 처리 완료")

    if not all_data: 
        log_error("처리된 데이터가 없습니다.")
        raise ValueError("처리된 데이터가 없습니다.")
    
    # 데이터 검증 로직 추가
    success_rate = len(all_data) / len(stock_records) * 100
    log_info(f"✅ 개별 종목 피처 데이터 생성 완료: {len(all_data)}개 종목 처리됨 (성공률: {success_rate:.1f}%)")
    
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
    import gc
    gc.collect()
    
    # 거시경제 데이터 추가
    macro_df = _fetch_macro_data(start_date, end_date)
    if not macro_df.empty:
        raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')
    
    raw_feature_df.sort_values(by=['date', '종목코드'], inplace=True)

    # 팩터 점수 계산 (옵션)
    if calculate_factor_scores:
        log_info("일별 팩터 점수 계산 중...")
        final_df = raw_feature_df.groupby('date', group_keys=False).apply(calculate_factor_scores_func).reset_index(drop=True)
    else:
        log_info("팩터 점수 계산을 건너뜁니다 (학습 데이터 수집 모드)")
        final_df = raw_feature_df.copy()
    
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
    import gc
    gc.collect()
    
    return final_df

def get_preprocessed_data(start_date, end_date, calculate_factor_scores=True):
    """
    실시간 데이터 전처리 함수
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        calculate_factor_scores: 팩터 점수 계산 여부 (기본값: True)
                                학습 데이터 수집 시에는 False로 설정하여 불필요한 연산 방지
    """
    try:
        log_info("🔄 실시간 데이터 수집 시작", context={
            "start_date": start_date,
            "end_date": end_date,
            "mode": "realtime",
            "calculate_factor_scores": calculate_factor_scores
        })
        
        return _fetch_and_prepare_data(start_date, end_date, calculate_factor_scores=calculate_factor_scores)
        
    except Exception as e:
        log_critical("실시간 데이터 수집 중 오류", exception=e, context={
            "start_date": start_date,
            "end_date": end_date
        })
        return pd.DataFrame()
