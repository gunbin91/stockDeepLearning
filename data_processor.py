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
from tqdm import tqdm
import os
import gc
import locale
import platform
import multiprocessing
import time

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
    
    KOSPI와 KOSDAQ에 상장된 모든 종목의 목록을 수집합니다.
    스팩, 리츠 등은 제외하고 일반 주식만 수집합니다.
    
    Returns:
        pandas.DataFrame: 종목코드, 종목명, 시장구분이 포함된 데이터프레임
    """
    max_retries = 3
    retry_delay = 2  # 초
    
    for attempt in range(max_retries):
        try:
            # KRX-MARCAP에서 주식 목록 가져오기 (더 안정적)
            # 1차 시도: KRX-MARCAP
            try:
                if attempt > 0:
                    log_info(f"주식 목록 수집 재시도 중... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                
                log_info("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
                df_marcap = fdr.StockListing('KRX-MARCAP')
                
                if df_marcap is not None and not df_marcap.empty:
                    # 스팩, 리츠 제외
                    df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
                    
                    # KONEX 제외 (KOSPI, KOSDAQ만 포함)
                    if 'Market' in df_marcap.columns:
                        df_marcap = df_marcap[df_marcap['Market'].isin(['KOSPI', 'KOSDAQ'])].copy()
                        log_info(f"KONEX 제외 후 종목 수: {len(df_marcap)}개")
                    
                    # 상장주식수가 있는 경우만 필터링
                    if 'Stocks' in df_marcap.columns:
                        df_marcap = df_marcap[df_marcap['Stocks'] > 0]
                    
                    # 시가총액 100억 미만 제외 (100억 = 10,000,000,000원)
                    if 'Marcap' in df_marcap.columns:
                        min_marcap = 10_000_000_000  # 100억원
                        before_count = len(df_marcap)
                        df_marcap = df_marcap[df_marcap['Marcap'] >= min_marcap].copy()
                        excluded_count = before_count - len(df_marcap)
                        if excluded_count > 0:
                            log_info(f"시가총액 100억 미만 종목 {excluded_count}개 제외")
                    
                    # 컬럼명 정리 및 종목코드 6자리 패딩
                    stock_list = df_marcap[['Code', 'Name']].copy()
                    stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
                    stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.zfill(6)
                    
                    # 시장구분 추가 (Market 컬럼이 있으면 사용, 없으면 추정)
                    if 'Market' in df_marcap.columns:
                        stock_list['시장구분'] = df_marcap['Market']
                    else:
                        # 종목코드로 시장구분 추정 (KOSPI: 000000-099999, KOSDAQ: 그 외)
                        stock_list['시장구분'] = stock_list['종목코드'].apply(
                            lambda x: 'KOSPI' if x.startswith('0') and len(x) >= 2 and int(x[:2]) < 10 else 'KOSDAQ'
                        )
                    
                    log_info(f"주식 목록 수집 완료: {len(stock_list)}개 종목")
                    return stock_list
                else:
                    log_warning("KRX-MARCAP에서 빈 데이터를 받았습니다. KRX로 재시도합니다.")
            except Exception as e1:
                log_warning(f"KRX-MARCAP 수집 실패, KRX로 재시도: {e1}")
            
            # 2차 시도: KRX (기존 방식)
            try:
                stock_list = fdr.StockListing('KRX')
                if stock_list is not None and not stock_list.empty:
                    # 필요한 컬럼만 선택
                    stock_list = stock_list[['Code', 'Name', 'Market']].copy()
                    stock_list.columns = ['종목코드', '종목명', '시장구분']
                    
                    # KOSPI, KOSDAQ만 필터링
                    stock_list = stock_list[stock_list['시장구분'].isin(['KOSPI', 'KOSDAQ'])]
                    
                    # 종목코드 6자리 패딩
                    stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.zfill(6)
                    
                    log_info(f"주식 목록 수집 완료: {len(stock_list)}개 종목")
                    return stock_list
                else:
                    log_warning("KRX에서 빈 데이터를 받았습니다.")
            except Exception as e2:
                log_warning(f"KRX 수집도 실패: {e2}")
            
            # 마지막 시도가 아니면 재시도
            if attempt < max_retries - 1:
                continue
            else:
                # 모든 시도 실패
                log_error("주식 목록을 가져올 수 없습니다 (모든 재시도 실패)")
                return pd.DataFrame()
                
        except Exception as e:
            if attempt < max_retries - 1:
                log_warning(f"주식 목록 수집 중 오류 발생 (재시도 예정): {e}")
                time.sleep(retry_delay)
                continue
            else:
                log_error(f"주식 목록 수집 실패 (모든 재시도 실패): {e}")
                return pd.DataFrame()
    
    # 여기 도달하면 안 되지만 안전장치
    log_error("주식 목록을 가져올 수 없습니다")
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
                
                # API 부하 방지를 위한 지연 추가
                time.sleep(0.2)
                
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
            # pct_1d는 KOSPI만 생성 (USDKRW, VIX는 생성하지 않음)
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
                    kospi_ma20 = kospi_close.rolling(window=20).mean()
                    macro_df['KOSPI_MA20_Slope'] = calculate_normalized_linear_regression_slope(kospi_ma20, window=5)
                except Exception as e:
                    log_warning(f"KOSPI_MA20_Slope 계산 실패: {e}")
                    macro_df['KOSPI_MA20_Slope'] = np.nan
            
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
        # =================================================================
        # 하이브리드 데이터 수집 방식 (3단계 폴백 시스템)
        # =================================================================
        # 1단계: Yahoo Finance (가장 빠르고 안정적)
        # 2단계: KRX (한국 거래소 공식 데이터)
        # 3단계: NAVER (최후의 수단)
        df_price = None
        
        # 1차 시도: Yahoo Finance (가장 빠름)
        try:
            df_price = fdr.DataReader(ticker, start_date, end_date)
        except:
            df_price = None
        
        if df_price is None or df_price.empty:
            # 2차 시도: KRX (한국 거래소 공식 데이터)
            try:
                df_price = fdr.DataReader(f'KRX:{ticker}', start_date, end_date)
            except:
                df_price = None
        
        if df_price is None or df_price.empty:
            # 3차 시도: NAVER (최후의 수단)
            try:
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
        import gc
        df = df_price.copy()
        
        # 전역 변수 검증
        if _global_marcap_data is None:
            log_error(f"종목 {ticker} 피처 계산 중 오류: 전역 시가총액 데이터가 설정되지 않았습니다.")
            return None
        
        # 시가총액 데이터 병합 (전역 변수 사용)
        df_marcap_ticker = _global_marcap_data[_global_marcap_data['Code'] == ticker].copy()
        if df_marcap_ticker.empty: 
            log_warning(f"⚠️ {ticker} 종목의 시가총액 데이터가 없습니다.")
            return None
            
        df_marcap_ticker.sort_values(by='date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['date', 'Marcap']], left_index=True, right_on='date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        
        # 시가총액 데이터 메모리 해제
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
            
            # ATRr_20 계산 (기준 - 1M): "이 종목의 기초 체급은?"
            if atr_20 is not None:
                df['ATRr_20'] = (atr_20 / df['종가']) * 100
            else:
                df['ATRr_20'] = np.nan
            
        # ATRr_60 피처 제거
        except Exception as e:
            log_warning(f"ATR 계산 실패 ({ticker}): {e}")
            df['ATRr_5'] = np.nan
            df['ATRr_20'] = np.nan
        
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
        
        # 수익률 계산
        df['수익률(1M)'] = df['종가'].pct_change(20)
        df['수익률(3M)'] = df['종가'].pct_change(60)
        
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
        
        # RVOL(1W): 5일 평균 거래량 / 20일 평균 거래량
        try:
            거래량_5일_평균 = df['거래량'].rolling(window=5).mean()
            거래량_20일_평균 = df['거래량'].rolling(window=20).mean()
            df['RVOL(1W)'] = 거래량_5일_평균 / 거래량_20일_평균
            df['RVOL(1W)'] = df['RVOL(1W)'].replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_warning(f"RVOL(1W) 계산 실패 ({ticker}): {e}")
            df['RVOL(1W)'] = np.nan
        
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
        
        # Z_Score_20 계산 (표준화 이격)
        try:
            mean_20 = df['종가'].rolling(20).mean()
            std_20 = df['종가'].rolling(20).std()
            df['Z_Score_20'] = (df['종가'] - mean_20) / std_20
        except Exception as e:
            log_warning(f"Z_Score_20 계산 실패 ({ticker}): {e}")
            df['Z_Score_20'] = np.nan
        
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
        
        # 2-1. [신규 추가] 로그 수익률(1M) (Log Return 1M)
        # 1개월(20거래일) 간의 로그 수익률 누적
        try:
            df['Log_Return_20'] = np.log(df['종가'] / df['종가'].shift(20))
        except Exception as e:
            log_warning(f"Log_Return_20 계산 실패 ({ticker}): {e}")
            df['Log_Return_20'] = np.nan
            
        # 2-2. [신규 추가] HV변동성(1M) (Historical Volatility 1M)
        # 일별 로그 수익률의 20일 이동 표준편차
        try:
            log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
            df['HV_Volatility_20'] = log_ret_1d.rolling(window=20).std()
        except Exception as e:
            log_warning(f"HV_Volatility_20 계산 실패 ({ticker}): {e}")
            df['HV_Volatility_20'] = np.nan
        
        # 2-4. [신규 추가] HV변동성(3M) (Historical Volatility 3M)
        # 일별 로그 수익률의 60일 이동 표준편차
        try:
            if 'log_ret_1d' not in locals():
                log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
            df['HV_Volatility_60'] = log_ret_1d.rolling(window=60).std()
        except Exception as e:
            log_warning(f"HV_Volatility_60 계산 실패 ({ticker}): {e}")
            df['HV_Volatility_60'] = np.nan
            
        # 2-3. [신규 추가] VWAP Disparity(1M) (VWAP 괴리율 1개월)
        # 최근 20일 거래대금 가중 평균 가격 대비 현재가 비율
        try:
            tp = (df['고가'] + df['저가'] + df['종가']) / 3
            money = tp * df['거래량']
            
            sum_money_20 = money.rolling(window=20).sum()
            sum_vol_20 = df['거래량'].rolling(window=20).sum()
            
            vwap_20 = sum_money_20 / (sum_vol_20 + 1e-9)
            df['VWAP_Disparity_20'] = (df['종가'] / vwap_20 - 1) * 100
        except Exception as e:
            log_warning(f"VWAP_Disparity_20 계산 실패 ({ticker}): {e}")
            df['VWAP_Disparity_20'] = np.nan
        
        # 2-1. [신규 추가] 로그 수익률(1M) (Log Return 1M)
        # 1개월(20거래일) 간의 로그 수익률 누적
        try:
            df['Log_Return_20'] = np.log(df['종가'] / df['종가'].shift(20))
        except Exception as e:
            log_warning(f"Log_Return_20 계산 실패 ({ticker}): {e}")
            df['Log_Return_20'] = np.nan
            
        # 2-2. [신규 추가] HV변동성(1M) (Historical Volatility 1M)
        # 일별 로그 수익률의 20일 이동 표준편차
        try:
            log_ret_1d = np.log(df['종가'] / df['종가'].shift(1))
            df['HV_Volatility_20'] = log_ret_1d.rolling(window=20).std()
        except Exception as e:
            log_warning(f"HV_Volatility_20 계산 실패 ({ticker}): {e}")
            df['HV_Volatility_20'] = np.nan
            
        # 2-3. [신규 추가] VWAP Disparity(1M) (VWAP 괴리율 1개월)
        # 최근 20일 거래대금 가중 평균 가격 대비 현재가 비율
        try:
            tp = (df['고가'] + df['저가'] + df['종가']) / 3
            money = tp * df['거래량']
            
            sum_money_20 = money.rolling(window=20).sum()
            sum_vol_20 = df['거래량'].rolling(window=20).sum()
            
            vwap_20 = sum_money_20 / (sum_vol_20 + 1e-9)
            df['VWAP_Disparity_20'] = (df['종가'] / vwap_20 - 1) * 100
        except Exception as e:
            log_warning(f"VWAP_Disparity_20 계산 실패 ({ticker}): {e}")
            df['VWAP_Disparity_20'] = np.nan
        
        # 3. 이격도 계산 (120일, 240일) - disparity_20 제거
        for p in [120, 240]:
            ma = df['종가'].rolling(window=p).mean()
            df[f'disparity_{p}'] = (df['종가'] / ma) * 100
        
        # MA120_Slope 계산 (120일 이동평균선 기울기)
        try:
            ma120 = df['종가'].rolling(window=120).mean()
            df['MA120_Slope'] = calculate_normalized_linear_regression_slope(ma120, window=5)
        except Exception as e:
            log_warning(f"MA120_Slope 계산 실패 ({ticker}): {e}")
            df['MA120_Slope'] = np.nan
        
        # MA240_Slope 계산 (240일 이동평균선 기울기)
        try:
            ma240 = df['종가'].rolling(window=240).mean()
            df['MA240_Slope'] = calculate_normalized_linear_regression_slope(ma240, window=5)
        except Exception as e:
            log_warning(f"MA240_Slope 계산 실패 ({ticker}): {e}")
            df['MA240_Slope'] = np.nan
        
        # 3. 52주 신고가 비율
        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        # target 변수 생성 (10일 내 5% 이하로 떨어지지 않고 7% 이상 상승)
        # 10일 후부터 역방향으로 10일 윈도우의 최소값과 최대값 계산
        min_price_10d = df['종가'].shift(-10).rolling(window=10, min_periods=1).min()
        max_price_10d = df['종가'].shift(-10).rolling(window=10, min_periods=1).max()
        # 조건: 최소값 >= 현재가격 * 0.95 (5% 이하로 떨어지지 않음) AND 최대값 >= 현재가격 * 1.07 (7% 이상 상승)
        df['target'] = ((min_price_10d / df['종가'] >= 0.95) & (max_price_10d / df['종가'] > 1.07)).astype(int)
        
        # 중간 변수 삭제 (메모리 최적화)
        del min_price_10d, max_price_10d
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
                
                # API 부하 방지를 위한 지연 추가
                time.sleep(0.2)
                
                completed_count += 1
                # PROGRESS 접두사로 진행률 로그 출력 - 매번 출력하되 같은 줄에서 덮어쓰기
                log_progress("시가총액 데이터 수집", completed_count, total_dates)
        
        if not marcap_dfs: 
            raise Exception("수집된 시가총액 데이터가 없습니다.")
            
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'date'], inplace=True)
        
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
                import gc
                gc.collect()
    finally:
        # 전역 변수 초기화 (메모리 해제)
        clear_global_feature_data()
        
        # 다운로드된 데이터 메모리 해제
        del downloaded_data
        import gc
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
    import gc
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
    import gc
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
