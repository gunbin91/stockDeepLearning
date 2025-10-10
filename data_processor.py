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

# 통일된 경로 사용
PROJECT_ROOT = str(path_manager.project_root)
DATA_DIR = str(path_manager.data_dir)

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

def process_single_ticker_data(stock_info, start_date, end_date, df_marcap_long, df_financial_long, pbar_lock):
    """
    단일 종목 데이터 처리 함수
    
    하나의 종목에 대해 다음 작업을 수행합니다:
    1. 주가 데이터 수집 (하이브리드 방식: Yahoo → KRX → NAVER)
    2. 시가총액 데이터 병합
    3. 재무 데이터 병합
    4. 기술적 지표 계산 (ATR, OBV, ADX, 볼린저 밴드 등)
    5. 수익률 및 변동성 계산
    6. 타겟 변수 생성 (15일 후 5% 상승 여부)
    
    Args:
        stock_info: 종목 정보 (종목코드, 종목명 등)
        start_date: 데이터 수집 시작일
        end_date: 데이터 수집 종료일
        df_marcap_long: 시가총액 데이터
        df_financial_long: 재무 데이터
        pbar_lock: 진행률 표시용 락
        
    Returns:
        pandas.DataFrame: 처리된 종목 데이터
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
            return None
            
        df_price.rename(columns={'Open':'시가', 'Close':'종가', 'High': '고가', 'Low': '저가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['시가', '종가', '고가', '저가', '거래량']].copy()
        df.sort_index(inplace=True)
        
        # 메모리 최적화: 원본 데이터프레임 해제
        # 대용량 데이터 처리 시 메모리 부족을 방지하기 위함
        del df_price
        import gc
        gc.collect()
        
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: 
            log_warning(f"⚠️ {ticker} 종목의 시가총액 데이터가 없습니다.")
            return None
            
        df_marcap_ticker.sort_values(by='date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['date', 'Marcap']], left_index=True, right_on='date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        
        # 시가총액 데이터 메모리 해제
        del df_marcap_ticker
        gc.collect()
        
        # 재무데이터 병합 (백업 프로젝트와 동일한 방식)
        if not df_financial_long.empty:
            df_financial_ticker = df_financial_long[df_financial_long['Code'] == ticker].copy()
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
        
        # 기존 방식과 동일한 최소한의 기술적 지표만 사용 (강화된 오류 처리)
        try:
            df.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        except Exception as e:
            log_warning(f"ATR 계산 실패 ({ticker}): {e}")
            df['ATRr_14'] = np.nan
        
        try:
            df.ta.obv(close='종가', volume='거래량', append=True)
        except Exception as e:
            log_warning(f"OBV 계산 실패 ({ticker}): {e}")
            df['OBV'] = np.nan
        
        try:
            df.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        except Exception as e:
            log_warning(f"ADX 계산 실패 ({ticker}): {e}")
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
            log_warning(f"볼린저 밴드 계산 실패 ({ticker}): {e}")
            df['BBW_20_2'] = np.nan
            df['BB_Position'] = np.nan
        
        # 볼린저 밴드 데이터 메모리 해제 (안전하게)
        try:
            if 'bbands' in locals():
                del bbands
        except:
            pass
        gc.collect()
        # 기존 방식과 동일한 기본 지표들만 사용 (과도한 기술적 지표 제거)
        
        # 수익률 계산
        df['수익률(1M)'] = df['종가'].pct_change(20)
        df['수익률(3M)'] = df['종가'].pct_change(60)
        
        # 변동성 계산
        df['변동성(1W)'] = df['종가'].rolling(5).std() / df['종가'].rolling(5).mean()
        df['변동성(1M)'] = df['종가'].rolling(20).std() / df['종가'].rolling(20).mean()
        df['변동성(3M)'] = df['종가'].rolling(60).std() / df['종가'].rolling(60).mean()
        
        # 거래대금 계산
        df['거래대금'] = df['종가'] * df['거래량']
        df['거래대금_MA5'] = df['거래대금'].rolling(5).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(20).mean()
        
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
        df['log_mktcap'] = np.log(df['시가총액'])
        
        # 2. 이격도 계산 (120일, 240일)
        for p in [120, 240]:
            ma = df['종가'].rolling(window=p).mean()
            df[f'disparity_{p}'] = (df['종가'] / ma) * 100
        
        # 3. 52주 신고가 비율
        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        # target 변수 생성
        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        
        # 데이터 구조 설정
        # merge_asof 후 date 컬럼이 제거되므로 다시 추가
        df['date'] = df.index
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        log_error(f"종목 {ticker} 처리 중 오류: {e}")
        # 오류 발생 시에도 메모리 정리
        try:
            gc.collect()
        except:
            pass
        return None



def _fetch_and_prepare_data(start_date, end_date):
    """실시간 데이터 수집 및 전처리"""
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
    
    log_info(f"개별 종목 피처 데이터 생성 시작: {len(stock_records)}개 종목")
    
    completed_count = 0
    total_count = len(stock_records)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_stock = {executor.submit(process_single_ticker_data, row, start_date, end_date, df_marcap_long, df_financial_long, None): row for row in stock_records}
        for future in concurrent.futures.as_completed(future_to_stock):
            try:
                result_df = future.result()
                if result_df is not None: 
                    all_data.append(result_df)
            except Exception: 
                pass
            
            completed_count += 1
            # 진행률 로그 메시지 (PROGRESS 접두사 자동 추가됨) - 매번 출력하되 같은 줄에서 덮어쓰기
            log_progress("개별 종목 피처 데이터 생성", completed_count, total_count)
            # 주기적 메모리 정리 (10개마다)
            if completed_count % 10 == 0:
                import gc
                gc.collect()

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

def get_preprocessed_data(start_date, end_date):
    """실시간 데이터 전처리 함수"""
    try:
        log_info("🔄 실시간 데이터 수집 시작", context={
            "start_date": start_date,
            "end_date": end_date,
            "mode": "realtime"
        })
        
        return _fetch_and_prepare_data(start_date, end_date)
        
    except Exception as e:
        log_critical("실시간 데이터 수집 중 오류", exception=e, context={
            "start_date": start_date,
            "end_date": end_date
        })
        return pd.DataFrame()
