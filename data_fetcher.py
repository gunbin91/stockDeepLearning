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



def _get_stock_list_from_marcap(analysis_date=None):
    """KRX-MARCAP에서 주식 목록 가져오기 (통일된 함수)"""
    try:
        if analysis_date:
            # 과거 날짜용
            date_str = analysis_date.strftime('%Y%m%d')
            log_info(f"FinanceDataReader를 통해 {date_str} 기준 종목 시가총액 정보 수집 (KRX-MARCAP)...")
            df_marcap = fdr.StockListing('KRX-MARCAP', date_str)
        else:
            # 현재 날짜용
            log_info("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
            df_marcap = fdr.StockListing('KRX-MARCAP')
        
        # 스팩, 리츠 제외
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        
        # 상장주식수가 있는 경우만 필터링
        if 'Stocks' in df_marcap.columns:
            df_marcap = df_marcap[df_marcap['Stocks'] > 0]
        
        # 컬럼명 정리 및 종목코드 6자리 패딩
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        
        # 종목코드를 6자리로 패딩
        stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.zfill(6)
        
        log_info(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
        
    except Exception as e:
        error_msg = f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}"
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
            kospi = fdr.DataReader('KS11', start_date_str, end_date_str)
            if not kospi.empty:
                # 날짜 인덱스를 안전하게 처리 (format='mixed' 사용)
                kospi_copy = kospi.copy()
                kospi_copy.index = pd.to_datetime(kospi_copy.index, format='mixed', errors='coerce')
                macro_data['KOSPI'] = kospi_copy['Close']
        except Exception as e:
            log_warning(f"KOSPI 데이터 수집 실패: {e}")
        
        try:
            usdkrw = fdr.DataReader('USD/KRW', start_date_str, end_date_str)
            if not usdkrw.empty:
                # 날짜 인덱스를 안전하게 처리 (format='mixed' 사용)
                usdkrw_copy = usdkrw.copy()
                usdkrw_copy.index = pd.to_datetime(usdkrw_copy.index, format='mixed', errors='coerce')
                macro_data['USDKRW'] = usdkrw_copy['Close']
        except Exception as e:
            log_warning(f"USD/KRW 데이터 수집 실패: {e}")
        
        try:
            vix = fdr.DataReader('^VIX', start_date_str, end_date_str)
            if not vix.empty:
                # 날짜 인덱스를 안전하게 처리 (format='mixed' 사용)
                vix_copy = vix.copy()
                vix_copy.index = pd.to_datetime(vix_copy.index, format='mixed', errors='coerce')
                macro_data['VIX'] = vix_copy['Close']
        except Exception as e:
            log_warning(f"VIX 데이터 수집 실패: {e}")
            
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
                
                # 변동성 계산 (종목 데이터와 동일한 방식)
                # 변동성(1W) = 5일 롤링 표준편차 / 5일 롤링 평균
                macro_df['KOSPI_변동성(1W)'] = kospi_close.rolling(5).std() / kospi_close.rolling(5).mean()
                # 변동성(1M) = 20일 롤링 표준편차 / 20일 롤링 평균
                macro_df['KOSPI_변동성(1M)'] = kospi_close.rolling(20).std() / kospi_close.rolling(20).mean()
                # 변동성(3M) = 60일 롤링 표준편차 / 60일 롤링 평균
                macro_df['KOSPI_변동성(3M)'] = kospi_close.rolling(60).std() / kospi_close.rolling(60).mean()
                
                # 이격도 계산 (종목 데이터와 동일한 방식)
                # 이격도 = (현재가 / 이동평균) * 100
                for period in [5, 20, 60, 120]:
                    ma = kospi_close.rolling(window=period).mean()
                    macro_df[f'KOSPI_disparity_{period}'] = (kospi_close / ma) * 100
            
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
        
        # 하이브리드 방식으로 주가 데이터 수집 (Yahoo Finance → KRX → NAVER)
        df_price_full = None
        try:
            df_price_full = fdr.DataReader(ticker, fetch_start, end_date_for_fetch)
        except:
            try:
                df_price_full = fdr.DataReader(f'KRX:{ticker}', fetch_start, end_date_for_fetch)
            except:
                try:
                    df_price_full = fdr.DataReader(f'NAVER:{ticker}', fetch_start, end_date_for_fetch)
                except:
                    df_price_full = None
        
        if df_price_full is None or df_price_full.empty or len(df_price_full) < 251 + 60: return None, None
        df_price_full.rename(columns={'Open':'시가', 'Close':'종가', 'High':'고가', 'Low':'저가', 'Volume':'거래량'}, inplace=True)

        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None
        actual_analysis_date = df_temp.index.max()
        reference_date_price = df_temp.loc[actual_analysis_date]['종가']
        latest_current_price = df_price_full.iloc[-1]['종가']
        
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty or fs_data[['PER', 'PBR']].isnull().values.any(): return None, None
        
        latest_data = {} 

        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        latest_data['log_mktcap'] = np.log(reference_date_price * shares) if (reference_date_price * shares) > 0 else np.nan
        latest_data['이익수익률'] = 1 / fs_data['PER'].values[0] if fs_data['PER'].values[0] != 0 else np.nan

        latest_data['수익률(1M)'] = df_for_indicators['종가'].pct_change(20).iloc[-1]
        latest_data['수익률(3M)'] = df_for_indicators['종가'].pct_change(60).iloc[-1]
        latest_data['변동성(1W)'] = (df_for_indicators['종가'].rolling(5).std() / df_for_indicators['종가'].rolling(5).mean()).iloc[-1]
        latest_data['변동성(1M)'] = (df_for_indicators['종가'].rolling(20).std() / df_for_indicators['종가'].rolling(20).mean()).iloc[-1]
        latest_data['변동성(3M)'] = (df_for_indicators['종가'].rolling(60).std() / df_for_indicators['종가'].rolling(60).mean()).iloc[-1]
        latest_data['거래대금_MA5'] = df_for_indicators['거래대금'].rolling(5).mean().iloc[-1]
        latest_data['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(20).mean().iloc[-1]
        
        df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        df_for_indicators.ta.obv(close='종가', volume='거래량', append=True)
        df_for_indicators.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        
        bbands = df_for_indicators.ta.bbands(close='종가', length=20, std=2)
        # pandas-ta 버전업에 따른 볼린저밴드 컬럼명 변경 대응
        if bbands is not None and not bbands.empty:
            # 새로운 컬럼명 형식 확인
            if all(col in bbands.columns for col in ['BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBM_20_2.0_2.0']):
                latest_data['BBW_20_2'] = ((bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']).iloc[-1]
                # BB_Position 계산: (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
                current_price = df_for_indicators['종가'].iloc[-1]
                bb_lower = bbands['BBL_20_2.0_2.0'].iloc[-1]
                bb_upper = bbands['BBU_20_2.0_2.0'].iloc[-1]
                if bb_upper != bb_lower:
                    latest_data['BB_Position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
                else:
                    latest_data['BB_Position'] = 0.5  # 중앙값
            elif all(col in bbands.columns for col in ['BBL_20_2.0', 'BBU_20_2.0', 'BBM_20_2.0']):
                latest_data['BBW_20_2'] = ((bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']).iloc[-1]
                # BB_Position 계산: (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
                current_price = df_for_indicators['종가'].iloc[-1]
                bb_lower = bbands['BBL_20_2.0'].iloc[-1]
                bb_upper = bbands['BBU_20_2.0'].iloc[-1]
                if bb_upper != bb_lower:
                    latest_data['BB_Position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
                else:
                    latest_data['BB_Position'] = 0.5  # 중앙값
            else:
                latest_data['BBW_20_2'] = np.nan
                latest_data['BB_Position'] = np.nan
        else:
             latest_data['BBW_20_2'] = np.nan
             latest_data['BB_Position'] = np.nan

        for p in [5, 10, 60, 120, 240]:
            ma = df_for_indicators['종가'].rolling(window=p).mean()
            latest_data[f'disparity_{p}'] = ((df_for_indicators['종가'] / ma) * 100).iloc[-1]

        # 거래량 변동성 계수 계산 (1W, 1M, 3M)
        volume_std_5 = df_for_indicators['거래량'].rolling(5).std()
        volume_mean_5 = df_for_indicators['거래량'].rolling(5).mean()
        latest_data['거래량 변동성 계수(1W)'] = (volume_std_5 / volume_mean_5).iloc[-1] if volume_mean_5.iloc[-1] != 0 else np.nan
        
        volume_std_20 = df_for_indicators['거래량'].rolling(20).std()
        volume_mean_20 = df_for_indicators['거래량'].rolling(20).mean()
        latest_data['거래량 변동성 계수(1M)'] = (volume_std_20 / volume_mean_20).iloc[-1] if volume_mean_20.iloc[-1] != 0 else np.nan
        
        volume_std_60 = df_for_indicators['거래량'].rolling(60).std()
        volume_mean_60 = df_for_indicators['거래량'].rolling(60).mean()
        latest_data['거래량 변동성 계수(3M)'] = (volume_std_60 / volume_mean_60).iloc[-1] if volume_mean_60.iloc[-1] != 0 else np.nan

        latest_data['52주_신고가_비율'] = (df_for_indicators['종가'] / df_for_indicators['종가'].rolling(250).max()).iloc[-1]

        technical_features_to_add = ['ATRr_14', 'OBV', 'ADX_14']
        for feature in technical_features_to_add:
            if feature in df_for_indicators.columns:
                 latest_data[feature] = df_for_indicators[feature].iloc[-1]

        latest_data.update(fs_data.iloc[0].to_dict())
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price
        latest_data['기준일가'] = reference_date_price
        latest_data['전날종가'] = df_price_full.iloc[-2]['종가'] if len(df_price_full) >= 2 else latest_current_price  # 전날 종가 추가
        if '시가총액_기준일' in stock_info and pd.notna(stock_info['시가총액_기준일']):
            latest_data['시가총액'] = stock_info['시가총액_기준일'] / 1_0000_0000
        else:
            latest_data['시가총액'] = (reference_date_price * shares) / 1_0000_0000
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
    
    if latest_fs_df.empty or latest_fs_df.dropna().empty:
        log_warning("사용 가능한 재무 데이터가 없어 분석을 중단합니다.")
        return pd.DataFrame(), None

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
    
    # 인덱스를 datetime으로 확실히 변환
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)
    
    log_info("   🔗 거시경제 지표(KOSPI, USD/KRW, VIX)를 종목 데이터와 병합 중...")
    
    # merge_asof 대신 더 안전한 방법 사용
    try:
        final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    except Exception as e:
        log_warning(f"   ⚠️ merge_asof 실패, 일반 merge로 시도: {e}")
        # 일반 merge로 대체
        macro_df_reset = macro_df.reset_index()
        macro_df_reset.rename(columns={'index': 'date'}, inplace=True)
        final_df = pd.merge(final_df, macro_df_reset, on='date', how='left')
    
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