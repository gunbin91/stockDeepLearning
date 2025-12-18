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

# =================================================================
# 유틸리티 함수: 정규화된 선형회귀기울기 계산 (최신 값만, gpuStock 피처 호환)
# =================================================================
def calculate_normalized_linear_regression_slope_latest(series: pd.Series, window: int = 5) -> float:
    """
    정규화된 선형회귀 기울기(%)의 최신 값만 반환합니다.
    - MA120_Slope/MA240_Slope 등 단일 시점 피처 계산에 사용합니다.
    """
    try:
        if series is None or len(series) < window:
            return np.nan
        y = series.iloc[-window:].values
        current_value = series.iloc[-1]
        if current_value == 0 or np.isnan(current_value) or np.isnan(y).any():
            return np.nan
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_centered = x - x_mean
        denom = np.sum(x_centered ** 2)
        if denom == 0:
            return np.nan
        y_mean = np.nanmean(y)
        y_centered = y - y_mean
        numerator = np.sum(x_centered * y_centered)
        abs_slope = numerator / denom
        return float((abs_slope / current_value) * 100)
    except Exception:
        return np.nan

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
            for col in macro_df.columns:
                macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
                macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)

            # gpuStock 피처 호환: KOSPI_disparity_20, KOSPI_변동성(1M), KOSPI_MA20_Slope
            try:
                if 'KOSPI' in macro_df.columns:
                    kospi_close = macro_df['KOSPI']
                    kospi_ma20 = kospi_close.rolling(window=20).mean()
                    macro_df['KOSPI_disparity_20'] = (kospi_close / kospi_ma20) * 100
                    kospi_std_20 = kospi_close.rolling(window=20).std()
                    kospi_mean_20 = kospi_close.rolling(window=20).mean()
                    macro_df['KOSPI_변동성(1M)'] = kospi_std_20 / kospi_mean_20
                    # 시계열 전체 slope는 data_processor 쪽에서 계산하지만,
                    # 여기서는 분석용으로 동일한 값이 나오도록 시계열로 계산
                    # (마지막 값만 사용되더라도 merge_asof 시 전체 필요)
                    from data_processor import calculate_normalized_linear_regression_slope
                    macro_df['KOSPI_MA20_Slope'] = calculate_normalized_linear_regression_slope(kospi_ma20, window=5)
            except Exception as e:
                log_warning(f"gpuStock 거시 피처 계산 실패: {e}")

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
        # 기존 로직은 PER/PBR 결측이면 종목 자체를 제외했지만,
        # gpuStock 피처 통일(특히 PBR_log) 관점에서 과도한 제외를 피합니다.
        if fs_data.empty:
            return None, None
        
        latest_data = {} 

        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        mktcap_won = (reference_date_price * shares)
        latest_data['log_mktcap'] = np.log(mktcap_won) if mktcap_won > 0 else np.nan
        # 루트 분석은 이익수익률을 기존 UI/호환을 위해 유지하되,
        # 학습 피처는 gpuStock 세트로 통일되므로 없어도 됨.
        try:
            per_val = fs_data['PER'].values[0] if 'PER' in fs_data.columns else np.nan
            latest_data['이익수익률'] = 1 / per_val if pd.notna(per_val) and per_val != 0 else np.nan
        except Exception:
            latest_data['이익수익률'] = np.nan

        latest_data['수익률(1M)'] = df_for_indicators['종가'].pct_change(20).iloc[-1]
        latest_data['수익률(3M)'] = df_for_indicators['종가'].pct_change(60).iloc[-1]
        latest_data['변동성(1W)'] = (df_for_indicators['종가'].rolling(5).std() / df_for_indicators['종가'].rolling(5).mean()).iloc[-1]
        latest_data['변동성(1M)'] = (df_for_indicators['종가'].rolling(20).std() / df_for_indicators['종가'].rolling(20).mean()).iloc[-1]
        latest_data['변동성(3M)'] = (df_for_indicators['종가'].rolling(60).std() / df_for_indicators['종가'].rolling(60).mean()).iloc[-1]
        latest_data['거래대금_MA5'] = df_for_indicators['거래대금'].rolling(5).mean().iloc[-1]
        latest_data['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(20).mean().iloc[-1]

        # 최신 gpuStock 피처: RVOL(1W)
        try:
            거래량_5일_평균 = df_for_indicators['거래량'].rolling(window=5).mean().iloc[-1]
            거래량_20일_평균 = df_for_indicators['거래량'].rolling(window=20).mean().iloc[-1]
            if pd.notna(거래량_20일_평균) and 거래량_20일_평균 != 0 and pd.notna(거래량_5일_평균):
                latest_data['RVOL(1W)'] = float(거래량_5일_평균 / 거래량_20일_평균)
            else:
                latest_data['RVOL(1W)'] = np.nan
        except Exception as e:
            log_warning(f"RVOL(1W) 계산 실패 ({ticker}): {e}")
            latest_data['RVOL(1W)'] = np.nan
        
        df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        df_for_indicators.ta.obv(close='종가', volume='거래량', append=True)
        df_for_indicators.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)

        # gpuStock 피처: ATRr_5/20/60
        try:
            atr_5 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=5)
            atr_20 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=20)
            atr_60 = df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=60)
            latest_close = df_for_indicators['종가'].iloc[-1]
            latest_data['ATRr_5'] = (atr_5.iloc[-1] / latest_close) * 100 if atr_5 is not None and pd.notna(latest_close) and latest_close != 0 else np.nan
            latest_data['ATRr_20'] = (atr_20.iloc[-1] / latest_close) * 100 if atr_20 is not None and pd.notna(latest_close) and latest_close != 0 else np.nan
            latest_data['ATRr_60'] = (atr_60.iloc[-1] / latest_close) * 100 if atr_60 is not None and pd.notna(latest_close) and latest_close != 0 else np.nan
        except Exception as e:
            log_warning(f"ATRr_5/20/60 계산 실패 ({ticker}): {e}")
            latest_data['ATRr_5'] = np.nan
            latest_data['ATRr_20'] = np.nan
            latest_data['ATRr_60'] = np.nan

        # gpuStock 피처: RVOL
        try:
            vol_ma20 = df_for_indicators['거래량'].rolling(window=20).mean().iloc[-1]
            latest_vol = df_for_indicators['거래량'].iloc[-1]
            if pd.notna(vol_ma20) and vol_ma20 != 0 and pd.notna(latest_vol):
                latest_data['RVOL'] = float(latest_vol / vol_ma20)
            else:
                latest_data['RVOL'] = np.nan
        except Exception:
            latest_data['RVOL'] = np.nan

        # gpuStock 피처: 시총 회전율(1W/3M) (거래대금 롤링평균 / 시가총액 * 100)
        try:
            traded_value_ma5 = df_for_indicators['거래대금'].rolling(window=5).mean().iloc[-1]
            traded_value_ma60 = df_for_indicators['거래대금'].rolling(window=60).mean().iloc[-1]
            if mktcap_won and mktcap_won > 0:
                latest_data['시총 회전율(1W)'] = float((traded_value_ma5 / mktcap_won) * 100) if pd.notna(traded_value_ma5) else np.nan
                latest_data['시총 회전율(3M)'] = float((traded_value_ma60 / mktcap_won) * 100) if pd.notna(traded_value_ma60) else np.nan
            else:
                latest_data['시총 회전율(1W)'] = np.nan
                latest_data['시총 회전율(3M)'] = np.nan
        except Exception:
            latest_data['시총 회전율(1W)'] = np.nan
            latest_data['시총 회전율(3M)'] = np.nan

        # gpuStock 피처: RSI_Signal_Oscillator
        try:
            rsi_14 = df_for_indicators.ta.rsi(close='종가', length=14)
            if rsi_14 is not None and len(rsi_14) >= 9:
                rsi_14_ma9 = rsi_14.rolling(window=9).mean()
                latest_data['RSI_Signal_Oscillator'] = float(rsi_14.iloc[-1] - rsi_14_ma9.iloc[-1]) if pd.notna(rsi_14.iloc[-1]) and pd.notna(rsi_14_ma9.iloc[-1]) else np.nan
            else:
                latest_data['RSI_Signal_Oscillator'] = np.nan
        except Exception:
            latest_data['RSI_Signal_Oscillator'] = np.nan

        # gpuStock 피처: Z_Score_20, Position_Range_60, Eff_Ratio_10
        try:
            mean_20 = df_for_indicators['종가'].rolling(20).mean().iloc[-1]
            std_20 = df_for_indicators['종가'].rolling(20).std().iloc[-1]
            latest_data['Z_Score_20'] = float((df_for_indicators['종가'].iloc[-1] - mean_20) / std_20) if pd.notna(mean_20) and pd.notna(std_20) and std_20 != 0 else np.nan
        except Exception:
            latest_data['Z_Score_20'] = np.nan
        try:
            high_60 = df_for_indicators['고가'].rolling(60).max().iloc[-1]
            low_60 = df_for_indicators['저가'].rolling(60).min().iloc[-1]
            if pd.notna(high_60) and pd.notna(low_60) and (high_60 - low_60) != 0:
                pos = (df_for_indicators['종가'].iloc[-1] - low_60) / (high_60 - low_60)
                latest_data['Position_Range_60'] = float(max(0, min(1, pos)))
            else:
                latest_data['Position_Range_60'] = 0.5
        except Exception:
            latest_data['Position_Range_60'] = np.nan
        try:
            change = df_for_indicators['종가'].diff(10).abs()
            volatility = df_for_indicators['종가'].diff(1).abs().rolling(10).sum()
            latest_data['Eff_Ratio_10'] = float(change.iloc[-1] / (volatility.iloc[-1] + 1e-9)) if pd.notna(change.iloc[-1]) and pd.notna(volatility.iloc[-1]) else np.nan
        except Exception:
            latest_data['Eff_Ratio_10'] = np.nan

        # gpuStock 피처: MA120_Slope / MA240_Slope
        try:
            ma120 = df_for_indicators['종가'].rolling(window=120).mean()
            latest_data['MA120_Slope'] = calculate_normalized_linear_regression_slope_latest(ma120, window=5)
        except Exception:
            latest_data['MA120_Slope'] = np.nan
        try:
            ma240 = df_for_indicators['종가'].rolling(window=240).mean()
            latest_data['MA240_Slope'] = calculate_normalized_linear_regression_slope_latest(ma240, window=5)
        except Exception:
            latest_data['MA240_Slope'] = np.nan
        
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

        # 이격도: 120/240 + disparity_20
        for p in [120, 240]:
            ma = df_for_indicators['종가'].rolling(window=p).mean()
            latest_data[f'disparity_{p}'] = ((df_for_indicators['종가'] / ma) * 100).iloc[-1]
        try:
            ma20 = df_for_indicators['종가'].rolling(window=20).mean()
            latest_data['disparity_20'] = (df_for_indicators['종가'] / ma20 * 100).iloc[-1]
        except Exception as e:
            log_warning(f"disparity_20 계산 실패 ({ticker}): {e}")
            latest_data['disparity_20'] = np.nan

        # Log_Return_20
        try:
            if len(df_for_indicators) >= 21 and pd.notna(df_for_indicators['종가'].shift(20).iloc[-1]) and df_for_indicators['종가'].shift(20).iloc[-1] != 0:
                latest_data['Log_Return_20'] = float(np.log(df_for_indicators['종가'].iloc[-1] / df_for_indicators['종가'].shift(20).iloc[-1]))
            else:
                latest_data['Log_Return_20'] = np.nan
        except Exception as e:
            log_warning(f"Log_Return_20 계산 실패 ({ticker}): {e}")
            latest_data['Log_Return_20'] = np.nan

        # HV_Volatility_5/20/60 (1일 로그수익률 rolling std)
        try:
            log_ret_1d = np.log(df_for_indicators['종가'] / df_for_indicators['종가'].shift(1))
            latest_data['HV_Volatility_5'] = float(log_ret_1d.rolling(window=5).std().iloc[-1]) if len(log_ret_1d) >= 5 else np.nan
            latest_data['HV_Volatility_20'] = float(log_ret_1d.rolling(window=20).std().iloc[-1]) if len(log_ret_1d) >= 20 else np.nan
            latest_data['HV_Volatility_60'] = float(log_ret_1d.rolling(window=60).std().iloc[-1]) if len(log_ret_1d) >= 60 else np.nan
        except Exception:
            latest_data['HV_Volatility_5'] = np.nan
            latest_data['HV_Volatility_20'] = np.nan
            latest_data['HV_Volatility_60'] = np.nan

        # VWAP_Disparity_5
        try:
            if len(df_for_indicators) >= 5:
                tp = (df_for_indicators['고가'] + df_for_indicators['저가'] + df_for_indicators['종가']) / 3
                money = tp * df_for_indicators['거래량']
                sum_money_5 = money.rolling(window=5).sum().iloc[-1]
                sum_vol_5 = df_for_indicators['거래량'].rolling(window=5).sum().iloc[-1]
                vwap_5 = sum_money_5 / (sum_vol_5 + 1e-9)
                latest_data['VWAP_Disparity_5'] = float((df_for_indicators['종가'].iloc[-1] / vwap_5 - 1) * 100) if pd.notna(vwap_5) and vwap_5 != 0 else np.nan
            else:
                latest_data['VWAP_Disparity_5'] = np.nan
        except Exception as e:
            log_warning(f"VWAP_Disparity_5 계산 실패 ({ticker}): {e}")
            latest_data['VWAP_Disparity_5'] = np.nan

        # Max_Drawdown_20
        try:
            roll_max_20 = df_for_indicators['고가'].rolling(window=20).max()
            daily_dd_20 = (df_for_indicators['저가'] / roll_max_20) - 1
            latest_data['Max_Drawdown_20'] = float((daily_dd_20.rolling(window=20).min() * 100).iloc[-1])
        except Exception as e:
            log_warning(f"Max_Drawdown_20 계산 실패 ({ticker}): {e}")
            latest_data['Max_Drawdown_20'] = np.nan

        # Trend_Pullback_Score (내부 z_score_20 + ma20_slope 기반)
        try:
            mean_20 = df_for_indicators['종가'].rolling(20).mean()
            std_20 = df_for_indicators['종가'].rolling(20).std()
            z_score_20 = (df_for_indicators['종가'] - mean_20) / std_20.replace(0, np.nan)
            z_score_20 = z_score_20.fillna(0)

            ma20 = df_for_indicators['종가'].rolling(window=20).mean()
            ma20_slope = calculate_normalized_linear_regression_slope_latest(ma20, window=5)
            ma20_slope_clean = 0 if pd.isna(ma20_slope) else float(ma20_slope)
            z_last = float(z_score_20.iloc[-1]) if len(z_score_20) else 0.0
            base_score = abs(z_last) * ma20_slope_clean

            if ma20_slope_clean > 0 and z_last < 0:
                latest_data['Trend_Pullback_Score'] = base_score * 1.0
            elif ma20_slope_clean > 0 and z_last >= 0:
                latest_data['Trend_Pullback_Score'] = base_score * 0.3
            elif ma20_slope_clean <= 0:
                latest_data['Trend_Pullback_Score'] = base_score * 0.1
            else:
                latest_data['Trend_Pullback_Score'] = 0.0
        except Exception as e:
            log_warning(f"Trend_Pullback_Score 계산 실패 ({ticker}): {e}")
            latest_data['Trend_Pullback_Score'] = np.nan

        latest_data['52주_신고가_비율'] = (df_for_indicators['종가'] / df_for_indicators['종가'].rolling(250).max()).iloc[-1]

        technical_features_to_add = ['ATRr_14', 'OBV', 'ADX_14']
        for feature in technical_features_to_add:
            if feature in df_for_indicators.columns:
                 latest_data[feature] = df_for_indicators[feature].iloc[-1]

        latest_data.update(fs_data.iloc[0].to_dict())

        # gpuStock 피처: PBR_log (fs_data 병합 후 계산)
        try:
            if 'PBR' in latest_data and pd.notna(latest_data['PBR']) and latest_data['PBR'] > 0:
                latest_data['PBR_log'] = float(np.log(latest_data['PBR']))
            else:
                latest_data['PBR_log'] = np.nan
        except Exception:
            latest_data['PBR_log'] = np.nan

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