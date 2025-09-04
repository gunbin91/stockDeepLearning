# data_fetcher.py

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm
import os
from datetime import datetime, timedelta
import config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# <<< ✨ 핵심 수정 1: 사용할 최종 데이터베이스 경로 지정 ✨ >>>
FINANCIAL_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'financial_data_pykrx_pit.parquet')

# <<< ✨ 핵심 수정 2: DART API 호출 함수 전체 삭제 ✨ >>>
# get_latest_annual_fs_http 함수가 더 이상 필요 없으므로 삭제합니다.

def get_fs_data_from_pit(stock_list, selected_analysis_date):
    """ 
    분석 기준일 시점에서 사용 가능한 최신 재무 지표를 pykrx DB에서 조회합니다.
    (오늘/과거 날짜 분석 모두 이 함수를 사용하도록 통일)
    """
    try:
        funda_df = pd.read_parquet(FINANCIAL_DB_PATH)
    except FileNotFoundError:
        print(f"[오류] 재무 지표 데이터베이스 파일({FINANCIAL_DB_PATH})을 찾을 수 없습니다.")
        print("먼저 `scripts/build_db_pykrx.py`를 실행하여 데이터베이스를 생성해주세요."); return pd.DataFrame()
    
    print(f"분석 기준일({selected_analysis_date.strftime('%Y-%m-%d')}): pykrx DB에서 최신 재무 지표 조회.")
    analysis_date_ts = pd.to_datetime(selected_analysis_date)
    
    available_funda = funda_df[funda_df['date'] <= analysis_date_ts].copy()
    if available_funda.empty:
        print(f"경고: {analysis_date_ts.strftime('%Y-%m-%d')} 이전에 집계된 재무 지표 데이터가 없습니다."); return pd.DataFrame()
    
    latest_funda = available_funda.sort_values('date').drop_duplicates(subset=['종목코드'], keep='last')
    
    result_df = pd.merge(stock_list[['종목코드']], latest_funda, on='종목코드', how='left')
    print(f"✅ 사용 가능한 재무 지표 처리 완료: {len(result_df.dropna())}개 기업")
    return result_df

def fetch_stock_list():
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
    try:
        df_marcap = fdr.StockListing('KRX-MARCAP')
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        stock_list = stock_list[stock_list['상장주식수'] > 0]
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}"); return pd.DataFrame()

def _fetch_macro_data(start_date, end_date):
    print(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date); usdkrw = fdr.DataReader('USD/KRW', start_date, end_date); vix = fdr.DataReader('^VIX', start_date, end_date)
        macro_df = pd.concat([kospi['Close'].rename('KOSPI'), usdkrw['Close'].rename('USDKRW'), vix['Close'].rename('VIX')], axis=1).ffill()
        for col in ['KOSPI', 'USDKRW', 'VIX']:
            if col in macro_df.columns: macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1); macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        print("✅ 거시 경제 지표 수집 완료."); return macro_df
    except Exception as e:
        print(f"거시 경제 지표 수집 실패: {e}"); return pd.DataFrame()

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']; shares = stock_info['상장주식수']
    try:
        df_price_full = fdr.DataReader(ticker, start_date_for_fetch, end_date_for_fetch)
        if df_price_full.empty or len(df_price_full) < 251: return None, None
        df_price_full.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None
        actual_analysis_date = df_temp.index.max(); reference_date_price = df_temp.loc[actual_analysis_date]['종가']
        latest_current_price = df_price_full.iloc[-1]['종가']
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        df_for_indicators['수익률(1W)'] = df_for_indicators['종가'].pct_change(5); df_for_indicators['수익률(2W)'] = df_for_indicators['종가'].pct_change(10)
        df_for_indicators['수익률(1M)'] = df_for_indicators['종가'].pct_change(20); df_for_indicators['수익률(3M)'] = df_for_indicators['종가'].pct_change(60)
        df_for_indicators['변동성(1M)'] = df_for_indicators['종가'].rolling(20).std() / df_for_indicators['종가'].rolling(20).mean()
        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']; df_for_indicators['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(20).mean()
        df_for_indicators['MA5'] = df_for_indicators['종가'].rolling(5).mean(); df_for_indicators['MA20'] = df_for_indicators['종가'].rolling(20).mean()
        df_for_indicators['단기 정배열'] = (df_for_indicators['MA5'] > df_for_indicators['MA20']).astype(int)
        df_for_indicators['52주_최고가'] = df_for_indicators['종가'].rolling(250).max(); df_for_indicators['52주_신고가_비율'] = df_for_indicators['종가'] / df_for_indicators['52주_최고가']
        df_for_indicators.ta.rsi(close='종가', length=14, append=True); df_for_indicators.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty or fs_data[['PER', 'PBR']].isnull().values.any(): return None, None
        
        latest_data = df_for_indicators.loc[actual_analysis_date].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']; latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price; latest_data['기준일가'] = reference_date_price
        
        if '시가총액_기준일' in stock_info and pd.notna(stock_info['시가총액_기준일']) and stock_info['시가총액_기준일'] > 0:
            market_cap = stock_info['시가총액_기준일']
        else:
            market_cap = reference_date_price * shares
        latest_data['시가총액'] = market_cap / 1_0000_0000
        
        # <<< ✨ 핵심 수정 3: pykrx 데이터 처리 로직으로 단일화 ✨ >>>
        # 이미 계산된 PER, PBR, ROE 값을 DB에서 바로 가져와 사용합니다.
        latest_data['PER'] = fs_data['PER'].values[0]
        latest_data['PBR'] = fs_data['PBR'].values[0]
        latest_data['ROE'] = fs_data['ROE'].values[0] if 'ROE' in fs_data.columns else np.nan
            
        latest_data['log_mktcap'] = np.log(market_cap) if market_cap > 0 else np.nan
        return latest_data, actual_analysis_date
    except Exception: return None, None

def fetch_all_data(stock_list, selected_analysis_date):
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=400)).strftime('%Y-%m-%d')

    # <<< ✨ 핵심 수정 4: 데이터 조회 로직 단일화 ✨ >>>
    # 오늘/과거 구분 없이 모든 재무 데이터 조회를 get_fs_data_from_pit으로 통일합니다.
    latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date)
    
    if latest_fs_df.empty or latest_fs_df.dropna().empty:
        print("사용 가능한 재무 데이터가 없어 분석을 중단합니다."); return pd.DataFrame(), None

    if selected_analysis_date.date() < today.date():
        print(f"과거 분석(기준일={selected_analysis_date.strftime('%Y-%m-%d')}): 기준일의 시가총액 데이터를 수집합니다.")
        try:
            df_marcap_past = fdr.StockListing('KRX-MARCAP', selected_analysis_date.strftime('%Y%m%d'))
            df_marcap_past.rename(columns={'Code': '종목코드'}, inplace=True)
            
            stock_list = pd.merge(
                stock_list[['종목코드', '종목명']],
                df_marcap_past[['종목코드', 'Marcap', 'Stocks']],
                on='종목코드',
                how='inner'
            )
            stock_list.rename(columns={'Stocks': '상장주식수', 'Marcap': '시가총액_기준일'}, inplace=True)
            print(f"✅ 기준일({selected_analysis_date.strftime('%Y-%m-%d')})에 존재했던 {len(stock_list)}개 종목으로 필터링되었습니다.")
        except Exception as e:
            print(f"경고: 기준일({selected_analysis_date.strftime('%Y-%m-%d')})의 시가총액 데이터를 가져오는 데 실패했습니다: {e}")
            print("최신 상장주식수 정보를 사용하여 분석을 계속합니다.")
            
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        print("거시 경제 데이터 수집에 실패하여 분석을 중단합니다."); return pd.DataFrame(), None
    all_feature_data, all_actual_dates = [], []
    stock_records = stock_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_stock = {executor.submit(fetch_and_process_ticker_data, r, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): r for r in stock_records}
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_records), desc="전 종목 피처 생성"):
            try:
                result, analysis_date = future.result()
                if result and analysis_date: all_feature_data.append(result); all_actual_dates.append(analysis_date)
            except Exception: continue
    if not all_feature_data: return pd.DataFrame(), None
    final_df = pd.DataFrame(all_feature_data)
    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)
    
    before_count = len(final_df)
    final_df.dropna(subset=['PER', 'PBR', 'ROE'], inplace=True)
    after_count = len(final_df)
    if before_count > after_count:
        print(f"✅ 필수 재무 피처(PER, PBR, ROE)가 없는 {before_count - after_count}개 종목을 분석에서 제외했습니다. (최종 {after_count}개)")

    if final_df.empty:
        print("필수 재무 피처를 가진 종목이 없어 분석을 중단합니다."); return pd.DataFrame(), None
        
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0])
    
    print("모든 피처 데이터 생성 완료!")
    return final_df, actual_analysis_date_final