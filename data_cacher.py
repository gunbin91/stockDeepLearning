# data_cacher.py

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import requests
import time
from datetime import datetime, timedelta
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm
import os
import json
import config

from scoring import calculate_factor_scores

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
PIT_FS_PATH = os.path.join(PROJECT_ROOT, 'data', 'financial_data_pit.parquet')

CACHE_END_DATE = datetime(datetime.now().year - 1, 12, 31).strftime('%Y-%m-%d')
CACHE_FILENAME = f"historical_data_up_to_{CACHE_END_DATE.replace('-', '')}.parquet"
CACHE_FILE_PATH = os.path.join(CACHE_DIR, CACHE_FILENAME)

try:
    pit_fs_df = pd.read_parquet(PIT_FS_PATH)
    pit_fs_df['공시일'] = pd.to_datetime(pit_fs_df['공시일'])
    pit_fs_df.sort_values('공시일', inplace=True)
    print(f"✅ 시점(Point-in-Time) 재무 데이터베이스 로드 완료: {PIT_FS_PATH}")
except FileNotFoundError:
    print(f"!!!!!!!! [치명적 오류] 재무 데이터베이스 파일({PIT_FS_PATH})을 찾을 수 없습니다. !!!!!!!!")
    print("먼저 `scripts/build_financial_db.py`를 실행하여 데이터베이스를 생성해주세요.")
    pit_fs_df = pd.DataFrame()

def fetch_stock_list():
    try:
        df_kospi = fdr.StockListing('KOSPI'); df_kosdaq = fdr.StockListing('KOSDAQ')
        stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
        return stock_list[['종목코드', '종목명']]
    except Exception: return pd.DataFrame()

def _fetch_macro_data(start_date, end_date):
    print("거시 경제 지표 데이터 수집 중 (KOSPI, USD/KRW, VIX)...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date)[['Close']].rename(columns={'Close': 'KOSPI'})
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']].rename(columns={'Close': 'USDKRW'})
        vix = fdr.DataReader('^VIX', start_date, end_date)[['Close']].rename(columns={'Close': 'VIX'})
        macro_df = pd.merge(kospi, usdkrw, left_index=True, right_index=True, how='outer')
        macro_df = pd.merge(macro_df, vix, left_index=True, right_index=True, how='outer')
        macro_df.sort_index(inplace=True); macro_df.ffill(inplace=True); macro_df.bfill(inplace=True)
        for col in macro_df.columns:
            macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
            macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        macro_df.reset_index(inplace=True); macro_df.rename(columns={'index': 'date'}, inplace=True)
        print("✅ 거시 경제 지표 수집 및 처리 완료.")
        return macro_df
    except Exception as e:
        print(f"!!!!!!!! [치명적 오류] 거시 경제 지표 수집 실패: {e} !!!!!!!!")
        raise e

def process_single_ticker_data(stock_info, start_date, end_date, df_marcap_long, pbar_lock):
    ticker = stock_info['종목코드']
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price is None or df_price.empty or len(df_price) < 251 + 60: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['종가', '거래량']].copy(); df.sort_index(inplace=True)
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: return None
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['Date', 'Marcap']], left_index=True, right_on='Date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True); df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        
        if not pit_fs_df.empty:
            ticker_fs = pit_fs_df[pit_fs_df['종목코드'] == ticker]
            if not ticker_fs.empty:
                df = pd.merge_asof(left=df, right=ticker_fs[['공시일', '당기순이익', '자본총계']], left_index=True, right_on='공시일', direction='backward')

        # '당기순이익', '자본총계'가 없으면 애초에 데이터를 생성하지 않음
        if '당기순이익' not in df.columns or df[['당기순이익', '자본총계']].isnull().values.any():
            return None

        df['거래대금'] = df['종가'] * df['거래량']
        df['PER'] = df['시가총액'] / df['당기순이익']; df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']; df['log_mktcap'] = np.log(df['시가총액'])
        df['수익률(1W)'] = df['종가'].pct_change(5); df['수익률(2W)'] = df['종가'].pct_change(10)
        df['수익률(1M)'] = df['종가'].pct_change(20); df['수익률(3M)'] = df['종가'].pct_change(60)
        df['변동성(1M)'] = df['종가'].rolling(20).std() / df['종가'].rolling(20).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(20).mean()
        df['MA5'] = df['종가'].rolling(5).mean(); df['MA20'] = df['종가'].rolling(20).mean()
        df['단기 정배열'] = (df['MA5'] > df['MA20']).astype(int)
        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker; df.set_index('Date', inplace=True)
        df.drop(columns=['공시일'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        with pbar_lock: tqdm.write(f"⚠️ {stock_info['종목명']}({ticker}) 데이터 처리 중 오류: {e} (건너뜀)"); return None

def _fetch_and_prepare_data(start_date, end_date):
    print(f"데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    if stock_list.empty: raise ValueError("종목 리스트를 가져올 수 없습니다.")
    try:
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in month_end_dates}
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(month_end_dates), desc="시가총액 데이터 수집"):
                try:
                    date_str = future_to_date[future]; result_df = future.result()
                    if not result_df.empty: result_df['Date'] = pd.to_datetime(date_str); marcap_dfs.append(result_df)
                except Exception: continue
        if not marcap_dfs: raise Exception("수집된 시가총액 데이터가 없습니다.")
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True)
    except Exception as e:
        raise ConnectionError(f"과거 시가총액 데이터 수집 실패: {e}")
        
    all_data = []; stock_records = stock_list.to_dict('records')
    with tqdm(total=len(stock_records), desc="개별 종목 피처 데이터 생성") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_stock = {executor.submit(process_single_ticker_data, row, start_date, end_date, df_marcap_long, pbar.get_lock()): row for row in stock_records}
            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    result_df = future.result()
                    if result_df is not None: all_data.append(result_df)
                except Exception: pass
                pbar.update(1)

    if not all_data: raise ValueError("처리된 데이터가 없습니다.")
    raw_feature_df = pd.concat(all_data).reset_index()
    raw_feature_df.rename(columns={'Date': 'date'}, inplace=True)
    raw_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # <<< ✨ 핵심 수정: data_fetcher.py와 일관성을 위해 필터링 로직 추가 ✨ >>>
    # process_single_ticker_data에서 이미 1차 필터링되었지만, 여기서 한 번 더 확인합니다.
    before_count = len(raw_feature_df)
    raw_feature_df.dropna(subset=['PER', 'PBR', 'ROE'], inplace=True)
    after_count = len(raw_feature_df)
    if before_count > after_count:
        print(f"✅ 필수 재무 피처가 없는 {before_count - after_count}개 레코드를 필터링했습니다.")
    
    raw_feature_df.dropna(subset=['date', '종목코드'], inplace=True)
    raw_feature_df['date'] = pd.to_datetime(raw_feature_df['date'])
    macro_df = _fetch_macro_data(start_date, end_date)
    raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')
    raw_feature_df.sort_values(by=['date', '종목코드'], inplace=True)
    print("일별 팩터 점수 계산 중..."); num_groups = len(raw_feature_df.groupby('date'))
    scored_data_list = []
    for date, daily_data in tqdm(raw_feature_df.groupby('date'), total=num_groups):
        daily_scored_data = calculate_factor_scores(daily_data.copy()); daily_scored_data['date'] = date
        scored_data_list.append(daily_scored_data)
    if not scored_data_list: return raw_feature_df
    all_scored_df = pd.concat(scored_data_list)
    score_cols_to_merge = ['종목코드', 'date'] + [col for col in all_scored_df.columns if '_score' in col]
    final_df = pd.merge(raw_feature_df, all_scored_df[score_cols_to_merge], on=['date', '종목코드'], how='left')
    final_df.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)
    return final_df

def get_preprocessed_data(start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE_PATH):
        print(f"✅ 캐시된 과거 데이터 로딩: {CACHE_FILE_PATH}"); historical_df = pd.read_parquet(CACHE_FILE_PATH)
    else:
        print(f"⚠️ 캐시 파일 없음. {CACHE_END_DATE}까지의 과거 데이터를 생성합니다 (시간이 소요됩니다)...")
        cache_start_date = (datetime.strptime(CACHE_END_DATE, '%Y-%m-%d') - timedelta(days=365*5)).strftime('%Y-%m-%d')
        historical_df = _fetch_and_prepare_data(cache_start_date, CACHE_END_DATE)
        historical_df.to_parquet(CACHE_FILE_PATH)
        print(f"✅ 과거 데이터 캐시 저장 완료: {CACHE_FILE_PATH}")

    fresh_start_date = (datetime.strptime(CACHE_END_DATE, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    if datetime.strptime(end_date, '%Y-%m-%d').date() > datetime.strptime(CACHE_END_DATE, '%Y-%m-%d').date():
        print(f"\n올해 최신 데이터 수집 중 ({fresh_start_date} ~ {end_date})...")
        fresh_start_date_with_warmup = (pd.to_datetime(fresh_start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        fresh_df = _fetch_and_prepare_data(fresh_start_date_with_warmup, end_date)
        if not fresh_df.empty: fresh_df = fresh_df[fresh_df['date'] >= pd.to_datetime(fresh_start_date)].copy()
    else:
        fresh_df = pd.DataFrame()

    combined_df = pd.concat([historical_df, fresh_df], ignore_index=True)
    final_df = combined_df[(combined_df['date'] >= pd.to_datetime(start_date)) & (combined_df['date'] <= pd.to_datetime(end_date))].copy()
    print('✅ 모든 데이터 준비 완료.'); return final_df