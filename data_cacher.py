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

# 내부 모듈 임포트
from scoring import calculate_factor_scores

# --- 설정 변수 ---
# DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_END_DATE = datetime(datetime.now().year - 1, 12, 31).strftime('%Y-%m-%d')
CACHE_FILENAME = f"historical_data_up_to_{CACHE_END_DATE.replace('-', '')}.parquet"
CACHE_FILE_PATH = os.path.join(CACHE_DIR, CACHE_FILENAME)

def get_financial_data_for_training_http(corp_codes, start_year, end_year):
    # 이 함수는 이전과 동일, 문제가 없습니다.
    if config.DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요": return {}
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    all_fs_data = {}
    current_year = datetime.now().year
    for year in range(start_year, end_year + 1):
        fs_cache_path = os.path.join(CACHE_DIR, f"fs_data_{year}.json")
        
        if year < current_year and os.path.exists(fs_cache_path):
            print(f"✅ 캐시된 재무 데이터 로딩: {fs_cache_path}")
            with open(fs_cache_path, 'r', encoding='utf-8') as f:
                all_fs_data[year] = json.load(f)
            continue

        if year < current_year:
            print(f"⚠️ {year}년 재무 데이터 캐시 없음. API를 통해 수집합니다.")
        else:
            print(f" 현재 연도({year}) 재무 데이터는 항상 새로 수집합니다.")

        year_fs_data = []
        for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터 수집"):
            corp_code_chunk = corp_codes[i:i+100]
            corp_code_str = ','.join(corp_code_chunk)
            
            url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
            params = { 'crtfc_key': config.DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': str(year), 'reprt_code': '11011' }
            
            try:
                res = requests.get(url, params=params)
                if res.status_code == 200 and res.json().get('status') == '000':
                    year_fs_data.extend(res.json()['list'])
            except Exception:
                continue
            time.sleep(0.2)
        
        if year_fs_data:
            df = pd.DataFrame(year_fs_data)
            df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'].str.replace(',', ''), errors='coerce')
            df_pivot = df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount')
            
            year_data_dict = df_pivot.where(pd.notnull(df_pivot), None).to_dict('index')
            all_fs_data[year] = year_data_dict
            
            if year < current_year:
                with open(fs_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(year_data_dict, f, ensure_ascii=False, indent=4)
                print(f"✅ {year}년 재무 데이터 캐시 저장 완료: {fs_cache_path}")
    return all_fs_data

def fetch_stock_list():
    # 이 함수는 이전과 동일, 문제가 없습니다.
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
        return stock_list[['종목코드', '종목명']]
    except Exception: return pd.DataFrame()

# <<< 핵심 수정: 훨씬 더 견고하게 만든 거시 경제 지표 수집 함수 >>>
def _fetch_macro_data(start_date, end_date):
    """지정된 기간의 주요 거시 경제 지표를 안정적으로 수집하고 변화율을 계산합니다."""
    print("거시 경제 지표 데이터 수집 중 (KOSPI, USD/KRW, VIX)...")
    try:
        # 1. 각 지표를 개별적으로 수집
        kospi = fdr.DataReader('KS11', start_date, end_date)[['Close']].rename(columns={'Close': 'KOSPI'})
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']].rename(columns={'Close': 'USDKRW'})
        vix = fdr.DataReader('^VIX', start_date, end_date)[['Close']].rename(columns={'Close': 'VIX'})

        # 2. Outer Join으로 안전하게 병합 (인덱스 기준)
        macro_df = pd.merge(kospi, usdkrw, left_index=True, right_index=True, how='outer')
        macro_df = pd.merge(macro_df, vix, left_index=True, right_index=True, how='outer')
        
        # 날짜 순으로 정렬
        macro_df.sort_index(inplace=True)

        # 3. 휴장일 등으로 인한 결측치를 직전 값으로 채우기 (Forward Fill)
        macro_df.ffill(inplace=True)
        
        # 맨 처음에 NaN이 있을 경우를 대비해 맨 뒤 값으로 채우기 (Backward Fill)
        macro_df.bfill(inplace=True)

        # 4. 모든 값이 채워진 후, 안전하게 변화율 계산
        for col in macro_df.columns:
            macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
            macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        
        # 5. 날짜 인덱스를 'date' 컬럼으로 변환
        macro_df.reset_index(inplace=True)
        macro_df.rename(columns={'index': 'date'}, inplace=True)
        
        print("✅ 거시 경제 지표 수집 및 처리 완료.")
        return macro_df
    except Exception as e:
        print(f"!!!!!!!! [치명적 오류] 거시 경제 지표 수집 실패. 프로세스를 중단합니다. (에러: {e}) !!!!!!!!")
        # 근본적인 문제일 수 있으므로, 프로그램을 중단시키는 것이 더 안전합니다.
        raise e

def process_single_ticker_data(stock_info, start_date, end_date, all_fs_data, df_marcap_long):
    # 이 함수는 이전과 동일, 문제가 없습니다.
    ticker = stock_info['종목코드']
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 251 + 60: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: return None
        df.sort_index(inplace=True)
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['Date', 'Marcap']], left_index=True, right_on='Date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        df['거래대금'] = df['종가'] * df['거래량']
        df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        
        df.sort_index(inplace=True)
        df[['당기순이익', '자본총계']] = df[['당기순이익', '자본총계']].ffill()

        if df[['당기순이익', '자본총계']].isnull().values.any(): return None
        
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        df['log_mktcap'] = np.log(df['시가총액'])
        df['수익률(1W)'] = df['종가'].pct_change(periods=5)
        df['수익률(2W)'] = df['종가'].pct_change(periods=10)
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(window=20).mean()
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['단기 정배열'] = (df['MA5'] > df['MA20']).astype(int)
        df['52주_최고가'] = df['종가'].rolling(window=250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        df.set_index('Date', inplace=True)
        return df
    except Exception: return None

def _fetch_and_prepare_data(start_date, end_date):
    # 이 함수는 이전과 동일, 문제가 없습니다.
    print(f"데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    if stock_list.empty: raise ValueError("종목 리스트를 가져올 수 없습니다.")
    
    try:
        df_corp_map = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'corp_code_map.csv'), dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError: raise FileNotFoundError(f"{os.path.join(PROJECT_ROOT, 'data', 'corp_code_map.csv')} 파일이 없습니다.")
    
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()
    
    all_fs_data = get_financial_data_for_training_http(corp_codes, int(start_date[:4]) - 1, int(end_date[:4]))
    if not all_fs_data: raise ValueError("재무 데이터를 가져올 수 없습니다.")
    
    try:
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in month_end_dates}
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(month_end_dates), desc="시가총액 데이터 수집"):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty:
                        result_df['Date'] = pd.to_datetime(date_str)
                        marcap_dfs.append(result_df)
                except Exception: continue
        if not marcap_dfs: raise Exception("수집된 시가총액 데이터가 없습니다.")
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True)
    except Exception as e:
        raise ConnectionError(f"과거 시가총액 데이터 수집 실패: {e}")
        
    all_data = []
    stock_records = target_stocks.to_dict('records')
    for row in tqdm(stock_records, desc="개별 종목 피처 데이터 생성"):
        result_df = process_single_ticker_data(row, start_date, end_date, all_fs_data, df_marcap_long)
        if result_df is not None:
            all_data.append(result_df)
            
    if not all_data: raise ValueError("처리된 데이터가 없습니다.")
    raw_feature_df = pd.concat(all_data).reset_index()
    raw_feature_df.rename(columns={'Date': 'date'}, inplace=True)
    raw_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_feature_df.dropna(subset=['date', '종목코드'], inplace=True)
    raw_feature_df['date'] = pd.to_datetime(raw_feature_df['date'])
    
    macro_df = _fetch_macro_data(start_date, end_date)

    raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')

    raw_feature_df.sort_values(by=['date', '종목코드'], inplace=True)

    print("일별 팩터 점수 계산 중...")
    scored_data_list = []
    for date, daily_data in tqdm(raw_feature_df.groupby('date')):
        daily_scored_data = calculate_factor_scores(daily_data.copy())
        daily_scored_data['date'] = date
        scored_data_list.append(daily_scored_data)
    
    if not scored_data_list:
        return raw_feature_df
        
    all_scored_df = pd.concat(scored_data_list)
    score_cols_to_merge = ['종목코드', 'date'] + [col for col in all_scored_df.columns if '_score' in col]
    
    final_df = pd.merge(raw_feature_df, all_scored_df[score_cols_to_merge], on=['date', '종목코드'], how='left')
    final_df.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)

    return final_df

def get_preprocessed_data(start_date, end_date):
    # 이 함수는 이전과 동일, 문제가 없습니다.
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(CACHE_FILE_PATH):
        print(f"✅ 캐시된 과거 데이터 로딩: {CACHE_FILE_PATH}")
        historical_df = pd.read_parquet(CACHE_FILE_PATH)
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
        if not fresh_df.empty:
            fresh_df = fresh_df[fresh_df['date'] >= pd.to_datetime(fresh_start_date)].copy()
    else:
        fresh_df = pd.DataFrame()

    combined_df = pd.concat([historical_df, fresh_df], ignore_index=True)
    
    final_df = combined_df[(combined_df['date'] >= pd.to_datetime(start_date)) & (combined_df['date'] <= pd.to_datetime(end_date))].copy()
    print('✅ 모든 데이터 준비 완료.')
    return final_df
