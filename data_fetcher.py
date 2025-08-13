import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
import time
import json
import os
import pandas_ta as ta # pandas-ta 임포트
import concurrent.futures
from tqdm import tqdm

# --- (fetch_stock_list 및 캐시 관련 함수는 기존과 동일하게 유지) ---

STOCK_LIST_CACHE_FILE = 'stock_list_cache.json'
CACHE_EXPIRATION_HOURS = 24

def _load_stock_list_from_cache():
    if os.path.exists(STOCK_LIST_CACHE_FILE):
        try:
            with open(STOCK_LIST_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRATION_HOURS):
                print(f"캐시에서 종목 리스트 로드 (업데이트: {cached_time.strftime('%Y-%m-%d %H:%M:%S')})")
                return pd.DataFrame(cache_data['stock_list'])
            else:
                print("캐시된 종목 리스트가 만료되었습니다. 새로 가져옵니다.")
        except Exception as e:
            print(f"캐시 로드 중 오류 발생: {e}. 새로 가져옵니다.")
    return None

def _save_stock_list_to_cache(stock_list_df):
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'stock_list': stock_list_df.to_dict(orient='records')
        }
        with open(STOCK_LIST_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
        print(f"종목 리스트를 캐시 파일('{STOCK_LIST_CACHE_FILE}')에 저장했습니다.")
    except Exception as e:
        print(f"캐시 저장 중 오류 발생: {e}")

def fetch_stock_list(retries=3, delay=2):
    cached_stock_list = _load_stock_list_from_cache()
    if cached_stock_list is not None:
        return cached_stock_list
    print("KOSPI 및 KOSDAQ 전 종목 리스트를 수집합니다...")
    try:
        latest_business_day = stock.get_nearest_business_day_in_a_week()
        tickers_kospi = stock.get_market_ticker_list(latest_business_day, market='KOSPI')
        tickers_kosdaq = stock.get_market_ticker_list(latest_business_day, market='KOSDAQ')
        all_tickers = list(set(tickers_kospi + tickers_kosdaq))
        stock_names = [stock.get_market_ticker_name(t) for t in all_tickers]
        stock_list = pd.DataFrame({'종목코드': all_tickers, '종목명': stock_names})
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        _save_stock_list_to_cache(stock_list)
        return stock_list
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return pd.DataFrame([
            {'종목코드': '005930', '종목명': '삼성전자'},
            {'종목코드': '000660', '종목명': 'SK하이닉스'},
        ])


# <<< 개선점: ML 피처 생성 로직을 data_fetcher로 통합 >>>
def fetch_and_process_ticker_data(ticker, stock_name, start_date, end_date):
    """ 한 종목의 데이터를 수집하고 train_model.py와 동일한 방식으로 피처를 생성합니다. """
    try:
        # 1. 가격 데이터 (OHLCV) 및 기본 데이터 수집
        df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if len(df_price) < 61:  # 최소 3개월 데이터 확보
            return None
            
        df_fundamental = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
        
        # 데이터 병합
        df = pd.merge(df_price, df_fundamental[['BPS', 'PER', 'PBR', 'EPS', 'DIV']], left_index=True, right_index=True, how='left')
        df.ffill(inplace=True)

        # 2. 피처 엔지니어링 (train_model.py와 동일한 로직)
        # 수익률 및 변동성
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()

        # 거래량 관련 피처
        df['거래량변화율'] = df['거래량'].pct_change(periods=20)
        df['거래량MA5'] = df['거래량'].rolling(window=5).mean()
        df['거래량MA20'] = df['거래량'].rolling(window=20).mean()

        # 기술적 지표 (pandas_ta 라이브러리 활용)
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        df.ta.sma(close='종가', length=5, append=True)
        df.ta.sma(close='종가', length=20, append=True)
        df.ta.sma(close='종가', length=60, append=True)
        df.ta.bbands(close='종가', length=20, std=2, append=True)
        
        # ROE (PBR과 PER로 계산)
        df['ROE'] = df['PBR'] / df['PER']
        
        # 수급 데이터 (최근 5일)
        supply_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=20)).strftime('%Y%m%d')
        df_trading = stock.get_market_trading_value_by_date(supply_start_date, end_date, ticker)
        df_trading_5d = df_trading.sort_index().tail(5)
        
        # 3. 최종적으로 필요한 최신 데이터만 선택
        latest_data = df.iloc[-1].to_dict()
        latest_data['종목코드'] = ticker
        latest_data['종목명'] = stock_name
        latest_data['현재가'] = df.iloc[-1]['종가']
        latest_data['시가총액'] = stock.get_market_cap_by_ticker(end_date).loc[ticker, '시가총액'] / 1_0000_0000 # 억원 단위
        
        # 수급 정보 추가
        latest_data['기관순매수(억)'] = df_trading_5d['기관합계'].sum() / 1_0000_0000
        latest_data['외국인순매수(억)'] = df_trading_5d['외국인합계'].sum() / 1_0000_0000

        # OHLCV 데이터 자체도 반환 (추후 DL 모델 등을 위해)
        return latest_data, df
        
    except Exception:
        # print(f"  - {stock_name}({ticker}) 데이터 수집/처리 실패: {e}")
        return None

def fetch_all_data(stock_list):
    """ 주어진 종목 리스트에 대해 모든 피처를 병렬로 계산합니다. """
    # --- 데이터 조회를 위한 날짜 설정 ---
    try:
        end_date = stock.get_nearest_business_day_in_a_week()
    except Exception:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')

    # --- 병렬 처리를 사용하여 모든 종목 데이터 수집 및 가공 ---
    all_feature_data = []
    all_price_data = {} # 각 종목의 전체 시계열 데이터를 저장
    MAX_WORKERS = 16 

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                fetch_and_process_ticker_data, 
                row['종목코드'], row['종목명'], 
                start_date, end_date
            ): row for _, row in stock_list.iterrows()
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stock_list), desc="전 종목 피처 생성"):
            result = future.result()
            if result:
                latest_data, df_price = result
                all_feature_data.append(latest_data)
                all_price_data[latest_data['종목코드']] = df_price

    # --- 결과 데이터를 DataFrame으로 변환 ---
    if not all_feature_data:
        print("수집된 데이터가 없습니다.")
        return pd.DataFrame(), {}

    final_df = pd.DataFrame(all_feature_data)
    final_df.sort_values(by='종목코드', inplace=True)
    
    # PER, PBR이 0 또는 음수일 경우 예측에 방해가 되므로 np.nan으로 변경
    final_df['PER'] = final_df['PER'].apply(lambda x: x if x > 0 else np.nan)
    final_df['PBR'] = final_df['PBR'].apply(lambda x: x if x > 0 else np.nan)

    # 무한대 값 NaN으로 변경
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 결측치 처리: ML 모델이 NaN 입력을 받지 않도록 중앙값(median)으로 채움
    final_df.fillna(final_df.median(numeric_only=True), inplace=True)
    # 그래도 남은 NaN이 있다면 0으로 채움
    final_df.fillna(0, inplace=True)

    print("모든 피처 데이터 생성 완료!")
    # app.py의 기존 로직과 호환성을 위해 두 개의 df를 반환하는 것처럼 보이지만, 사실상 동일한 df
    # ml_feature_df, factor_base_df, all_price_data
    return final_df, final_df, all_price_data