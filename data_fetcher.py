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

def fetch_stock_list():
    """
    KOSPI와 KOSDAQ의 전 종목 리스트를 API를 통해 수집합니다.
    API 통신 실패 시, 오류 메시지를 출력하고 빈 데이터프레임을 반환합니다.
    """
    print("KOSPI 및 KOSDAQ 전 종목 리스트를 수집합니다...")
    try:
        latest_business_day = stock.get_nearest_business_day_in_a_week()
        tickers_kospi = stock.get_market_ticker_list(latest_business_day, market='KOSPI')
        tickers_kosdaq = stock.get_market_ticker_list(latest_business_day, market='KOSDAQ')
        all_tickers = list(set(tickers_kospi + tickers_kosdaq))
        stock_names = [stock.get_market_ticker_name(t) for t in all_tickers]
        stock_list = pd.DataFrame({'종목코드': all_tickers, '종목명': stock_names})
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"API 통신 실패: 종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(columns=['종목코드', '종목명'])


# <<< 개선점: ML 피처 생성 로직을 data_fetcher로 통합 >>>
def fetch_and_process_ticker_data(ticker, stock_name, start_date, end_date):
    """ 한 종목의 데이터를 수집하고 train_model.py와 동일한 방식으로 피처를 생성합니다. (오류 처리 강화) """
    try:
        # 1. 가격 데이터 (OHLCV) 수집 및 검증
        df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        required_price_cols = ['종가', '거래량']
        if df_price.empty or not all(col in df_price.columns for col in required_price_cols) or len(df_price) < 61:
            return None # 필수 가격 정보가 없거나 데이터 기간이 짧으면 건너뜀

        # 2. 재무 데이터 수집 및 병합 (오류에 강건하게 처리)
        df_fundamental = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
        
        df = df_price.copy() # 가격 데이터를 기반으로 기본 DataFrame 생성

        fundamental_cols = ['BPS', 'PER', 'PBR', 'EPS', 'DIV']
        
        if not df_fundamental.empty:
            existing_fundamental_cols = [col for col in fundamental_cols if col in df_fundamental.columns]
            if existing_fundamental_cols:
                df = pd.merge(df, df_fundamental[existing_fundamental_cols], left_index=True, right_index=True, how='left')

        for col in fundamental_cols:
            if col not in df.columns:
                df[col] = 0
        
        df.ffill(inplace=True)

        # 3. 피처 엔지니어링
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래량변화율'] = df['거래량'].pct_change(periods=20)
        df['거래량MA5'] = df['거래량'].rolling(window=5).mean()
        df['거래량MA20'] = df['거래량'].rolling(window=20).mean()
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        df.ta.sma(close='종가', length=5, append=True)
        df.ta.sma(close='종가', length=20, append=True)
        df.ta.sma(close='종가', length=60, append=True)
        df.ta.bbands(close='종가', length=20, std=2, append=True)
        
        df['ROE'] = np.where(df['PER'] != 0, df['PBR'] / df['PER'], 0)
        
        # 4. 수급 데이터
        supply_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=20)).strftime('%Y%m%d')
        df_trading = stock.get_market_trading_value_by_date(supply_start_date, end_date, ticker)
        df_trading_5d = df_trading.sort_index().tail(5)
        
        # 5. 최종 데이터 선택
        latest_data = df.iloc[-1].to_dict()
        latest_data['종목코드'] = ticker
        latest_data['종목명'] = stock_name
        latest_data['현재가'] = df.iloc[-1]['종가']
        latest_data['시가총액'] = stock.get_market_cap_by_ticker(end_date).loc[ticker, '시가총액'] / 1_0000_0000

        latest_data['기관순매수(억)'] = df_trading_5d['기관합계'].sum() / 1_0000_0000
        latest_data['외국인순매수(억)'] = df_trading_5d['외국인합계'].sum() / 1_0000_0000

        return latest_data, df
        
    except Exception as e:
        # print(f"  - {stock_name}({ticker}) 최종 처리 실패: {e}") # 최종 단계에서 오류 발생 시 건너뛰기
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
        return pd.DataFrame(), pd.DataFrame()

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
    return final_df, final_df