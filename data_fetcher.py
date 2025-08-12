import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
import time

def fetch_stock_list(retries=3, delay=2):
    """
    KRX에서 KOSPI와 KOSDAQ의 모든 종목 티커와 이름을 가져옵니다.
    재시도 로직을 포함하여 일시적인 네트워크 오류에 대응합니다.
    """
    print("KOSPI 및 KOSDAQ 전 종목 리스트를 수집합니다...")
    try:
        latest_business_day = stock.get_nearest_business_day_in_a_week()
        
        tickers_kospi = []
        tickers_kosdaq = []

        # KOSPI 티커 조회 (재시도 포함)
        for i in range(retries):
            try:
                tickers_kospi = stock.get_market_ticker_list(latest_business_day, market='KOSPI')
                print(f"KOSPI 티커 {len(tickers_kospi)}개 수집 성공.")
                break
            except Exception as e:
                print(f"경고: KOSPI 티커 리스트 조회 실패 (시도 {i+1}/{retries}): {e}")
                if i < retries - 1:
                    time.sleep(delay)
        
        # KOSDAQ 티커 조회 (재시도 포함)
        for i in range(retries):
            try:
                tickers_kosdaq = stock.get_market_ticker_list(latest_business_day, market='KOSDAQ')
                print(f"KOSDAQ 티커 {len(tickers_kosdaq)}개 수집 성공.")
                break
            except Exception as e:
                print(f"경고: KOSDAQ 티커 리스트 조회 실패 (시도 {i+1}/{retries}): {e}")
                if i < retries - 1:
                    time.sleep(delay)

        if not tickers_kospi and not tickers_kosdaq:
            print("경고: KOSPI 및 KOSDAQ에서 가져온 티커가 없습니다.")
            return pd.DataFrame() # 빈 데이터프레임 반환

        all_tickers = tickers_kospi + tickers_kosdaq
        
        if not all_tickers:
            print("경고: KOSPI 및 KOSDAQ에서 가져온 티커가 없습니다.")
            return pd.DataFrame()

        # 3. 각 티커에 대해 종목명 조회 (오류 발생 시 건너뛰기)
        stock_names = []
        valid_tickers = []
        for ticker in all_tickers:
            try:
                name = stock.get_market_ticker_name(ticker)
                stock_names.append(name)
                valid_tickers.append(ticker)
            except Exception as e:
                print(f"경고: 종목명 조회 실패 ({ticker}): {e}. 이 종목은 건너뜁니다.")
        
        if not valid_tickers:
            print("경고: 유효한 종목명을 가져오지 못했습니다.")
            return pd.DataFrame()

        # 4. 데이터프레임으로 생성
        stock_list = pd.DataFrame({
            '종목코드': valid_tickers,
            '종목명': stock_names
        })
        
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list

    except Exception as e:
        print(f"Error fetching stock list from KRX: {e}. (주말/공휴일 또는 API 문제일 수 있습니다.)")
        # 에러 발생 시 미리 준비된 백업 리스트 반환
        return pd.DataFrame([
            {'종목코드': '005930', '종목명': '삼성전자'},
            {'종목코드': '373220', '종목명': 'LG에너지솔루션'},
            {'종목코드': '000660', '종목명': 'SK하이닉스'},
            {'종목코드': '207940', '종목명': '삼성바이오로직스'},
            {'종목코드': '005380', '종목명': '현대차'},
        ])

import concurrent.futures
from tqdm import tqdm

# --- 병렬 처리를 위한 Worker 함수 ---
def fetch_data_for_ticker(ticker, stock_name, latest_business_day, momentum_dates, supply_start_date, end_date, df_fund_all):
    """한 종목에 대한 모든 데이터를 수집하는 함수"""
    try:
        # 1. 재무 데이터
        if df_fund_all is not None and ticker in df_fund_all.index:
            fund_data = df_fund_all.loc[ticker]
        else:
            fund_data = stock.get_market_fundamental_by_ticker(latest_business_day, market="ALL").loc[ticker]
        
        per = fund_data.get('PER', np.nan) if fund_data.get('PER', 0) != 0 else np.nan
        pbr = fund_data.get('PBR', np.nan) if fund_data.get('PBR', 0) != 0 else np.nan
        roe = fund_data.get('ROE', np.nan) if fund_data.get('ROE', 0) != 0 else np.nan

        # 2. 수급 데이터
        df_trading = stock.get_market_trading_value_by_date(supply_start_date, end_date, ticker)
        df_trading_5d = df_trading.sort_index().tail(5)
        supply_inst = df_trading_5d['기관합계'].sum() / 1_0000_0000
        supply_for = df_trading_5d['외국인합계'].sum() / 1_0000_0000

        # 3. 모멘텀 및 변동성 데이터
        df_price = stock.get_market_ohlcv_by_date(momentum_dates['12M'], end_date, ticker)
        if not df_price.empty and len(df_price) > 61:
            price_1m_ago = df_price.iloc[-21]['종가']
            return_1m = (df_price.iloc[-1]['종가'] - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0
            
            price_3m_ago = df_price.iloc[-61]['종가']
            return_3m = (df_price.iloc[-1]['종가'] - price_3m_ago) / price_3m_ago if price_3m_ago > 0 else 0
            
            volatility_1m = df_price.iloc[-20:]['종가'].std() / df_price.iloc[-20:]['종가'].mean()
        else:
            return_1m, return_3m, volatility_1m = 0, 0, 0

        return {
            '종목코드': ticker, '종목명': stock_name, 'PER': per, 'PBR': pbr, 'ROE': roe,
            '기관순매수(억)': supply_inst, '외국인순매수(억)': supply_for,
            '수익률(1M)': return_1m, '수익률(3M)': return_3m, '변동성(1M)': volatility_1m
        }

    except Exception as e:
        # print(f"  - {stock_name}({ticker}) 데이터 수집 실패: {e}")
        return {
            '종목코드': ticker, '종목명': stock_name, 'PER': np.nan, 'PBR': np.nan, 'ROE': np.nan,
            '기관순매수(억)': 0, '외국인순매수(억)': 0,
            '수익률(1M)': 0, '수익률(3M)': 0, '변동성(1M)': 0
        }

def fetch_all_data(stock_list):
    """
    주어진 종목 리스트에 대해 pykrx를 사용하여 실제 데이터를 병렬로 수집합니다.
    """
    # --- 데이터 조회를 위한 날짜 설정 ---
    today = datetime.now()
    end_date = today.strftime('%Y%m%d')
    try:
        latest_business_day = stock.get_nearest_business_day_in_a_week(date=end_date)
    except Exception as e:
        print(f"경고: 가장 최근 영업일 조회 실패: {e}. 데이터 수집을 건너뜁니다.")
        return pd.DataFrame() # 빈 데이터프레임 반환하여 앱 크래시 방지
    
    supply_start_date = (today - timedelta(days=20)).strftime('%Y%m%d')
    momentum_dates = {
        '1M': (today - timedelta(days=31)).strftime('%Y%m%d'),
        '3M': (today - timedelta(days=92)).strftime('%Y%m%d'),
        '12M': (today - timedelta(days=366)).strftime('%Y%m%d')
    }

    # --- 모든 종목의 재무 데이터를 한 번에 가져오기 ---
    try:
        print("전체 종목 재무 데이터 수집 중...")
        df_fund_all = stock.get_market_fundamental_by_ticker(latest_business_day, market="ALL")
    except Exception as e:
        print(f"전체 재무 데이터 수집 실패: {e}. 개별 조회로 전환합니다.")
        df_fund_all = None

    # --- 병렬 처리를 사용하여 모든 종목 데이터 수집 ---
    all_data = []
    # 최적의 worker 수를 설정합니다. 너무 많으면 API 서버에 부담을 줄 수 있습니다.
    MAX_WORKERS = 16 

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 각 종목에 대한 future를 생성합니다.
        futures = {
            executor.submit(
                fetch_data_for_ticker, 
                row['종목코드'], row['종목명'], 
                latest_business_day, momentum_dates, supply_start_date, end_date, df_fund_all
            ): row for _, row in stock_list.iterrows()
        }

        # tqdm을 사용하여 진행 상황을 표시합니다.
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stock_list), desc="전 종목 데이터 수집"):
            result = future.result()
            if result:
                all_data.append(result)

    # --- 결과 데이터를 DataFrame으로 변환 ---
    if not all_data:
        print("수집된 데이터가 없습니다.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_data)
    
    # 데이터가 없는 경우(NaN)는 해당 팩터 계산에서 불리하도록 최하위 값으로 채움
    final_df.fillna({'PER': 999, 'PBR': 99, 'ROE': -99, '변동성(1M)': 99}, inplace=True)

    print("모든 실제 데이터 수집 완료!")
    return final_df