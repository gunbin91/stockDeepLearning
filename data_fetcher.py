
import pandas as pd
import numpy as np
from pykrx import stock

def fetch_stock_list(top_n=50):
    """
    KRX에서 시가총액 상위 N개 종목의 티커와 이름을 가져옵니다.
    주말/휴일에도 안정적으로 동작하도록 로직을 수정했습니다.
    """
    try:
        # 1. 가장 최근 영업일 조회
        today = pd.Timestamp.now().strftime('%Y%m%d')
        latest_business_day = stock.get_nearest_business_day_in_a_week(date=today)

        # 2. 해당 날짜의 시가총액 순으로 티커 목록 조회
        tickers = stock.get_market_cap_by_ticker(latest_business_day).sort_values(
            by="시가총액", ascending=False
        ).head(top_n).index.tolist()

        # 3. 각 티커에 대해 종목명 조회
        stock_names = [stock.get_market_ticker_name(ticker) for ticker in tickers]
        
        # 4. 데이터프레임으로 생성
        stock_list = pd.DataFrame({
            '종목코드': tickers,
            '종목명': stock_names
        })
        
        return stock_list

    except Exception as e:
        print(f"Error fetching stock list from KRX: {e}")
        # 에러 발생 시 미리 준비된 백업 리스트 반환
        return pd.DataFrame([
            {'종목코드': '005930', '종목명': '삼성전자'},
            {'종목코드': '373220', '종목명': 'LG에너지솔루션'},
            {'종목코드': '000660', '종목명': 'SK하이닉스'},
            {'종목코드': '207940', '종목명': '삼성바이오로직스'},
            {'종목코드': '005380', '종목명': '현대차'},
        ])


import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
import time

def fetch_stock_list(top_n=50):
    """
    KRX에서 시가총액 상위 N개 종목의 티커와 이름을 가져옵니다.
    주말/휴일에도 안정적으로 동작하도록 로직을 수정했습니다.
    """
    try:
        # 1. 가장 최근 영업일 조회
        today = datetime.now().strftime('%Y%m%d')
        latest_business_day = stock.get_nearest_business_day_in_a_week(date=today)

        # 2. 해당 날짜의 시가총액 순으로 티커 목록 조회
        tickers = stock.get_market_cap_by_ticker(latest_business_day).sort_values(
            by="시가총액", ascending=False
        ).head(top_n).index.tolist()

        # 3. 각 티커에 대해 종목명 조회
        stock_names = [stock.get_market_ticker_name(ticker) for ticker in tickers]
        
        # 4. 데이터프레임으로 생성
        stock_list = pd.DataFrame({
            '종목코드': tickers,
            '종목명': stock_names
        })
        
        return stock_list

    except Exception as e:
        print(f"Error fetching stock list from KRX: {e}")
        # 에러 발생 시 미리 준비된 백업 리스트 반환
        return pd.DataFrame([
            {'종목코드': '005930', '종목명': '삼성전자'},
            {'종목코드': '373220', '종목명': 'LG에너지솔루션'},
            {'종목코드': '000660', '종목명': 'SK하이닉스'},
            {'종목코드': '207940', '종목명': '삼성바이오로직스'},
            {'종목코드': '005380', '종목명': '현대차'},
        ])

def fetch_all_data(stock_list):
    """
    주어진 종목 리스트에 대해 pykrx를 사용하여 실제 데이터를 수집합니다.
    모멘텀 계산을 위한 과거 수익률 데이터를 포함합니다.
    [주의] 각 종목별로 API를 호출하므로 시간이 다소 소요됩니다. (약 2~3분)
    """
    final_df = stock_list.copy()
    
    # --- 데이터 조회를 위한 날짜 설정 ---
    today = datetime.now()
    end_date = today.strftime('%Y%m%d')
    latest_business_day = stock.get_nearest_business_day_in_a_week(date=end_date)
    
    # 수급 데이터용 날짜 (최근 5일)
    supply_start_date = (today - timedelta(days=10)).strftime('%Y%m%d')

    # 모멘텀 데이터용 날짜
    momentum_dates = {
        '3M': (today - timedelta(days=90)).strftime('%Y%m%d'),
        '6M': (today - timedelta(days=180)).strftime('%Y%m%d'),
        '12M': (today - timedelta(days=365)).strftime('%Y%m%d')
    }

    # --- 각 종목에 대한 데이터를 저장할 리스트 ---
    per_list, pbr_list, roe_list = [], [], []
    supply_inst_list, supply_for_list = [], []
    short_ratio_list = []
    return_3m_list, return_6m_list, return_12m_list = [], [], []

    for i, row in final_df.iterrows():
        ticker = row['종목코드']
        print(f"({i+1}/{len(final_df)}) {row['종목명']}({ticker}) 데이터 수집 중...")
        
        # API 과부하 방지를 위한 약간의 딜레이
        time.sleep(0.2)

        try:
            # 1. 재무 데이터 (PER, PBR, ROE)
            df_fund = stock.get_market_fundamental_by_ticker(latest_business_day, market="ALL")
            fund_data = df_fund.loc[ticker]
            per_list.append(fund_data['PER'] if fund_data['PER'] != 0 else np.nan)
            pbr_list.append(fund_data['PBR'] if fund_data['PBR'] != 0 else np.nan)
            roe_list.append(fund_data['ROE'] if 'ROE' in fund_data and fund_data['ROE'] != 0 else np.nan)

            # 2. 투자자 수급 데이터 (최근 5일 합산)
            df_trading = stock.get_market_trading_value_by_date(supply_start_date, end_date, ticker)
            supply_inst = df_trading['기관합계'].sum() / 1_0000_0000 # 억원 단위
            supply_for = df_trading['외국인합계'].sum() / 1_0000_0000 # 억원 단위
            supply_inst_list.append(supply_inst)
            supply_for_list.append(supply_for)

            # 3. 공매도 데이터 (가장 최근일)
            df_short = stock.get_shorting_status_by_date(end_date, end_date, ticker)
            short_ratio_list.append(df_short['비중'].iloc[0] if not df_short.empty else 0.0)

            # 4. 모멘텀 데이터 (3, 6, 12개월 수익률)
            returns = {}
            for name, start_date in momentum_dates.items():
                df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                if not df_price.empty and df_price.iloc[0]['종가'] > 0:
                    returns[name] = (df_price.iloc[-1]['종가'] - df_price.iloc[0]['종가']) / df_price.iloc[0]['종가']
                else:
                    returns[name] = 0
            return_3m_list.append(returns.get('3M', 0))
            return_6m_list.append(returns.get('6M', 0))
            return_12m_list.append(returns.get('12M', 0))

        except Exception as e:
            print(f"  - {row['종목명']}({ticker}) 데이터 수집 실패: {e}")
            # 실패 시 모든 데이터에 기본값 추가
            per_list.append(np.nan); pbr_list.append(np.nan); roe_list.append(np.nan)
            supply_inst_list.append(0); supply_for_list.append(0); short_ratio_list.append(0.0)
            return_3m_list.append(0); return_6m_list.append(0); return_12m_list.append(0)

    # --- 수집된 데이터를 DataFrame에 추가 ---
    final_df['PER'] = per_list
    final_df['PBR'] = pbr_list
    final_df['ROE'] = roe_list
    final_df['기관순매수(억)'] = supply_inst_list
    final_df['외국인순매수(억)'] = supply_for_list
    final_df['공매도비중(%)'] = short_ratio_list
    final_df['수익률(3M)'] = return_3m_list
    final_df['수익률(6M)'] = return_6m_list
    final_df['수익률(12M)'] = return_12m_list
    
    # 데이터가 없는 경우(NaN)는 해당 팩터 계산에서 불리하도록 최하위 값으로 채움
    final_df.fillna({'PER': 999, 'PBR': 99, 'ROE': -99}, inplace=True)

    print("모든 실제 데이터 수집 완료!")
    return final_df

