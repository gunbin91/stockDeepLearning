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

DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_latest_annual_fs_http(stock_list):
    # 이 함수는 변경할 필요가 없습니다.
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키가 설정되지 않아 재무 데이터를 수집할 수 없습니다.")
        return pd.DataFrame()
        
    try:
        corp_code_map_path = os.path.join(PROJECT_ROOT, 'data', 'corp_code_map.csv')
        df_corp_map = pd.read_csv(corp_code_map_path, dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError:
        print(f"[오류] '{corp_code_map_path}' 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
        
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()
    
    year = str(datetime.now().year - 1)
    print(f"HTTP 통신을 통해 {year}년 재무 데이터 수집을 시작합니다.")
    
    all_fs_data = []
    
    for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터 수집"):
        corp_code_chunk = corp_codes[i:i+100]
        corp_code_str = ','.join(corp_code_chunk)
        
        url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
        params = {
            'crtfc_key': config.DART_API_KEY,
            'corp_code': corp_code_str,
            'bsns_year': year,
            'reprt_code': '11011',
        }
        
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            
            if data.get('status') == '000':
                all_fs_data.extend(data['list'])
        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생 (건너뜀): {e}")
            continue
        except ValueError:
            print(f"API 응답 JSON 파싱 오류 (건너뜀)")
            continue
            
        time.sleep(0.2)

    if not all_fs_data:
        return pd.DataFrame()
        
    fs_df = pd.DataFrame(all_fs_data)
    
    required_accounts = ['당기순이익', '자본총계', '유동자산', '유동부채']
    fs_df = fs_df[fs_df['account_nm'].isin(required_accounts)]
    
    fs_df['thstrm_amount'] = pd.to_numeric(fs_df['thstrm_amount'].str.replace(',', ''), errors='coerce')

    fs_pivot = fs_df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount').reset_index()
    fs_pivot.rename(columns={'stock_code':'종목코드'}, inplace=True)
    
    print(f"✅ {year}년 재무 데이터 수집 및 처리 완료: {len(fs_pivot)}개 기업")
    return fs_pivot

def fetch_stock_list():
    # 이 함수는 변경할 필요가 없습니다.
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보를 수집합니다 (KRX-MARCAP)...")
    try:
        df_marcap = fdr.StockListing('KRX-MARCAP')
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        
        stock_list = stock_list[stock_list['상장주식수'] > 0]

        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}")
        return pd.DataFrame(columns=['종목코드', '종목명', '상장주식수'])

# <<< 핵심 수정 1: data_cacher와 동일한 거시 경제 지표 수집 함수 추가 >>>
def _fetch_latest_macro_data(start_date, end_date):
    print("최신 거시 경제 지표 데이터 수집 중...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date)
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)
        vix = fdr.DataReader('^VIX', start_date, end_date)

        macro_df = pd.concat([
            kospi['Close'].rename('KOSPI'),
            usdkrw['Close'].rename('USDKRW'),
            vix['Close'].rename('VIX')
        ], axis=1)
        
        macro_df.ffill(inplace=True)

        for col in macro_df.columns:
            macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
            macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        
        # 가장 마지막 날짜의 데이터(오늘의 시장 상황)만 반환
        return macro_df.iloc[-1:] 
    except Exception as e:
        print(f"최신 거시 경제 지표 수집 실패: {e}")
        return pd.DataFrame()

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        df_price = fdr.DataReader(ticker, start_date_for_fetch, end_date_for_fetch)
        if df_price.empty or len(df_price) < 251: return None, None, None # Return None for all in case of error

        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        df = df_price.copy()
        
        # Convert selected_analysis_date to Timestamp for comparison with df.index
        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)

        # --- 기준일가 (selected_analysis_date 기준) --- 
        # selected_analysis_date에 가장 가까운 거래일의 종가를 찾음
        # df.index는 datetime 인덱스이므로, selected_analysis_date와 비교하여 가장 가까운 날짜를 찾음
        reference_date_price = None
        actual_analysis_date = None
        if selected_analysis_date_ts in df.index:
            reference_date_price = df.loc[selected_analysis_date_ts]['종가']
            actual_analysis_date = selected_analysis_date_ts
        else:
            # 선택된 날짜가 휴장일인 경우, 그 이전 가장 가까운 거래일을 찾음
            # df.index는 이미 거래일만 포함하고 있으므로, selected_analysis_date보다 작거나 같은 날짜 중 가장 큰 날짜를 찾음
            prior_trading_days = df.index[df.index <= selected_analysis_date_ts]
            if not prior_trading_days.empty:
                actual_analysis_date = prior_trading_days.max()
                reference_date_price = df.loc[actual_analysis_date]['종가']

        if reference_date_price is None: return None, None, None # 기준일가 찾지 못하면 처리 중단

        # --- 최신 현재가 (오늘 기준) --- 
        latest_current_price = df.iloc[-1]['종가']

        df['수익률(1W)'] = df['종가'].pct_change(periods=5)
        df['수익률(2W)'] = df['종가'].pct_change(periods=10)
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래대금'] = df['종가'] * df['거래량']
        df['거래대금_MA20'] = df['거래대금'].rolling(window=20).mean()
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['단기 정배열'] = (df['MA5'] > df['MA20']).astype(int)
        df['52주_최고가'] = df['종가'].rolling(window=250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)

        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty: return None, None, None
        
        # 분석 기준일의 데이터를 기반으로 피처 생성
        # df_price에서 actual_analysis_date에 해당하는 행을 찾음
        # 만약 actual_analysis_date가 df_price의 인덱스에 없다면, 가장 가까운 이전 날짜를 사용
        if actual_analysis_date not in df.index:
            # 이 경우는 위에서 이미 처리되었어야 하지만, 혹시 모를 상황 대비
            prior_trading_days = df.index[df.index <= actual_analysis_date]
            if not prior_trading_days.empty:
                actual_analysis_date = prior_trading_days.max()
            else:
                return None, None, None # 유효한 분석 기준일 데이터를 찾을 수 없음

        latest_data = df.loc[actual_analysis_date].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price # 최신 현재가
        latest_data['기준일가'] = reference_date_price # 분석 기준일가

        market_cap = latest_data['현재가'] * shares # 시가총액은 최신 현재가 기준으로 계산
        latest_data['시가총액'] = market_cap / 1_0000_0000
        
        net_income = fs_data['당기순이익'].iloc[0] if '당기순이익' in fs_data.columns and not fs_data['당기순이익'].empty else 0
        total_equity = fs_data['자본총계'].iloc[0] if '자본총계' in fs_data.columns and not fs_data['자본총계'].empty else 0
        
        latest_data['PER'] = market_cap / net_income if net_income and net_income > 0 else np.nan
        latest_data['PBR'] = market_cap / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['ROE'] = net_income / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['log_mktcap'] = np.log(market_cap) if market_cap and market_cap > 0 else np.nan
        
        return latest_data, actual_analysis_date, latest_current_price # Return latest_data, actual_analysis_date, and latest_current_price
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None, None # Return None for all in case of error

def fetch_all_data(stock_list, selected_analysis_date):
    # <<< 핵심 수정 2: 모든 피처 데이터에 최신 거시 경제 지표를 추가 >>>
    # end_date는 항상 오늘 날짜로 설정하여 최신 현재가를 가져올 수 있도록 함
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    # 시작 날짜는 분석에 필요한 최소 기간 (예: 400일)을 유지
    start_date_for_fetch = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    
    latest_fs_df = get_latest_annual_fs_http(stock_list)

    if latest_fs_df.empty:
        print("재무 데이터 수집에 실패하여 분석을 중단합니다.")
        return pd.DataFrame(), None # Return empty DataFrame and None for analysis_date

    all_feature_data = []
    actual_analysis_date_final = None # Initialize to store the actual analysis date
    MAX_WORKERS = min(8, os.cpu_count() + 4)

    stock_records = stock_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(fetch_and_process_ticker_data, row, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): row 
            for row in stock_records
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_records), desc="전 종목 피처 생성"):
            try:
                result, date, _ = future.result()
                if result:
                    all_feature_data.append(result)
                    if actual_analysis_date_final is None and date is not None:
                        actual_analysis_date_final = date # Capture the date from the first successful fetch
            except Exception as e:
                # print(f"Error in feature generation for a stock: {e}") # 디버깅용
                continue

    if not all_feature_data: return pd.DataFrame(), None # Return empty DataFrame and None for analysis_date

    final_df = pd.DataFrame(all_feature_data)
    
    # 최신 거시 경제 데이터 가져오기
    latest_macro_df = _fetch_latest_macro_data(start_date_for_fetch, end_date_for_fetch)
    if not latest_macro_df.empty:
        # 모든 종목 데이터에 동일한 (오늘의) 거시 경제 상황을 적용
        for col in latest_macro_df.columns:
            final_df[col] = latest_macro_df[col].values[0]

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)

    print("모든 피처 데이터 생성 완료!")
    return final_df, actual_analysis_date_final # Return both DataFrame and analysis_date