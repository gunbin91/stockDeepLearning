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

# -----------------------------------------------------------------------------
# ⚠️ [필수] 여기에 발급받은 DART API 인증키를 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체
# -----------------------------------------------------------------------------

def get_latest_annual_fs_http(stock_list):
    """'다중회사 주요계정' API를 직접 HTTP 통신으로 호출하여 재무 데이터를 가져옵니다."""
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키가 설정되지 않아 재무 데이터를 수집할 수 없습니다.")
        return pd.DataFrame()
        
    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError:
        print("[오류] 'corp_code_map.csv' 파일을 찾을 수 없습니다.")
        print("사전 준비 단계의 'make_corp_map.py' 스크립트를 먼저 실행하여 파일을 생성해주세요.")
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
            'crtfc_key': DART_API_KEY,
            'corp_code': corp_code_str,
            'bsns_year': year,
            'reprt_code': '11011', # 사업보고서
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
    """FinanceDataReader를 사용하여 KOSPI와 KOSDAQ의 전 종목 리스트와 시가총액 정보를 수집합니다."""
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


def fetch_and_process_ticker_data(stock_info, start_date, end_date, latest_fs_df):
    """한 종목의 데이터를 수집/가공합니다."""
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 251: return None # 52주 데이터 확보

        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        df = df_price.copy()
        
        # 기술적 지표 계산
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
        if fs_data.empty: return None
        
        # 최신 데이터 추출
        latest_data = df.iloc[-1].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = df.iloc[-1]['종가']

        market_cap = latest_data['현재가'] * shares
        latest_data['시가총액'] = market_cap / 1_0000_0000
        
        # 재무 지표 계산
        net_income = fs_data['당기순이익'].iloc[0] if '당기순이익' in fs_data.columns and not fs_data['당기순이익'].empty else 0
        total_equity = fs_data['자본총계'].iloc[0] if '자본총계' in fs_data.columns and not fs_data['자본총계'].empty else 0
        
        latest_data['PER'] = market_cap / net_income if net_income and net_income > 0 else np.nan
        latest_data['PBR'] = market_cap / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['ROE'] = net_income / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['log_mktcap'] = np.log(market_cap) if market_cap and market_cap > 0 else np.nan
        
        return latest_data
    except Exception:
        return None

def fetch_all_data(stock_list):
    """주어진 종목 리스트에 대해 모든 피처를 병렬로 계산합니다."""
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    
    latest_fs_df = get_latest_annual_fs_http(stock_list)

    if latest_fs_df.empty:
        print("재무 데이터 수집에 실패하여 분석을 중단합니다.")
        return pd.DataFrame()

    all_feature_data = []
    MAX_WORKERS = min(8, os.cpu_count() + 4)

    stock_records = stock_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(fetch_and_process_ticker_data, row, start_date, end_date, latest_fs_df): row 
            for row in stock_records
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_records), desc="전 종목 피처 생성"):
            try:
                result = future.result()
                if result: all_feature_data.append(result)
            except Exception:
                continue

    if not all_feature_data: return pd.DataFrame()

    final_df = pd.DataFrame(all_feature_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 필수 컬럼 결측치 제거
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)


    print("모든 피처 데이터 생성 완료!")
    # <<< 개선됨: 하나의 데이터프레임만 반환
    return final_df
