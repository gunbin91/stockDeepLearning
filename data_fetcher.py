import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import dart_fss as dart
from datetime import datetime, timedelta
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm

# -----------------------------------------------------------------------------
# ⚠️ [필수] 여기에 발급받은 DART API 인증키를 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522"
# -----------------------------------------------------------------------------

# DART API 키 설정
if DART_API_KEY != "여기에_발급받은_DART_인증키를_붙여넣으세요":
    dart.set_api_key(api_key=DART_API_KEY)
else:
    print("="*80); print(" [경고] data_fetcher.py 파일에 DART API 키를 입력해야 합니다. "); print("="*80)

def get_all_financial_data(stock_list, year):
    """지정한 연도의 모든 상장기업 재무제표를 DART에서 일괄 수집합니다."""
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        return pd.DataFrame()
    
    try:
        
        corp_list = dart.get_corp_list()
        
        all_fs_data = []
        
        MAX_WORKERS_FS = 4 # DART API 호출 제한을 고려하여 워커 수 설정
        
        def _fetch_single_fs_data(corp_obj, stock_code_fdr, year, required_accounts):
            try:
                corp_code = corp_obj.corp_code
                fs = dart.fs.extract(corp_code=corp_code, bgn_de=f'{year}0101', fs_tp=('bs', 'is'))
                
                if fs is not None:
                    bs_df = fs['bs']
                    is_df = fs['is']
                    
                    combined_fs = pd.concat([bs_df, is_df], ignore_index=True)
                    
                    filtered_fs = combined_fs[combined_fs['account_nm'].isin(required_accounts)]
                    
                    if not filtered_fs.empty:
                        filtered_fs['stock_code'] = stock_code_fdr
                        filtered_fs['corp_name'] = corp_obj.corp_name
                        return filtered_fs
                return None
            except Exception as e:
                print(f"Error fetching data for {stock_code_fdr} ({corp_code}): {e}")
                return None

        all_fs_data = []
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_FS) as executor:
            required_accounts = ['자산총계', '부채총계', '자본총계', '당기순이익']
            for index, row in stock_list.iterrows():
                stock_code_fdr = row['종목코드']
                corp_obj = corp_list.find_by_stock_code(stock_code_fdr)
                
                if corp_obj is None:
                    continue
                
                futures.append(executor.submit(_fetch_single_fs_data, corp_obj, stock_code_fdr, year, required_accounts))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{year}년 재무 데이터 수집"):
                result = future.result()
                if result is not None:
                    all_fs_data.append(result)

        if not all_fs_data:
            print(f"{year}년 재무 데이터 수집에 실패했습니다.")
            return pd.DataFrame()

        combined_df = pd.concat(all_fs_data, ignore_index=True)
        
        fs_pivot = combined_df.pivot_table(index=['stock_code', 'corp_name'], 
                                         columns='account_nm', 
                                         values='thstrm_amount').reset_index()

        required_accounts_final = ['자산총계', '부채총계', '자본총계', '당기순이익']
        for col in required_accounts_final:
            if col in fs_pivot.columns:
                fs_pivot[col] = pd.to_numeric(fs_pivot[col].str.replace(',', ''), errors='coerce')
        
        fs_pivot.rename(columns={'stock_code':'종목코드'}, inplace=True)
        print(f"✅ {year}년 재무 데이터 수집 및 처리 완료: {len(fs_pivot)}개 기업")
        return fs_pivot[['종목코드'] + required_accounts_final]
        
    except Exception as e:
        print(f"DART 데이터 전체 수집 중 오류 발생: {e}")
        return pd.DataFrame()


def fetch_stock_list():
    """FinanceDataReader를 사용하여 KOSPI와 KOSDAQ의 전 종목 리스트를 수집합니다."""
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 리스트를 수집합니다...")
    try:
        df_krx = fdr.StockListing('KRX')
        df_krx = df_krx[df_krx['Market'].isin(['KOSPI', 'KOSDAQ']) & 
                        df_krx['Code'].str.match(r'^\d{6}$') & 
                        ~df_krx['Name'].str.contains('스팩|리츠')].copy()
        if 'Shares' in df_krx.columns:
            stock_list = df_krx[['Code', 'Name', 'Shares']].copy()
            stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Shares': '상장주식수'}, inplace=True)
        else:
            stock_list = df_krx[['Code', 'Name']].copy()
            stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
            stock_list['상장주식수'] = np.nan
            print("경고: 'Shares' 컬럼을 찾을 수 없어 '상장주식수'를 NaN으로 설정합니다.")
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패: {e}")
        return pd.DataFrame(columns=['종목코드', '종목명', '상장주식수'])


def fetch_and_process_ticker_data(stock_info, start_date, end_date, latest_fs_df):
    """한 종목의 데이터를 수집/가공합니다. (fdr + dart 조합)"""
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 61: return None

        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        df = df_price.copy()
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)

        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty or '당기순이익' not in fs_data.columns or '자본총계' not in fs_data.columns: return None
        
        latest_data = df.iloc[-1].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = df.iloc[-1]['종가']

        market_cap = latest_data['현재가'] * shares
        latest_data['시가총액'] = market_cap / 1_0000_0000
        
        net_income = fs_data['당기순이익'].iloc[0]
        total_equity = fs_data['자본총계'].iloc[0]
        
        latest_data['PER'] = market_cap / net_income if net_income and net_income > 0 else np.nan
        latest_data['PBR'] = market_cap / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['ROE'] = net_income / total_equity if total_equity and total_equity > 0 else np.nan
        
        latest_data['기관순매수(억)'] = 0
        latest_data['외국인순매수(억)'] = 0
        return latest_data
    except Exception:
        return None

def fetch_all_data(stock_list):
    """주어진 종목 리스트에 대해 모든 피처를 병렬로 계산합니다."""
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    
    latest_fs_year = today.year - 1
    latest_fs_df = get_all_financial_data(stock_list, latest_fs_year)

    if latest_fs_df.empty:
        print("재무 데이터 수집에 실패하여 분석을 중단합니다.")
        return pd.DataFrame(), pd.DataFrame()

    all_feature_data = []
    failed_tickers = []
    MAX_WORKERS = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(fetch_and_process_ticker_data, row, start_date, end_date, latest_fs_df): row 
            for row in stock_list.to_dict('records')
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_list), desc="전 종목 피처 생성"):
            stock_info = future_to_stock[future]
            try:
                result = future.result()
                if result: all_feature_data.append(result)
                else: failed_tickers.append(stock_info['종목명'])
            except Exception: failed_tickers.append(stock_info['종목명'])

    print(f"\n✅ 최종 분석 대상: {len(all_feature_data)} 종목")
    if failed_tickers: print(f"❌ 데이터 부족/실패로 제외: {len(failed_tickers)} 종목")
    
    if not all_feature_data: return pd.DataFrame(), pd.DataFrame()

    final_df = pd.DataFrame(all_feature_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.fillna(final_df.median(numeric_only=True), inplace=True)
    final_df.fillna(0, inplace=True)

    print("모든 피처 데이터 생성 완료!")
    return final_df, final_df
