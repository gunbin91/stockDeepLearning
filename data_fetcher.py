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
import config

# DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_latest_annual_fs_http(stock_list):
    # 이 함수는 변경할 필요가 없습니다.
    if config.DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
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

def _fetch_macro_data(start_date, end_date):
    """지정된 기간 동안의 거시 경제 지표 데이터를 수집합니다."""
    print(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
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

        for col in ['KOSPI', 'USDKRW', 'VIX']:
            if col in macro_df.columns:
                macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
                macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        
        print("✅ 거시 경제 지표 수집 완료.")
        return macro_df
    except Exception as e:
        print(f"거시 경제 지표 수집 실패: {e}")
        return pd.DataFrame()

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        # 1. 전체 기간 데이터 로드 (현재가 확인용)
        df_price_full = fdr.DataReader(ticker, start_date_for_fetch, end_date_for_fetch)
        if df_price_full.empty or len(df_price_full) < 251: return None, None

        df_price_full.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        # 2. 실제 분석 기준일 및 기준일가 확정
        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        
        # 기준일 이전 데이터만 필터링
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None # 기준일 이전에 데이터가 없으면 처리 불가

        actual_analysis_date = df_temp.index.max()
        reference_date_price = df_temp.loc[actual_analysis_date]['종가']

        # 3. 최신 현재가 저장
        latest_current_price = df_price_full.iloc[-1]['종가']

        # 4. 기술적 지표 계산용 데이터프레임 생성 (분석 기준일까지만 포함)
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()

        # 5. 기술적 지표 계산 (수정된 df_for_indicators 사용)
        df_for_indicators['수익률(1W)'] = df_for_indicators['종가'].pct_change(periods=5)
        df_for_indicators['수익률(2W)'] = df_for_indicators['종가'].pct_change(periods=10)
        df_for_indicators['수익률(1M)'] = df_for_indicators['종가'].pct_change(periods=20)
        df_for_indicators['수익률(3M)'] = df_for_indicators['종가'].pct_change(periods=60)
        df_for_indicators['변동성(1M)'] = df_for_indicators['종가'].rolling(window=20).std() / df_for_indicators['종가'].rolling(window=20).mean()
        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        df_for_indicators['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(window=20).mean()
        df_for_indicators['MA5'] = df_for_indicators['종가'].rolling(window=5).mean()
        df_for_indicators['MA20'] = df_for_indicators['종가'].rolling(window=20).mean()
        df_for_indicators['단기 정배열'] = (df_for_indicators['MA5'] > df_for_indicators['MA20']).astype(int)
        df_for_indicators['52주_최고가'] = df_for_indicators['종가'].rolling(window=250).max()
        df_for_indicators['52주_신고가_비율'] = df_for_indicators['종가'] / df_for_indicators['52주_최고가']
        
        df_for_indicators.ta.rsi(close='종가', length=14, append=True)
        df_for_indicators.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)

        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty: return None, None
        
        # 6. 분석 기준일의 데이터 추출
        latest_data = df_for_indicators.loc[actual_analysis_date].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price   # 최신 현재가
        latest_data['기준일가'] = reference_date_price # 분석 기준일 종가

        # 7. 재무 지표 계산 (시가총액은 현재가 기준)
        market_cap = latest_current_price * shares
        latest_data['시가총액'] = market_cap / 1_0000_0000
        
        net_income = fs_data['당기순이익'].iloc[0] if '당기순이익' in fs_data.columns and not fs_data['당기순이익'].empty else 0
        total_equity = fs_data['자본총계'].iloc[0] if '자본총계' in fs_data.columns and not fs_data['자본총계'].empty else 0
        
        latest_data['PER'] = market_cap / net_income if net_income and net_income > 0 else np.nan
        latest_data['PBR'] = market_cap / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['ROE'] = net_income / total_equity if total_equity and total_equity > 0 else np.nan
        latest_data['log_mktcap'] = np.log(market_cap) if market_cap and market_cap > 0 else np.nan
        
        return latest_data, actual_analysis_date
    except Exception as e:
        # print(f"Error processing {ticker}: {e}") # 디버깅용
        return None, None

def fetch_all_data(stock_list, selected_analysis_date):
    # 1. 데이터 수집 기간 설정
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    
    # 2. 재무 데이터 및 거시 경제 데이터 미리 수집
    latest_fs_df = get_latest_annual_fs_http(stock_list)
    if latest_fs_df.empty:
        print("재무 데이터 수집에 실패하여 분석을 중단합니다.")
        return pd.DataFrame(), None

    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        print("거시 경제 데이터 수집에 실패하여 분석을 중단합니다.")
        return pd.DataFrame(), None

    # 3. 개별 종목 피처 생성 (병렬 처리)
    all_feature_data = []
    all_actual_dates = []
    MAX_WORKERS = min(8, os.cpu_count() + 4)

    stock_records = stock_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(fetch_and_process_ticker_data, row, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): row 
            for row in stock_records
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_records), desc="전 종목 피처 생성"):
            try:
                result, analysis_date = future.result()
                if result and analysis_date:
                    all_feature_data.append(result)
                    all_actual_dates.append(analysis_date)
            except Exception as e:
                # print(f"Error in feature generation for a stock: {e}") # 디버깅용
                continue

    if not all_feature_data: return pd.DataFrame(), None

    # 4. 결과 통합 및 후처리
    final_df = pd.DataFrame(all_feature_data)
    final_df['date'] = all_actual_dates
    final_df['date'] = pd.to_datetime(final_df['date'])

    # 5. 거시 경제 데이터 병합
    # 각 종목의 실제 분석 날짜(date)를 기준으로 거시 경제 데이터를 매핑
    # merge_asof를 사용하여 각 분석일에 가장 가까운 과거의 거시경제 데이터를 사용
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)

    # 최종 분석 기준일은 모든 데이터 처리 후 가장 빈번했던 날짜로 확정
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0]) if not final_df.empty else None

    print("모든 피처 데이터 생성 완료!")
    return final_df, actual_analysis_date_final