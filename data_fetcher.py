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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIT_FS_PATH = os.path.join(PROJECT_ROOT, 'data', 'financial_data_pit.parquet')

def get_latest_annual_fs_http(stock_list):
    """ DART API를 통해 '작년' 재무 데이터를 직접 조회합니다. (실시간 분석용) """
    if config.DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키가 설정되지 않아 재무 데이터를 수집할 수 없습니다."); return pd.DataFrame()
    try:
        corp_code_map_path = os.path.join(PROJECT_ROOT, 'data', 'corp_code_map.csv')
        df_corp_map = pd.read_csv(corp_code_map_path, dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError:
        print(f"[오류] '{corp_code_map_path}' 파일을 찾을 수 없습니다."); return pd.DataFrame()
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()
    year = str(datetime.now().year - 1)
    print(f"실시간 분석(기준일=오늘): HTTP 통신으로 {year}년 재무 데이터 수집 시작.")
    all_fs_data = []
    for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터 수집"):
        corp_code_chunk = corp_codes[i:i+100]; corp_code_str = ','.join(corp_code_chunk)
        url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
        params = {'crtfc_key': config.DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': year, 'reprt_code': '11011'}
        try:
            res = requests.get(url, params=params, timeout=30); res.raise_for_status()
            data = res.json()
            if data.get('status') == '000': all_fs_data.extend(data.get('list', []))
        except Exception as e: tqdm.write(f"API 요청 중 오류 발생 (건너뜀): {e}"); continue
        time.sleep(0.2)
    if not all_fs_data: return pd.DataFrame()
    fs_df = pd.DataFrame(all_fs_data)
    required_accounts = ['당기순이익', '자본총계', '유동자산', '유동부채']
    fs_df = fs_df[fs_df['account_nm'].isin(required_accounts)]
    # <<< 핵심 수정 2: 여기서 미리 숫자형으로 변환 >>>
    fs_df['thstrm_amount'] = pd.to_numeric(fs_df['thstrm_amount'].str.replace(',', ''), errors='coerce')
    fs_pivot = fs_df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount').reset_index()
    fs_pivot.rename(columns={'stock_code':'종목코드'}, inplace=True)
    print(f"✅ {year}년 재무 데이터 수집 및 처리 완료: {len(fs_pivot)}개 기업")
    return fs_pivot

def get_fs_data_from_pit(stock_list, selected_analysis_date):
    """ 분석 기준일 시점에서 사용 가능한 최신 재무 데이터를 PIT 데이터베이스에서 조회합니다. """
    try:
        pit_fs_df = pd.read_parquet(PIT_FS_PATH)
    except FileNotFoundError:
        print(f"[오류] 재무 데이터베이스 파일({PIT_FS_PATH})을 찾을 수 없습니다.")
        print("먼저 `scripts/build_financial_db.py`를 실행하여 데이터베이스를 생성해주세요."); return pd.DataFrame()
    print(f"과거 분석(기준일={selected_analysis_date.strftime('%Y-%m-%d')}): PIT DB에서 재무 데이터 조회.")
    analysis_date_ts = pd.to_datetime(selected_analysis_date)
    available_fs = pit_fs_df[pit_fs_df['공시일'] <= analysis_date_ts].copy()
    if available_fs.empty:
        print(f"경고: {analysis_date_ts.strftime('%Y-%m-%d')} 이전에 공시된 재무 데이터가 없습니다."); return pd.DataFrame()
    latest_fs = available_fs.sort_values('공시일').drop_duplicates(subset=['종목코드'], keep='last')
    result_df = pd.merge(stock_list[['종목코드']], latest_fs, on='종목코드', how='left')
    print(f"✅ {analysis_date_ts.strftime('%Y-%m-%d')} 기준, 사용 가능한 재무 데이터 처리 완료: {len(result_df)}개 기업")
    return result_df

def fetch_stock_list():
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
    try:
        df_marcap = fdr.StockListing('KRX-MARCAP')
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        stock_list = stock_list[stock_list['상장주식수'] > 0]
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}"); return pd.DataFrame()

def _fetch_macro_data(start_date, end_date):
    print(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date); usdkrw = fdr.DataReader('USD/KRW', start_date, end_date); vix = fdr.DataReader('^VIX', start_date, end_date)
        macro_df = pd.concat([kospi['Close'].rename('KOSPI'), usdkrw['Close'].rename('USDKRW'), vix['Close'].rename('VIX')], axis=1).ffill()
        for col in ['KOSPI', 'USDKRW', 'VIX']:
            if col in macro_df.columns: macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1); macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        print("✅ 거시 경제 지표 수집 완료."); return macro_df
    except Exception as e:
        print(f"거시 경제 지표 수집 실패: {e}"); return pd.DataFrame()

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']; shares = stock_info['상장주식수']
    try:
        df_price_full = fdr.DataReader(ticker, start_date_for_fetch, end_date_for_fetch)
        if df_price_full.empty or len(df_price_full) < 251: return None, None
        df_price_full.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None
        actual_analysis_date = df_temp.index.max(); reference_date_price = df_temp.loc[actual_analysis_date]['종가']
        latest_current_price = df_price_full.iloc[-1]['종가']
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        df_for_indicators['수익률(1W)'] = df_for_indicators['종가'].pct_change(5); df_for_indicators['수익률(2W)'] = df_for_indicators['종가'].pct_change(10)
        df_for_indicators['수익률(1M)'] = df_for_indicators['종가'].pct_change(20); df_for_indicators['수익률(3M)'] = df_for_indicators['종가'].pct_change(60)
        df_for_indicators['변동성(1M)'] = df_for_indicators['종가'].rolling(20).std() / df_for_indicators['종가'].rolling(20).mean()
        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']; df_for_indicators['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(20).mean()
        df_for_indicators['MA5'] = df_for_indicators['종가'].rolling(5).mean(); df_for_indicators['MA20'] = df_for_indicators['종가'].rolling(20).mean()
        df_for_indicators['단기 정배열'] = (df_for_indicators['MA5'] > df_for_indicators['MA20']).astype(int)
        df_for_indicators['52주_최고가'] = df_for_indicators['종가'].rolling(250).max(); df_for_indicators['52주_신고가_비율'] = df_for_indicators['종가'] / df_for_indicators['52주_최고가']
        df_for_indicators.ta.rsi(close='종가', length=14, append=True); df_for_indicators.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty: return None, None
        latest_data = df_for_indicators.loc[actual_analysis_date].to_dict()
        latest_data['종목코드'] = stock_info['종목코드']; latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price; latest_data['기준일가'] = reference_date_price
        
        market_cap = reference_date_price * shares; latest_data['시가총액'] = market_cap / 1_0000_0000
        # <<< 핵심 수정 2: .iloc[0] 대신 .values[0]을 사용하여 더 안정적으로 값 추출 >>>
        net_income = fs_data['당기순이익'].values if '당기순이익' in fs_data.columns and not fs_data['당기순이익'].isnull().all() else 0
        total_equity = fs_data['자본총계'].values if '자본총계' in fs_data.columns and not fs_data['자본총계'].isnull().all() else 0
        latest_data['PER'] = market_cap / net_income if net_income > 0 else np.nan
        latest_data['PBR'] = market_cap / total_equity if total_equity > 0 else np.nan
        latest_data['ROE'] = net_income / total_equity if total_equity > 0 else np.nan
        latest_data['log_mktcap'] = np.log(market_cap) if market_cap > 0 else np.nan
        return latest_data, actual_analysis_date
    except Exception: return None, None

def fetch_all_data(stock_list, selected_analysis_date):
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    if selected_analysis_date.date() == today.date():
        latest_fs_df = get_latest_annual_fs_http(stock_list)
    else:
        latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date)
    if latest_fs_df.empty:
        print("재무 데이터 수집에 실패하여 분석을 중단합니다."); return pd.DataFrame(), None
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        print("거시 경제 데이터 수집에 실패하여 분석을 중단합니다."); return pd.DataFrame(), None
    all_feature_data, all_actual_dates = [], []
    stock_records = stock_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_stock = {executor.submit(fetch_and_process_ticker_data, r, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): r for r in stock_records}
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), total=len(stock_records), desc="전 종목 피처 생성"):
            try:
                result, analysis_date = future.result()
                if result and analysis_date: all_feature_data.append(result); all_actual_dates.append(analysis_date)
            except Exception: continue
    if not all_feature_data: return pd.DataFrame(), None
    final_df = pd.DataFrame(all_feature_data)
    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)
    
    # <<< 핵심 수정 1: .mode()의 결과에서 첫 번째 값([0])을 선택하여 Series가 아닌 Timestamp를 반환 >>>
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0]) if not final_df.empty else None
    
    print("모든 피처 데이터 생성 완료!")
    return final_df, actual_analysis_date_final