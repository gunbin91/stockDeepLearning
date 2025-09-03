# scripts/build_financial_db.py

import pandas as pd
from tqdm import tqdm
import time
import os
import sys
import io
from datetime import datetime
import FinanceDataReader as fdr
import requests
import concurrent.futures

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 설정 ---
START_YEAR = 2018
END_YEAR = datetime.now().year
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'financial_data_pit.parquet')
API_TIMEOUT = 30
MAX_WORKERS = 16

ACCOUNTS_TO_EXTRACT = ['유동자산', '유동부채', '자본총계', '당기순이익']

def get_corp_list_for_processing():
    print("1. FinanceDataReader를 통해 KOSPI, KOSDAQ 보통주 목록을 필터링합니다...")
    try:
        df_kospi = fdr.StockListing('KOSPI'); df_kosdaq = fdr.StockListing('KOSDAQ')
        df_krx = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        df_krx = df_krx[~df_krx['Name'].str.contains('스팩|리츠', na=False, case=False)]
        df_krx_common = df_krx[df_krx['Code'].str.endswith('0')].copy()
        clean_stock_list = df_krx_common[['Code', 'Name']].rename(columns={'Code': '종목코드', 'Name': '종목명'})
    except Exception as e:
        print(f"   - FinanceDataReader에서 종목 목록 가져오기 실패: {e}"); return None
    print("2. DART 고유번호 목록과 필터링된 목록을 매핑합니다...")
    try:
        import dart_fss as dart
        dart.set_api_key(api_key=config.DART_API_KEY)
        corp_list = dart.get_corp_list()
        dart_corps_data = [{'corp_code': c.corp_code, '종목코드': c.stock_code} for c in corp_list if c.stock_code]
        df_dart_map = pd.DataFrame(dart_corps_data)
        corps_to_process_df = pd.merge(clean_stock_list, df_dart_map, on='종목코드', how='inner')
        return corps_to_process_df
    except Exception as e:
        print(f"   - DART 고유번호 목록 가져오기 실패: {e}"); return None

def worker_fetch_financial_data(params):
    year, r_code, chunk = params
    corp_code_str = ','.join(chunk)
    url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
    api_params = {'crtfc_key': config.DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': str(year), 'reprt_code': r_code}
    try:
        res = requests.get(url, params=api_params, timeout=API_TIMEOUT); res.raise_for_status()
        data = res.json()
        if data.get('status') == '000': return data.get('list', [])
    except requests.exceptions.RequestException: pass
    return []

def worker_fetch_filing_dates_by_corp(corp_code):
    all_filings = []
    business_report_types = ['A001', 'A002', 'A003']
    for bsn_tp in business_report_types:
        page_no = 1
        while True:
            url = "https://opendart.fss.or.kr/api/list.json"
            params = {
                'crtfc_key': config.DART_API_KEY, 'corp_code': corp_code,
                'bgn_de': f"{START_YEAR}0101", 'end_de': f"{END_YEAR}1231",
                'bsn_tp': bsn_tp, 'page_no': page_no, 'page_count': 100
            }
            try:
                res = requests.get(url, params=params, timeout=API_TIMEOUT); res.raise_for_status()
                data = res.json()
                if data.get('status') == '000':
                    filings_list = data.get('list', [])
                    all_filings.extend(filings_list)
                    if data.get('page_no', 1) >= data.get('total_page', 1): break
                    page_no += 1
                else: break
            except requests.exceptions.RequestException: break
            time.sleep(0.1)
    return all_filings

def build_point_in_time_db():
    if not config.DART_API_KEY or config.DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("오류: config.py에 DART API 키를 설정해주세요."); return
    corps_to_process_df = get_corp_list_for_processing()
    if corps_to_process_df is None: return
    print(f"\n - 총 {len(corps_to_process_df)}개 기업을 대상으로 수집을 시작합니다.")

    all_fs_data = []
    corp_codes = corps_to_process_df['corp_code'].tolist()
    corp_code_chunks = [corp_codes[i:i+100] for i in range(0, len(corp_codes), 100)]
    report_codes = ['11013', '11012', '11014', '11011']
    tasks_fs = [(year, r_code, chunk) for year in range(START_YEAR, END_YEAR + 1) for r_code in report_codes for chunk in corp_code_chunks]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(tasks_fs), desc="재무 데이터 병렬 수집") as pbar:
            future_to_task = {executor.submit(worker_fetch_financial_data, task): task for task in tasks_fs}
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result(); pbar.update(1)
                if result: all_fs_data.extend(result)
    
    if not all_fs_data: print("\n수집된 재무 데이터가 없습니다."); return

    print(f"\n3. {len(all_fs_data)}개 재무 레코드 수집 완료. 후처리를 시작합니다...")
    fs_df = pd.DataFrame(all_fs_data)
    
    # <<< 핵심 수정: 'corp_code'를 선택 컬럼에 포함 >>>
    fs_df = fs_df[['rcept_no', 'corp_code', 'stock_code', 'account_nm', 'thstrm_amount', 'fs_div']]
    fs_df.rename(columns={'stock_code': '종목코드'}, inplace=True)
    
    fs_df = fs_df[fs_df['account_nm'].isin(ACCOUNTS_TO_EXTRACT)]
    fs_df.sort_values(by='fs_div', ascending=False, inplace=True)
    fs_df.drop_duplicates(subset=['rcept_no', '종목코드', 'account_nm'], keep='first', inplace=True)
    pivot_df = fs_df.pivot_table(index=['rcept_no', 'corp_code', '종목코드'], columns='account_nm', values='thstrm_amount', aggfunc='first').reset_index()
    
    print("4. 공시일 정보 수집 및 병합 중...")
    unique_corp_codes = pivot_df['corp_code'].unique().tolist()
    all_filings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(unique_corp_codes), desc="공시일 정보 병렬 수집 (기업별)") as pbar:
            future_to_corp = {executor.submit(worker_fetch_filing_dates_by_corp, code): code for code in unique_corp_codes}
            for future in concurrent.futures.as_completed(future_to_corp):
                result = future.result(); pbar.update(1)
                if result: all_filings.extend(result)

    if not all_filings: print("\n경고: 공시일 정보를 전혀 수집하지 못했습니다."); return
    date_map = {item['rcept_no']: pd.to_datetime(item['rcept_dt']) for item in all_filings}

    pivot_df['공시일'] = pivot_df['rcept_no'].map(date_map)
    pivot_df.dropna(subset=['공시일'], inplace=True)

    final_df = pivot_df[['공시일', '종목코드'] + ACCOUNTS_TO_EXTRACT].copy()
    for col in ACCOUNTS_TO_EXTRACT:
        final_df[col] = pd.to_numeric(final_df[col].str.replace(',', ''), errors='coerce')
    final_df.dropna(subset=ACCOUNTS_TO_EXTRACT, inplace=True)

    if final_df.empty: print("\n경고: 최종 데이터 생성에 실패했습니다."); return

    final_df.sort_values(by=['종목코드', '공시일'], inplace=True)
    final_df = final_df[final_df['종목코드'].isin(corps_to_process_df['종목코드'])].copy()
    final_df.drop_duplicates(subset=['종목코드', '공시일'], keep='last', inplace=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    unique_stocks_count = len(final_df['종목코드'].unique())
    print("-" * 50)
    print(f"✅ 재무 데이터베이스 구축 완료! 총 {len(final_df)}개의 유효 레코드가 생성되었습니다.")
    print(f"   - 최종적으로 {unique_stocks_count}개 종목의 데이터가 성공적으로 수집되었습니다.")
    print(f"   - 저장 경로: {OUTPUT_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    build_point_in_time_db()