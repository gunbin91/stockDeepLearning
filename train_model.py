import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import requests
import time
from datetime import datetime, timedelta
import joblib
import pandas_ta as ta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import concurrent.futures
from tqdm import tqdm
import warnings
import argparse
import os

warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# ⚠️ [필수] data_fetcher.py와 동일한 DART API 인증키를 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체
# -----------------------------------------------------------------------------

def get_financial_data_for_training_http(corp_codes, start_year, end_year):
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요": return {}
    
    all_fs_data = {}
    for year in range(start_year, end_year + 1):
        print(f"HTTP 통신으로 {year}년 재무 데이터 수집 중...")
        year_fs_data = []
        for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터"):
            corp_code_chunk = corp_codes[i:i+100]
            corp_code_str = ','.join(corp_code_chunk)
            
            url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
            params = { 'crtfc_key': DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': str(year), 'reprt_code': '11011' }
            
            try:
                res = requests.get(url, params=params)
                if res.status_code == 200 and res.json().get('status') == '000':
                    year_fs_data.extend(res.json()['list'])
            except Exception:
                continue
            time.sleep(0.2)
        
        if year_fs_data:
            df = pd.DataFrame(year_fs_data)
            df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'].str.replace(',', ''), errors='coerce')
            df_pivot = df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount')
            all_fs_data[year] = df_pivot.to_dict('index')
            
    print("✅ 전체 재무 데이터 수집 완료")
    return all_fs_data

def fetch_stock_list():
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 리스트를 수집합니다...")
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list[['종목코드', '종목명']]
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패: {e}")
        return pd.DataFrame(columns=['종목코드', '종목명'])

# <<< 수정됨: 함수 이름 명확화 및 merge_asof 로직 적용
def process_single_ticker_data(stock_info, start_date, end_date, all_fs_data, df_marcap_long):
    ticker = stock_info['종목코드']
    
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 251 + 60: return None

        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year

        # <<< 핵심 수정: merge_asof 를 위한 데이터 준비 및 실행 >>>
        # 1. 전체 시가총액 데이터에서 해당 종목 데이터만 필터링
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: return None

        # 2. merge_asof를 위해 양쪽 데이터프레임의 인덱스를 정렬
        df.sort_index(inplace=True)
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        
        # 3. merge_asof 실행
        #    - on='Date' 대신 left_index=True, right_on='Date' 사용
        df = pd.merge_asof(
            left=df,
            right=df_marcap_ticker[['Date', 'Marcap']],
            left_index=True,
            right_on='Date',
            direction='backward' # 각 일별 주가에 대해 가장 가까운 과거의 시총 데이터를 찾아줌
        )
        
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        df['거래대금'] = df['종가'] * df['거래량']
        df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        
        df[['당기순이익', '자본총계']] = df[['당기순이익', '자본총계']].ffill()
        if df[['당기순이익', '자본총계']].isnull().values.any(): return None
        
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        df['log_mktcap'] = np.log(df['시가총액'])
        
        df['수익률(1W)'] = df['종가'].pct_change(periods=5)
        df['수익률(2W)'] = df['종가'].pct_change(periods=10)
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(window=20).mean()

        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['단기 정배열'] = (df['MA5'] > df['MA20']).astype(int)
        df['52주_최고가'] = df['종가'].rolling(window=250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        
        # Date 컬럼을 다시 인덱스로 설정하여 반환
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None

def create_training_data(stock_list, period_days=365*4):
    print("안정적인 API 기반으로 학습 데이터 생성을 시작합니다...")
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=period_days)).strftime('%Y-%m-%d')
    
    print("KRX 전 종목의 과거 시가총액 데이터를 수집합니다 (매월 말 기준, 시간이 다소 소요될 수 있습니다)...")
    try:
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()
        
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in month_end_dates}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(month_end_dates), desc="전체 시가총액 데이터 수집"):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty:
                        result_df['Date'] = pd.to_datetime(date_str)
                        marcap_dfs.append(result_df)
                except Exception:
                    continue
        
        if not marcap_dfs:
            raise Exception("수집된 시가총액 데이터가 없습니다.")

        # <<< 수정됨: 피벗 대신 long format 데이터프레임으로 사용
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True) # merge_asof를 위해 정렬

    except Exception as e:
        print(f"과거 시가총액 데이터 수집 실패: {e}")
        return None, None, None

    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError: return None, None, None
    
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()

    all_fs_data = get_financial_data_for_training_http(corp_codes, today.year - 5, today.year - 1)
    if not all_fs_data: return None, None, None

    all_data = []
    MAX_WORKERS = min(8, os.cpu_count() + 4)
    stock_records = target_stocks.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # <<< 수정됨: 함수 이름 및 전달 인자 변경
        futures = {
            executor.submit(process_single_ticker_data, row, start_date, end_date, all_fs_data, df_marcap_long): row
            for row in stock_records
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stock_records), desc="학습 데이터 생성"):
            result_df = future.result()
            if result_df is not None:
                all_data.append(result_df)

    if not all_data: 
        print("생성된 데이터가 없습니다. 프로세스를 중단합니다.")
        return None, None, None

    final_df = pd.concat(all_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"\n--- 생성된 학습 데이터 요약 ---")
    print(f"1. 전체 수집 데이터 (Raw): {len(final_df):,} 행")
    
    training_start_date = (today - timedelta(days=365*3)).strftime('%Y-%m-%d')
    final_df = final_df[final_df.index >= training_start_date]
    print(f"2. 워밍업 기간 제외 후: {len(final_df):,} 행")

    features = [
        '수익률(1W)', '수익률(2W)', '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 'log_mktcap', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', '거래대금_MA20', '단기 정배열', '52주_신고가_비율'
    ]
    target = 'target'
    
    final_df.dropna(subset=[target] + features, inplace=True)
    print(f"3. 결측치(NaN) 제거 후: {len(final_df):,} 행")

    if final_df.empty:
        print("오류: 최종 학습 데이터가 비어있습니다.")
        return None, None, None

    X = final_df[features].astype(np.float32)
    y = final_df[target]
    
    print(f"4. 최종 학습 데이터셋 (X): {X.shape}")
    print(f"   - 타겟 분포 (y):\n{y.value_counts(normalize=True).to_string()}")
    print("---------------------------------\
")

    print("학습 데이터 생성 완료!")
    return X, y, features

def train_evaluate_and_save_model(X, y, features, n_jobs, n_iter, max_depth_list, model_path='stock_prediction_model_rf_upgraded.joblib'):
    if X is None or y is None or X.empty or y.empty:
        print("학습 데이터가 없어 모델링을 건너뜁니다.")
        return

    print("모델 학습 및 평가를 시작합니다...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\n피처 스케일링 (StandardScaler) 적용...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n학습 데이터 타겟 분포:\n", y_train.value_counts(normalize=True))

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': max_depth_list,
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_samples': [0.7, 0.8, 0.9, None]
    }

    print("RandomizedSearchCV를 사용하여 최적 파라미터 탐색...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced', oob_score=True)
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=n_iter, cv=3, n_jobs=n_jobs, 
                                       verbose=2, random_state=42, scoring='roc_auc')

    random_search.fit(X_train_scaled, y_train)

    print("\n--- 최적 파라미터 탐색 결과 ---")
    print(f"최고 점수 (ROC-AUC): {random_search.best_score_:.4f}")
    print("최적 파라미터:", random_search.best_params_)

    best_model = random_search.best_estimator_
    
    if hasattr(best_model, 'oob_score_') and best_model.oob_score_:
        print(f"OOB Score (자체 검증 점수): {best_model.oob_score_:.4f}")

    print("최적 모델로 테스트 데이터 평가...")
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- 최종 모델 평가 결과 ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n분류 보고서 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['하락(0)', '상승(1)']))

    joblib.dump({'model': best_model, 'features': features, 'scaler': scaler}, model_path)
    print(f"\n✅ 새로운 데이터로 학습된 최적 모델과 스케일러를 '{model_path}' 경로에 저장했습니다.")

def main():
    parser = argparse.ArgumentParser(description="RandomForest 모델 학습 및 하이퍼파라미터 튜닝")
    parser.add_argument('--n_jobs', type=int, default=-1, help='사용할 CPU 코어 수 (-1은 모든 코어 사용)')
    parser.add_argument('--n_iter', type=int, default=10, help='RandomizedSearchCV 반복 횟수')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[10, 20, 30], help='max_depth 후보 리스트')
    args = parser.parse_args()

    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키를 train_model.py 파일에 입력해주세요.")
        return

    stock_list = fetch_stock_list()
    if not stock_list.empty:
        X, y, features = create_training_data(stock_list)
        if X is not None:
            train_evaluate_and_save_model(X, y, features, args.n_jobs, args.n_iter, args.max_depth)

if __name__ == '__main__':
    main()
