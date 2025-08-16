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
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
import concurrent.futures
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# ⚠️ [필수] data_fetcher.py와 동일한 DART API 인증키를 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522"
# -----------------------------------------------------------------------------

def get_financial_data_for_training_http(corp_codes, start_year, end_year):
    """HTTP 통신으로 여러 연도의 재무 데이터를 수집합니다."""
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
    """FinanceDataReader를 사용하여 KOSPI와 KOSDAQ의 전 종목 리스트와 시가총액 정보를 수집합니다."""
    print("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보를 수집합니다 (KRX-MARCAP)...")
    try:
        # KRX-MARCAP을 사용하여 시가총액, 상장주식수 정보를 가져옴
        df_marcap = fdr.StockListing('KRX-MARCAP') # Defaults to latest date
        
        # 스팩, 리츠 등 제외
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠')].copy()
        
        # 필요한 컬럼 선택 및 이름 변경
        # 'Stocks' 컬럼이 상장주식수를 의미함
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        
        # 상장주식수가 0인 경우 제외 (관리종목 등)
        stock_list = stock_list[stock_list['상장주식수'] > 0]

        print(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        print(f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}")
        return pd.DataFrame(columns=['종목코드', '종목명', '상장주식수'])


def fetch_and_process_ticker_data(stock_info, start_date, end_date, all_fs_data):
    # 이 함수는 data_fetcher.py와 동일한 로직 + target 변수 생성
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 120: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year
        
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        
        df[['당기순이익', '자본총계']] = df[['당기순이익', '자본총계']].ffill().bfill()
        if df[['당기순이익', '자본총계']].isnull().values.any(): return None

        df['시가총액'] = df['종가'] * shares
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        df['target'] = (df['종가'].shift(-20) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        return df
    except Exception:
        return None

def create_training_data(stock_list, period_days=365*3):
    print("안정적인 API(FDR, DART) 기반으로 학습 데이터 생성을 시작합니다...")
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=period_days)).strftime('%Y-%m-%d')
    
    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError: return None, None, None
    
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()

    all_fs_data = get_financial_data_for_training_http(corp_codes, today.year - 4, today.year - 1)
    if not all_fs_data: return None, None, None

    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_and_process_ticker_data, row, start_date, end_date, all_fs_data): row
            for row in target_stocks.to_dict('records')
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_stocks), desc="학습 데이터 생성"):
            result_df = future.result()
            if result_df is not None:
                all_data.append(result_df)

    if not all_data: return None, None, None

    final_df = pd.concat(all_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features = [
        '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    ]
    target = 'target'
    
    final_df.dropna(subset=[target] + features, inplace=True)
    
    X = final_df[features].astype(np.float32)
    y = final_df[target]
    
    print("학습 데이터 생성 완료!")
    return X, y, features

def train_evaluate_and_save_model(X, y, features, n_jobs, n_iter, max_depth_list, model_path='stock_prediction_model_rf_upgraded.joblib'):
    """RandomizedSearchCV를 사용하여 모델을 학습, 평가 및 저장합니다."""
    if X is None or y is None or X.empty or y.empty:
        print("학습 데이터가 없어 모델링을 건너뜁니다.")
        return

    print("모델 학습 및 평가를 시작합니다...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\nSMOTE 적용 전 학습 데이터 타겟 분포:\n", y_train.value_counts(normalize=True))
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("\nSMOTE 적용 후 학습 데이터 타겟 분포:\n", y_train_res.value_counts(normalize=True))

    # RandomizedSearchCV를 위한 파라미터 분포 설정
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': max_depth_list,
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20)
    }

    print("\nRandomizedSearchCV를 사용하여 최적 파라미터 탐색...")
    print("탐색 대상 파라미터 분포:")
    print(f"- n_estimators: 100 ~ 500 사이의 임의의 값")
    print(f"- max_depth: {max_depth_list} 중에서 선택")
    print(f"- min_samples_split: 2 ~ 20 사이의 임의의 값")
    print(f"- min_samples_leaf: 1 ~ 20 사이의 임의의 값")

    model = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV 설정
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=n_iter, cv=3, n_jobs=n_jobs, 
                                       verbose=2, random_state=42, scoring='roc_auc')

    random_search.fit(X_train_res, y_train_res)

    print("\n--- 최적 파라미터 탐색 결과 ---")
    print(f"최고 점수 (ROC-AUC): {random_search.best_score_:.4f}")
    print("최적 파라미터:", random_search.best_params_)

    best_model = random_search.best_estimator_

    print("\n최적 모델로 테스트 데이터 평가...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- 최종 모델 평가 결과 ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n분류 보고서 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['하락(0)', '상승(1)']))

    joblib.dump({'model': best_model, 'features': features}, model_path)
    print(f"\n✅ 새로운 데이터로 학습된 최적 모델을 '{model_path}' 경로에 저장했습니다.")

def main():
    """스크립트 실행을 위한 메인 함수"""
    parser = argparse.ArgumentParser(description="RandomForest 모델 학습 및 RandomizedSearchCV를 이용한 하이퍼파라미터 튜닝")

    # CPU 코어 수 설정
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='RandomizedSearchCV에서 사용할 CPU 코어 수 (-1은 모든 코어 사용)')

    # RandomizedSearchCV 반복 횟수 설정
    parser.add_argument('--n_iter', type=int, default=10,
                        help='RandomizedSearchCV에서 시도할 파라미터 조합의 수')

    # max_depth 리스트 설정
    parser.add_argument('--max_depth', type=int, nargs='+', default=[10, 20, 30],
                        help='max_depth 파라미터 후보 리스트 (예: 10 20 30)')

    args = parser.parse_args()

    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키를 train_model.py 파일에 입력해주세요.")
        exit()

    stock_list = fetch_stock_list()
    if not stock_list.empty:
        X, y, features = create_training_data(stock_list)
        if X is not None:
            train_evaluate_and_save_model(X, y, features, args.n_jobs, args.n_iter, args.max_depth)

if __name__ == '__main__':
    main()