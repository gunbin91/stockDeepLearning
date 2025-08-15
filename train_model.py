import pandas as pd
import numpy as np
import os
from pykrx import stock
from datetime import datetime, timedelta
import time
import joblib
import pandas_ta as ta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import concurrent.futures
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. 데이터 수집 및 피처 엔지니어링 ---
def fetch_and_process_ticker_data(ticker, start_date, end_date):
    """한 종목의 데이터를 수집하고 기술적/기본적 피처를 생성합니다."""
    try:
        df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if len(df_price) < 120:
            return None

        df_fundamental = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
        
        df = pd.merge(df_price, df_fundamental[['BPS', 'PER', 'PBR', 'EPS', 'DIV']], left_index=True, right_index=True, how='left')
        df.ffill(inplace=True)

        df['target'] = (df['종가'].shift(-20) / df['종가'] > 1.05).astype(int)
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래량변화율'] = df['거래량'].pct_change(periods=20)
        df['거래량MA5'] = df['거래량'].rolling(window=5).mean()
        df['거래량MA20'] = df['거래량'].rolling(window=20).mean()

        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        df.ta.sma(close='종가', length=5, append=True)
        df.ta.sma(close='종가', length=20, append=True)
        df.ta.sma(close='종가', length=60, append=True)
        df.ta.bbands(close='종가', length=20, std=2, append=True)
        
        df['ROE'] = df['PBR'] / df['PER']
        df['종목코드'] = ticker
        return df
    except Exception:
        return None

# <<< 2차 수정사항: data_fetcher.py와 동일한 데이터 처리 로직 적용 >>>
def create_training_data(today_str, period_days=365*3):
    """3년간의 데이터를 기반으로 학습 데이터를 병렬로 생성합니다."""
    print("향상된 피처로 학습 데이터 생성을 시작합니다...")
    
    today = datetime.strptime(today_str, '%Y%m%d')
    start_date = (today - timedelta(days=period_days)).strftime('%Y%m%d')
    
    tickers_kospi = stock.get_market_ticker_list(market='KOSPI', date=today_str)
    tickers_kosdaq = stock.get_market_ticker_list(market='KOSDAQ', date=today_str)
    all_tickers = list(set(tickers_kospi + tickers_kosdaq))
    print(f"총 {len(all_tickers)}개 종목에 대한 데이터 생성을 시도합니다.")

    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_and_process_ticker_data, ticker, start_date, today_str): ticker for ticker in all_tickers}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tickers), desc="피처 생성"):
            result_df = future.result()
            if result_df is not None:
                all_data.append(result_df)

    if not all_data:
        print("생성된 데이터가 없습니다.")
        return None, None, None

    final_df = pd.concat(all_data)
    
    # 무한대 값(inf)을 NaN으로 변경
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 타겟 변수의 결측치(NaN)는 예측이 불가능하므로 해당 행을 제거
    final_df.dropna(subset=['target'], inplace=True)

    # 피처들의 결측치를 각 피처의 중앙값(median)으로 채움
    final_df.fillna(final_df.median(numeric_only=True), inplace=True)
    
    # 그래도 남아있는 결측치가 있다면 0으로 채움 (안전장치)
    final_df.fillna(0, inplace=True);

    final_df.reset_index(drop=True, inplace=True)

    features = [
        '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 
        '거래량변화율', '거래량MA5', '거래량MA20', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'SMA_5', 'SMA_20', 'SMA_60',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
    ]
    target = 'target'
    
    features = [f for f in features if f in final_df.columns]
    
    X = final_df[features].astype(np.float32)
    y = final_df[target]
    
    print("학습 데이터 생성 완료!")
    return X, y, features

# --- 2. 모델 학습, 튜닝 및 평가 (이하 부분은 기존과 동일) ---
def train_evaluate_and_save_model(X, y, features, n_jobs=2, n_iter=10, model_path='stock_prediction_model_rf_upgraded.joblib'):
    if X is None or y is None or X.empty or y.empty:
        print("학습 데이터가 없어 모델링을 건너뜁니다.")
        return

    print("모델 학습 및 평가를 시작합니다...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"학습 데이터 타겟 분포:\n{y_train.value_counts(normalize=True)}")

    print("\nSMOTE를 적용하여 클래스 불균형을 처리합니다...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE 적용 후 학습 데이터 타겟 분포:\n{y_train_res.value_counts(normalize=True)}")

    print("\nRandomizedSearchCV를 사용하여 최적의 하이퍼파라미터를 탐색합니다...")
    # 메모리 오류 방지를 위해 하이퍼파라미터 범위 조정
    param_dist = {
        'n_estimators': randint(100, 200),
        'max_depth': [10, 20, 30], # None 제거
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=n_iter, # 사용자 입력값으로 변경
        cv=3, 
        scoring='roc_auc', 
        verbose=2, 
        n_jobs=n_jobs,
        random_state=42
    )
    
    random_search.fit(X_train_res, y_train_res)
    
    print(f"\n최적 하이퍼파라미터: {random_search.best_params_}")
    best_model = random_search.best_estimator_

    print("\n최적 모델을 테스트 데이터로 평가합니다...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- 최종 모델 평가 결과 ---")
    print(f"정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"정밀도 (Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"재현율 (Recall): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 점수 (F1 Score): {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n분류 보고서 (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    joblib.dump({'model': best_model, 'features': features}, model_path)
    print(f"\n업그레이드된 모델을 '{model_path}' 경로에 저장했습니다.")

def load_and_predict(input_data, model_path='stock_prediction_model_rf_upgraded.joblib'):
    try:
        loaded_data = joblib.load(model_path)
        model = loaded_data['model']
        features = loaded_data['features']
        input_data_aligned = input_data[features]
        probabilities = model.predict_proba(input_data_aligned)
        return probabilities
    except FileNotFoundError:
        print(f"에러: '{model_path}'에서 모델을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"예측 중 에러 발생: {e}")
        return None

if __name__ == '__main__':
    # --- 모델 학습 실행 ---
    total_cores = os.cpu_count()
    default_jobs = max(1, total_cores // 2)
    
    while True:
        try:
            user_input = input(f"사용할 CPU 코어 수를 입력하세요 (최대: {total_cores}, 추천/기본값: {default_jobs}): ")
            if user_input == "":
                num_jobs = default_jobs
                break
            num_jobs = int(user_input)
            if 0 < num_jobs <= total_cores:
                break
            else:
                print(f"에러: 1과 {total_cores} 사이의 값을 입력해야 합니다.")
        except ValueError:
            print("에러: 숫자를 입력해야 합니다.")

    # n_iter 값 입력 받기
    default_n_iter = 10
    while True:
        try:
            user_input = input(f"RandomizedSearchCV의 n_iter 값을 입력하세요 (추천/기본값: {default_n_iter}): ")
            if user_input == "":
                num_iter = default_n_iter
                break
            num_iter = int(user_input)
            if num_iter > 0:
                break
            else:
                print("에러: 1 이상의 값을 입력해야 합니다.")
        except ValueError:
            print("에러: 숫자를 입력해야 합니다.")

    # <<< 1차 수정사항: 안정적으로 최신 영업일 찾는 로직 >>>
    print("\n가장 최근의 영업일을 탐색합니다...")
    try:
        today_str = stock.get_nearest_business_day_in_a_week(prev=True)
        print(f"성공: 가장 최근 영업일은 '{today_str}' 입니다.")
    except Exception as e:
        print(f"에러: 가장 최근 영업일을 찾지 못했습니다. {e}. 모델 학습을 중단합니다.")
        exit()

    X, y, features = create_training_data(today_str)

    if X is not None and y is not None:
        train_evaluate_and_save_model(X, y, features, n_jobs=num_jobs, n_iter=num_iter)
        print("\n--- 저장된 모델 로드 및 예측 테스트 ---")
        sample_data = X.head()
        predictions = load_and_predict(sample_data)
        if predictions is not None:
            print("샘플 데이터에 대한 예측 확률 (0일 확률, 1일 확률):")
            print(predictions)
    else:
        print("학습 데이터 생성에 실패하여 모델링을 진행하지 않습니다.")