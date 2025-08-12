<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:2e93c896e9a17d1d48d68789a045d910751c2fe5d37d737467dc918b8c93c899
size 5446
=======
import pandas as pd
import numpy as np
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
        # 가격 데이터 (OHLCV)
        df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if len(df_price) < 120:  # 최소 6개월 데이터 확보
            return None

        # 기본적 분석 데이터 (PER, PBR, ROE 등)
        df_fundamental = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
        
        # 데이터 병합
        df = pd.merge(df_price, df_fundamental[['BPS', 'PER', 'PBR', 'EPS', 'DIV']], left_index=True, right_index=True, how='left')
        df.ffill(inplace=True) # 누락된 기본적 데이터는 이전 값으로 채움

        # --- 피처 엔지니어링 ---
        # 1. 타겟 변수: 20 거래일 후 5% 이상 상승 시 1, 아니면 0
        df['target'] = (df['종가'].shift(-20) / df['종가'] > 1.05).astype(int)

        # 2. 수익률 및 변동성
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()

        # 3. 거래량 관련 피처
        df['거래량변화율'] = df['거래량'].pct_change(periods=20)
        df['거래량MA5'] = df['거래량'].rolling(window=5).mean()
        df['거래량MA20'] = df['거래량'].rolling(window=20).mean()

        # 4. 기술적 지표 (pandas_ta 라이브러리 활용)
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        df.ta.sma(close='종가', length=5, append=True)
        df.ta.sma(close='종가', length=20, append=True)
        df.ta.sma(close='종가', length=60, append=True)
        
        # 5. ROE (PBR과 PER로 계산)
        df['ROE'] = df['PBR'] / df['PER']
        
        df['종목코드'] = ticker
        return df

    except Exception:
        return None

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
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)
    final_df.reset_index(inplace=True)

    features = [
        '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 
        '거래량변화율', '거래량MA5', '거래량MA20', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'SMA_5', 'SMA_20', 'SMA_60'
    ]
    target = 'target'
    
    X = final_df[features]
    y = final_df[target]
    
    print("학습 데이터 생성 완료!")
    return X, y, features

# --- 2. 모델 학습, 튜닝 및 평가 ---
def train_evaluate_and_save_model(X, y, features, model_path='stock_prediction_model_rf_upgraded.joblib'):
    """모델 학습, SMOTE, GridSearchCV, 평가 및 저장을 수행합니다."""
    if X is None or y is None or X.empty or y.empty:
        print("학습 데이터가 없어 모델링을 건너뜁니다.")
        return

    print("모델 학습 및 평가를 시작합니다...")
    
    # 1. 데이터 분할 (학습 70%, 테스트 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    print(f"학습 데이터 타겟 분포:\n{y_train.value_counts(normalize=True)}")

    # 2. SMOTE를 사용한 오버샘플링 (학습 데이터에만 적용)
    print("\nSMOTE를 적용하여 클래스 불균형을 처리합니다...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE 적용 후 학습 데이터 타겟 분포:\n{y_train_res.value_counts(normalize=True)}")

    # 3. RandomizedSearchCV를 사용한 하이퍼파라미터 튜닝
    print("\nRandomizedSearchCV를 사용하여 최적의 하이퍼파라미터를 탐색합니다...")
    # 탐색할 파라미터 범위를 지정합니다.
    param_dist = {
        'n_estimators': randint(100, 300),       # 100~299 사이에서 무작위 정수 선택
        'max_depth': [10, 20, 30, 40, None],    # 리스트 중에서 무작위 선��
        'min_samples_split': randint(2, 11),     # 2~10 사이에서 무작위 정수 선택
        'min_samples_leaf': randint(1, 5),       # 1~4 사이에서 무작위 정수 선택
        'max_features': ['sqrt', 'log2']         # 리스트 중에서 무작위 선택
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # n_iter: 몇 개의 조합을 테스트할지 결정 (예: 20개)
    # random_state: 재현 가능성을 위해 추가
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=20, 
        cv=3, 
        scoring='roc_auc', 
        verbose=2, 
        n_jobs=-1, 
        random_state=42
    )
    
    # 진행 상황을 명확하게 보여주기 위한 로그 추가
    print(f"\n총 {random_search.n_iter}개의 파라미터 조합을 탐색하며, 각 조합마다 {random_search.cv}번의 교차 검증을 수행합니다.")
    print(f"따라서 총 {random_search.n_iter * random_search.cv}번의 모델 학습이 진행됩니다.")
    print("학습을 시작합니다. verbose=2 설정으로 인해 각 모델의 학습 진행 상황이 곧 출력됩니다...")
    
    random_search.fit(X_train_res, y_train_res)
    
    print("\n하이퍼파라미터 탐색이 완료되었습니다.")
    print(f"최적 하이퍼파라미터: {random_search.best_params_}")
    best_model = random_search.best_estimator_

    # 4. 테스트 데이터로 최종 모델 평가
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
    
    # 5. 모델과 피처 목록 저장
    joblib.dump({'model': best_model, 'features': features}, model_path)
    print(f"\n업그레이드된 모델을 '{model_path}' 경로에 저장했습니다.")

# --- 3. 저장된 모델 로드 및 예측 ---
def load_and_predict(input_data, model_path='stock_prediction_model_rf_upgraded.joblib'):
    """저장된 모델을 불러와 새로운 데이터에 대한 예측 확률을 반환합니다."""
    try:
        loaded_data = joblib.load(model_path)
        model = loaded_data['model']
        features = loaded_data['features']
        
        # 입력 데이터가 피처 순서를 따르도록 정렬
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
    today_str = stock.get_nearest_business_day_in_a_week()
    X, y, features = create_training_data(today_str)

    if X is not None and y is not None:
        train_evaluate_and_save_model(X, y, features)

        # --- 모델 로드 및 예측 예시 ---
        print("\n--- 저장된 모델 로드 및 예측 테스트 ---")
        # 테스트 데이터의 첫 5개 샘플로 예측 시뮬레이션
        sample_data = X.head()
        predictions = load_and_predict(sample_data)
        if predictions is not None:
            print("샘플 데이터에 대한 예측 확률 (0일 확률, 1일 확률):")
            print(predictions)
    else:
        print("학습 데이터 생성에 실패하여 모델링을 진행하지 않습니다.")
>>>>>>> origin/window
