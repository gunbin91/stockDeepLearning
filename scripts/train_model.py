import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import warnings
import argparse
from datetime import datetime, timedelta
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_cacher 

warnings.filterwarnings('ignore', category=FutureWarning)

def create_training_data(period_days=365*4):
    print("캐시 관리 모듈을 통해 학습 데이터 생성을 시작합니다...")
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=period_days)).strftime('%Y-%m-%d')
    
    final_df = data_cacher.get_preprocessed_data(start_date, end_date)
    
    if final_df is None or final_df.empty:
        print("데이터를 가져오는 데 실패했습니다.")
        return None, None, None

    print(f"\n--- 생성된 학습 데이터 요약 ---")
    print(f"1. 전체 수집 데이터 (Raw): {len(final_df):,} 행")
    
    training_start_date = (today - timedelta(days=365*3)).strftime('%Y-%m-%d')
    final_df = final_df[final_df['date'] >= pd.to_datetime(training_start_date)]
    print(f"2. 워밍업 기간 제외 후: {len(final_df):,} 행")

    # <<< 핵심 수정: features 리스트에 거시 경제 지표 추가 >>>
    features = [
        # 기존 피처
        '수익률(1W)', '수익률(2W)', '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 'log_mktcap', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', '거래대금_MA20', '단기 정배열', '52주_신고가_비율',
        # 새로 추가된 거시 경제 피처
        'KOSPI_pct_1d', 'KOSPI_pct_5d', 
        'USDKRW_pct_1d', 'USDKRW_pct_5d', 
        'VIX_pct_1d', 'VIX_pct_5d'
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
    print("---------------------------------")

    print("학습 데이터 생성 완료!")
    return X, y, features

def train_evaluate_and_save_model(X, y, features, n_jobs, n_iter, max_depth_list, model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_prediction_model_rf_upgraded.joblib')):
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
    
    X, y, features = create_training_data()
    if X is not None:
        train_evaluate_and_save_model(X, y, features, args.n_jobs, args.n_iter, args.max_depth)

if __name__ == '__main__':
    main()