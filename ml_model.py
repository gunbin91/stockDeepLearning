# ml_model.py

import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_prediction_model_rf_upgraded.joblib')

def predict_with_ml_model(df):
    """
    학습된 RandomForest 모델과 Scaler, Imputation 값을 사용하여 상승 확률을 예측합니다.
    """
    if df.empty:
        return pd.DataFrame(columns=['종목코드', 'ml_pred_proba'])

    result_df = df[['종목코드']].copy()

    if not os.path.exists(MODEL_PATH):
        print(f"분석 실패: 모델 파일('{MODEL_PATH}')을 찾을 수 없습니다. 모델 학습을 먼저 실행해주세요.")
        return None

    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
        # <<< ✨ 핵심 수정 1: 저장된 '대표 중앙값' 불러오기 ✨ >>>
        imputation_values = model_data['imputation_values']
        print(f"성공: 모델 파일('{MODEL_PATH}')과 스케일러, 중앙값을 정상적으로 로드했습니다.")
    except Exception as e:
        print(f"분석 실패: 모델 파일('{MODEL_PATH}')을 로드하는 중 문제가 발생했습니다: {e}")
        return None

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"경고: 예측에 필요한 피처가 부족합니다: {missing_features}. 예측을 건너뜁니다.")
        result_df['ml_pred_proba'] = np.nan # 예측 불가 시 NaN 반환
        return result_df

    X_pred = df[features].copy()
    
    # <<< ✨ 핵심 수정 2: 불러온 '대표 중앙값'으로 결측치 대체 ✨ >>>
    X_pred.fillna(imputation_values, inplace=True)

    try:
        X_pred_scaled = scaler.transform(X_pred)
    except Exception as e:
        print(f"데이터 스케일링 중 오류 발생: {e}")
        print("스케일링 없이 예측을 시도합니다. (정확도에 영향이 있을 수 있습니다.)")
        X_pred_scaled = X_pred

    y_pred_proba = model.predict_proba(X_pred_scaled)[:, 1]
    
    result_df['ml_pred_proba'] = y_pred_proba
    
    return result_df