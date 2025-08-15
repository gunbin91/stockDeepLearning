import pandas as pd
import numpy as np
import joblib
import os

# 모델 파일 경로
MODEL_PATH = 'stock_prediction_model_rf_upgraded.joblib'

def predict_with_ml_model(df):
    """
    학습된 RandomForest 모델을 사용하여 상승 확률을 예측합니다.
    결과로 '종목코드'와 'ml_pred_proba' 컬럼을 가진 데이터프레임을 반환합니다.
    """
    if df.empty:
        return pd.DataFrame(columns=['종목코드', 'ml_pred_proba'])

    # 결과 데이터프레임은 원본의 '종목코드'를 유지해야 함
    result_df = df[['종목코드']].copy()

    # 모델 파일 존재 여부 및 로드 확인
    if not os.path.exists(MODEL_PATH):
        print(f"분석 실패: 모델 파일('{MODEL_PATH}')을 찾을 수 없습니다. 모델 학습을 먼저 실행해주세요.")
        return None

    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        print(f"성공: 모델 파일('{MODEL_PATH}')을 정상적으로 로드했습니다.")
    except Exception as e:
        print(f"분석 실패: 모델 파일('{MODEL_PATH}')을 로드하는 중 문제가 발생했습니다: {e}")
        return None

    # --- 실제 모델 예측 ---
    # 학습에 사용된 피처가 현재 데이터프레임에 있는지 확인
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"경고: 예측에 필요한 피처가 부족합니다: {missing_features}. 예측을 건너뜁니다.")
        result_df['ml_pred_proba'] = 0.0
        return result_df

    # 피처 선택 및 예측
    X_pred = df[features]
    
    # predict_proba는 각 클래스에 대한 확률을 반환. [:, 1]은 '상승(1)' 클래스에 대한 확률을 선택.
    y_pred_proba = model.predict_proba(X_pred)[:, 1]
    
    result_df['ml_pred_proba'] = y_pred_proba
    
    return result_df