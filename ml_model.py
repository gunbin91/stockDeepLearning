<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:57dcd888f24974f58c800c4404aa893a1097927be9d9703468d9b438906dabf3
size 2060
=======
import pandas as pd
import numpy as np
import joblib
import os

# 모델 파일 경로
MODEL_PATH = 'stock_prediction_model_rf.joblib'

def predict_with_ml_model(df):
    """
    학습된 RandomForest 모델을 사용하여 상승 확률을 예측합니다.

    [책임]
    - 저장된 모델 로드
    - 모델 입력에 맞게 데이터 준비 (피처 선택)
    - RandomForest 모델로 예측 수행
    - 예측 결과를 'ml_pred_proba' 컬럼으로 추가
    """
    predicted_df = df.copy()

    # 모델 파일 존재 여부 확인
    if not os.path.exists(MODEL_PATH):
        print(f"경고: 모델 파일('{MODEL_PATH}')을 찾을 수 없습니다. 예측을 건너뛰고 더미 데이터를 반환합니다.")
        predicted_df['ml_pred_proba'] = np.random.uniform(30, 70, len(predicted_df)).round(2)
        return predicted_df

    # 모델 로드 (모델과 피처 목록 포함)
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
    except Exception as e:
        print(f"에러: 모델 파일('{MODEL_PATH}')을 로드하는 중 문제가 발생했습니다: {e}")
        predicted_df['ml_pred_proba'] = np.random.uniform(30, 70, len(predicted_df)).round(2)
        return predicted_df

    # --- 실제 모델 예측 ---
    # 학습에 사용된 피처가 현재 데이터프레임에 있는지 확인
    missing_features = [f for f in features if f not in predicted_df.columns]
    if missing_features:
        print(f"경고: 예측에 필요한 피처가 부족합니다: {missing_features}. 예측을 건너뜁니다.")
        predicted_df['ml_pred_proba'] = 0.0
        return predicted_df

    # 피처 선택 및 예측
    X_pred = predicted_df[features]
    
    # predict_proba는 각 클래스에 대한 확률을 반환. [:, 1]은 '상승(1)' 클래스에 대한 확률을 선택.
    y_pred_proba = model.predict_proba(X_pred)[:, 1]
    
    predicted_df['ml_pred_proba'] = (y_pred_proba * 100).round(2)
    
    return predicted_df
>>>>>>> origin/window
