"""
머신러닝 모델 모듈
=================

이 파일은 훈련된 머신러닝 모델을 사용하여 주식의 상승 확률을 예측합니다.
RandomForest 모델을 사용하여 15일 후 5% 이상 상승할 확률을 계산합니다.

주요 기능:
- 훈련된 모델 로드 (RandomForest)
- 데이터 전처리 (스케일링, 결측값 처리)
- 상승 확률 예측
- 예측 결과 반환
"""

import pandas as pd
import numpy as np
import joblib
import os
from logger import log_info, log_warning, log_error, log_critical
from exceptions import ModelPredictionError, DataValidationError

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_prediction_model_rf_upgraded.joblib')

def predict_with_ml_model(df):
    """
    머신러닝 모델을 사용한 상승 확률 예측 함수
    
    훈련된 RandomForest 모델을 사용하여 각 종목의 15일 후 5% 이상 상승할 확률을 예측합니다.
    데이터 전처리(스케일링, 결측값 처리)를 자동으로 수행합니다.
    
    Args:
        df: 분석할 종목 데이터가 포함된 데이터프레임
        
    Returns:
        pandas.DataFrame: 종목코드와 예측 확률이 포함된 데이터프레임
    """
    if df.empty:
        log_warning("입력 데이터프레임이 비어있습니다.")
        return pd.DataFrame(columns=['종목코드', 'ml_pred_proba'])

    # 입력 데이터 검증
    if '종목코드' not in df.columns:
        error_msg = "입력 데이터에 '종목코드' 컬럼이 없습니다."
        log_error(error_msg)
        raise DataValidationError(error_msg, field_name="종목코드")

    result_df = df[['종목코드']].copy()

    if not os.path.exists(MODEL_PATH):
        error_msg = f"모델 파일('{MODEL_PATH}')을 찾을 수 없습니다. 모델 학습을 먼저 실행해주세요."
        log_critical(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")

    try:
        log_info(f"🤖 머신러닝 모델 예측 중... ({len(df):,}개 종목)")
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
        imputation_values = model_data['imputation_values']
    except Exception as e:
        error_msg = f"모델 파일('{MODEL_PATH}')을 로드하는 중 문제가 발생했습니다: {e}"
        log_error(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        log_warning(f"   ⚠️ 필요한 피처 부족: {len(missing_features)}개")
        result_df['ml_pred_proba'] = np.nan
        return result_df

    X_pred = df[features].copy()
    X_pred.fillna(imputation_values, inplace=True)

    try:
        X_pred_scaled = scaler.transform(X_pred)
    except Exception as e:
        log_warning("   ⚠️ 스케일링 오류, 원본 데이터 사용")
        X_pred_scaled = X_pred

    try:
        y_pred_proba = model.predict_proba(X_pred_scaled)[:, 1]
        
        # 예측 결과 통계
        avg_proba = np.mean(y_pred_proba)
        high_proba_count = np.sum(y_pred_proba > 0.7)
        log_info(f"   ✅ 예측 완료 (평균 확률: {avg_proba:.3f}, 고확률: {high_proba_count:,}개)")
        
    except Exception as e:
        error_msg = f"모델 예측 중 오류 발생: {e}"
        log_error(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")
    
    result_df['ml_pred_proba'] = y_pred_proba
    log_info("🎉 머신러닝 예측 완료!")
    
    return result_df