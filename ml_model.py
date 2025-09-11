# ml_model.py

import pandas as pd
import numpy as np
import joblib
import os
from logger import log_info, log_warning, log_error, log_critical
from exceptions import ModelPredictionError, DataValidationError

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_prediction_model_rf_upgraded.joblib')

def predict_with_ml_model(df):
    """
    학습된 RandomForest 모델과 Scaler, Imputation 값을 사용하여 상승 확률을 예측합니다.
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
        log_info("🤖 머신러닝 모델 예측을 시작합니다...")
        log_info(f"   📁 모델 파일 로딩 시작: {MODEL_PATH}")
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
        # <<< ✨ 핵심 수정 1: 저장된 '대표 중앙값' 불러오기 ✨ >>>
        imputation_values = model_data['imputation_values']
        log_info(f"   ✅ 모델 로딩 완료: RandomForest 모델, {len(features)}개 피처")
        log_info(f"   🔧 스케일러 및 결측값 대체값 로딩 완료")
    except Exception as e:
        error_msg = f"모델 파일('{MODEL_PATH}')을 로드하는 중 문제가 발생했습니다: {e}"
        log_error(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        warning_msg = f"예측에 필요한 피처가 부족합니다: {missing_features}. 예측을 건너뜁니다."
        log_warning(warning_msg)
        result_df['ml_pred_proba'] = np.nan # 예측 불가 시 NaN 반환
        return result_df

    log_info(f"   📊 입력 데이터 검증 완료: {len(df):,}개 종목")
    X_pred = df[features].copy()
    
    # <<< ✨ 핵심 수정 2: 불러온 '대표 중앙값'으로 결측치 대체 ✨ >>>
    log_info("   🔧 결측값 대체 중...")
    X_pred.fillna(imputation_values, inplace=True)

    try:
        log_info("   📏 데이터 스케일링 시작...")
        X_pred_scaled = scaler.transform(X_pred)
        log_info("   ✅ 데이터 스케일링 완료")
    except Exception as e:
        warning_msg = f"데이터 스케일링 중 오류 발생: {e}. 스케일링 없이 예측을 시도합니다. (정확도에 영향이 있을 수 있습니다.)"
        log_warning(warning_msg)
        X_pred_scaled = X_pred

    try:
        log_info("   🎯 RandomForest 모델 예측 시작...")
        y_pred_proba = model.predict_proba(X_pred_scaled)[:, 1]
        log_info(f"   ✅ 모델 예측 완료: {len(y_pred_proba):,}개 종목의 상승 확률 계산")
        
        # 예측 결과 통계
        avg_proba = np.mean(y_pred_proba)
        high_proba_count = np.sum(y_pred_proba > 0.7)
        log_info(f"   📈 예측 결과: 평균 상승 확률 {avg_proba:.3f}, 고확률 종목 {high_proba_count:,}개")
        
    except Exception as e:
        error_msg = f"모델 예측 중 오류 발생: {e}"
        log_error(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")
    
    result_df['ml_pred_proba'] = y_pred_proba
    log_info("🎉 머신러닝 모델 예측이 완료되었습니다!")
    
    return result_df