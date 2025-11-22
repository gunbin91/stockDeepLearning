"""
머신러닝 모델 모듈
=================

이 파일은 훈련된 머신러닝 모델을 사용하여 주식의 상승 확률을 예측합니다.
RandomForest 모델을 사용하여 15일 후 5% 이상 상승할 확률을 계산합니다.

주요 기능:
- 훈련된 모델 로드 (RandomForest / cuML 앙상블)
- 데이터 전처리 (스케일링, 결측값 처리)
- 상승 확률 예측
- 예측 결과 반환
"""

import pandas as pd
import numpy as np
import joblib
import os
import gc
from logger import log_info, log_warning, log_error, log_critical
from exceptions import ModelPredictionError, DataValidationError
from path_manager import path_manager

# cuML 모델 경로 (우선 사용)
CUML_MODEL_PATH = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
# 기존 모델 경로 (fallback)
LEGACY_MODEL_PATH = str(path_manager.get_model_path())

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

    # cuML 모델 파일 우선 확인, 없으면 기존 모델 파일 확인
    model_path = None
    is_cuml_model = False
    
    if os.path.exists(CUML_MODEL_PATH):
        model_path = CUML_MODEL_PATH
        is_cuml_model = True
        log_info(f"📦 cuML 앙상블 모델 파일 발견: {CUML_MODEL_PATH}")
    elif os.path.exists(LEGACY_MODEL_PATH):
        model_path = LEGACY_MODEL_PATH
        is_cuml_model = False
        log_info(f"📦 기존 모델 파일 발견: {LEGACY_MODEL_PATH}")
    else:
        error_msg = f"모델 파일을 찾을 수 없습니다. (cuML: {CUML_MODEL_PATH}, 기존: {LEGACY_MODEL_PATH})"
        log_critical(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")

    try:
        log_info(f"🤖 머신러닝 모델 예측 중... ({len(df):,}개 종목)")
        model_data = joblib.load(model_path)
        
        # cuML 앙상블 모델인지 확인
        if is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'mini_batch_ensemble':
            # cuML 앙상블 모델 처리
            from ml_model_wrapper import EnsembleModelWrapper
            models = model_data['models']
            scaler = model_data['scaler']
            features = model_data['features']
            imputation_values = model_data['imputation_values']
            
            # 앙상블 모델 래퍼 생성
            model = EnsembleModelWrapper(models, scaler)
            log_info(f"   ✅ cuML 앙상블 모델 로드 완료 ({len(models)}개 모델)")
        else:
            # 단일 모델 구조 (cuML 또는 sklearn)
            model = model_data['model']
            features = model_data['features']
            scaler = model_data['scaler']
            imputation_values = model_data['imputation_values']
            model_type_str = model_data.get('model_type', 'unknown')
            if model_type_str == 'single_model':
                log_info(f"   ✅ cuML 단일 모델 로드 완료")
            else:
                log_info(f"   ✅ 기존 모델 로드 완료")
            
    except Exception as e:
        error_msg = f"모델 파일('{model_path}')을 로드하는 중 문제가 발생했습니다: {e}"
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
        # cuML scaler인지 확인
        if is_cuml_model and hasattr(scaler, 'transform'):
            # cuML scaler는 cuDF DataFrame을 받아야 함
            try:
                import cudf
                X_pred_cudf = cudf.from_pandas(X_pred)
                X_pred_scaled_cudf = scaler.transform(X_pred_cudf)
                # cuDF를 pandas로 변환
                if hasattr(X_pred_scaled_cudf, 'to_pandas'):
                    X_pred_scaled = X_pred_scaled_cudf.to_pandas().values
                elif hasattr(X_pred_scaled_cudf, 'values'):
                    X_pred_scaled = X_pred_scaled_cudf.values
                else:
                    X_pred_scaled = X_pred_scaled_cudf
            except ImportError:
                log_warning("   ⚠️ cuDF를 사용할 수 없습니다. 원본 데이터 사용")
                X_pred_scaled = X_pred
            except Exception as e:
                log_warning(f"   ⚠️ cuML 스케일링 오류: {e}, 원본 데이터 사용")
                X_pred_scaled = X_pred
        else:
            # sklearn scaler
            X_pred_scaled = scaler.transform(X_pred)
    except Exception as e:
        log_warning(f"   ⚠️ 스케일링 오류: {e}, 원본 데이터 사용")
        X_pred_scaled = X_pred

    try:
        # cuML 모델인 경우 래퍼가 자동으로 처리
        pred_proba = model.predict_proba(X_pred_scaled)
        
        # 반환 형태에 따라 처리
        if isinstance(pred_proba, np.ndarray):
            if pred_proba.ndim == 2:
                y_pred_proba = pred_proba[:, 1]
            else:
                y_pred_proba = pred_proba
        else:
            # 기타 형태는 그대로 사용
            y_pred_proba = pred_proba[:, 1] if hasattr(pred_proba, '__getitem__') else pred_proba
        
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