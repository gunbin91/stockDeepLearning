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
import warnings

# scikit-learn 버전 불일치 경고 억제
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except (ImportError, AttributeError):
    pass

# 호환성 패치를 최상단에서 적용 (cuDF/cuML import 전에 실행되어야 함)
# scikit-learn 호환성 패치: cuML이 BaseEstimator._get_default_requests를 사용하는데 최신 scikit-learn에서는 제거됨
def _apply_sklearn_compatibility_patch_early():
    """scikit-learn 최신 버전과 cuML 호환성을 위한 패치 적용 (모듈 로드 전)"""
    try:
        from sklearn.base import BaseEstimator
        # _get_default_requests가 없으면 추가 (cuML 호환성)
        if not hasattr(BaseEstimator, '_get_default_requests'):
            # scikit-learn 1.3+에서는 _get_metadata_request를 사용하거나, 없으면 빈 함수로 대체
            if hasattr(BaseEstimator, '_get_metadata_request'):
                # _get_metadata_request를 _get_default_requests로 별칭 생성
                original_get_metadata_request = BaseEstimator._get_metadata_request
                def _get_default_requests(self, *args, **kwargs):
                    return original_get_metadata_request(self, *args, **kwargs)
                BaseEstimator._get_default_requests = _get_default_requests
            else:
                # 둘 다 없으면 빈 함수로 대체
                def _get_default_requests(self, *args, **kwargs):
                    return {}
                BaseEstimator._get_default_requests = _get_default_requests
    except Exception:
        pass

# pandas 호환성 패치: cuDF가 pandas.api.types.is_interval을 사용하는데 최신 pandas에서는 제거됨
def _apply_pandas_compatibility_patch_early():
    """pandas 최신 버전과 cuDF 호환성을 위한 패치 적용 (모듈 로드 전)"""
    try:
        import pandas.api.types as pd_types
        # is_interval이 없으면 추가 (cuDF 호환성)
        if not hasattr(pd_types, 'is_interval'):
            # pandas 2.0+에서는 IntervalDtype을 사용하여 체크
            def is_interval(arr):
                """Interval 타입 체크 함수"""
                try:
                    from pandas import IntervalDtype
                    return hasattr(arr, 'dtype') and isinstance(arr.dtype, IntervalDtype)
                except:
                    return False
            pd_types.is_interval = is_interval
    except Exception:
        pass

# 모듈 로드 시점에 패치 적용
_apply_sklearn_compatibility_patch_early()
_apply_pandas_compatibility_patch_early()

from logger import log_info, log_warning, log_error, log_critical
from exceptions import ModelPredictionError, DataValidationError
from path_manager import path_manager

# 패치 함수를 외부에서도 사용할 수 있도록 별칭 제공
apply_pandas_compatibility_patch = _apply_pandas_compatibility_patch_early
apply_sklearn_compatibility_patch = _apply_sklearn_compatibility_patch_early

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
        # 호환성 패치 적용 (cuDF/cuML 로드 전에 실행)
        apply_pandas_compatibility_patch()
        apply_sklearn_compatibility_patch()
        
        # scikit-learn 버전 불일치 경고 억제 (모델 로드 시)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except (ImportError, AttributeError):
                pass
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
        elif is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'single_model':
            # cuML 단일 모델 처리
            model = model_data['model']
            features = model_data['features']
            scaler = model_data['scaler']
            imputation_values = model_data['imputation_values']
            log_info(f"   ✅ cuML 단일 모델 로드 완료")
        else:
            # 기존 모델 구조 (sklearn) 또는 model_type이 없는 경우
            model = model_data['model']
            features = model_data['features']
            scaler = model_data['scaler']
            imputation_values = model_data['imputation_values']
            
            # cuML 모델인지 확인 (타입으로 판단)
            if 'cuml' in str(type(model)).lower() or (hasattr(model, 'predict_proba') and 'cuml' in str(type(model).__module__).lower()):
                is_cuml_model = True
                log_info(f"   ✅ cuML 모델 로드 완료 (타입 감지) - {type(model).__name__}")
            else:
                is_cuml_model = False
                log_info(f"   ✅ sklearn 모델 로드 완료 - {type(model).__name__}")
            
    except Exception as e:
        error_msg = f"모델 파일('{model_path}')을 로드하는 중 문제가 발생했습니다: {e}"
        log_error(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")

    # 모델이 기대하는 피처만 선택 (데이터에 있는 피처만 사용)
    # 중요: 모델이 저장된 피처 순서를 정확히 따라야 함
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        log_warning(f"   ⚠️ 필요한 피처 부족: {len(missing_features)}개 - {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        if len(available_features) == 0:
            log_error("   ❌ 사용 가능한 피처가 없습니다. 예측을 수행할 수 없습니다.")
            result_df['ml_pred_proba'] = np.nan
            return result_df
        log_info(f"   ℹ️ 사용 가능한 피처 {len(available_features)}개로 예측 진행")
    
    # 모델의 피처 수 확인 (cuML 모델은 n_features_in_ 속성 확인)
    # 주의: cuML 모델은 내부적으로 학습 시 사용된 피처 수를 저장하므로 정확히 일치해야 함
    model_expected_features = None
    try:
        if hasattr(model, 'n_features_in_'):
            model_expected_features = model.n_features_in_
        elif hasattr(model, 'n_features_'):
            model_expected_features = model.n_features_
        # cuML 모델의 경우 내부 모델에서 확인
        elif hasattr(model, 'cuml_model') and hasattr(model.cuml_model, 'n_features_in_'):
            model_expected_features = model.cuml_model.n_features_in_
        # EnsembleModelWrapper의 경우 내부 모델들 확인
        elif hasattr(model, 'models') and len(model.models) > 0:
            first_model = model.models[0]
            if hasattr(first_model, 'n_features_in_'):
                model_expected_features = first_model.n_features_in_
    except Exception:
        pass  # 모델 피처 수 확인 실패는 무시
    
    # 추가 검증: 모델이 저장된 features 리스트와 실제 모델의 피처 수가 일치하는지 확인 (먼저 확인)
    if model_expected_features is not None and model_expected_features != len(features):
        log_error(f"   ❌ 심각한 불일치: 모델 내부는 {model_expected_features}개 피처를 기대하지만, 저장된 features 리스트는 {len(features)}개입니다.")
        log_error(f"   ❌ 저장된 features: {features}")
        log_error(f"   ❌ 이 모델은 피처 수가 불일치합니다. 모델을 다시 학습해야 합니다.")
        error_msg = f"모델 내부 피처 수({model_expected_features})와 저장된 features 리스트({len(features)})가 일치하지 않습니다. 모델을 다시 학습해야 합니다."
        log_critical(error_msg)
        raise ModelPredictionError(error_msg, model_name="RandomForest")
    
    # 모델이 기대하는 피처 수와 사용 가능한 피처 수가 일치하는지 확인
    if model_expected_features is not None and model_expected_features != len(available_features):
        log_error(f"   ❌ 모델 피처 수 불일치: 모델은 {model_expected_features}개를 기대하지만 데이터는 {len(available_features)}개입니다.")
        log_error(f"   ❌ 모델이 저장된 features 리스트: {len(features)}개 - {features}")
        log_error(f"   ❌ 사용 가능한 features: {len(available_features)}개 - {available_features}")
        if model_expected_features > len(available_features):
            error_msg = f"모델은 {model_expected_features}개 피처를 기대하지만 {len(available_features)}개만 제공되었습니다. 모델을 다시 학습해야 합니다."
            log_critical(error_msg)
            raise ModelPredictionError(error_msg, model_name="RandomForest")
        else:
            log_warning(f"   ⚠️ 모델이 기대하는 피처 수({model_expected_features})가 제공된 피처 수({len(available_features)})보다 적습니다. 예측을 시도하지만 오류가 발생할 수 있습니다.")
    
    # 사용 가능한 피처만 선택 (모델이 저장된 순서대로)
    X_pred = df[available_features].copy()
    
    # imputation_values도 사용 가능한 피처만 필터링
    available_imputation = {k: v for k, v in imputation_values.items() if k in available_features} if imputation_values else {}
    X_pred.fillna(available_imputation, inplace=True)
    
    # 스케일러의 피처 수 확인
    scaler_expected_features = None
    if scaler and hasattr(scaler, 'n_features_in_'):
        scaler_expected_features = scaler.n_features_in_
    elif scaler and hasattr(scaler, 'mean_'):
        # cuML StandardScaler는 mean_ 속성의 길이로 피처 수 확인 가능
        scaler_expected_features = len(scaler.mean_) if hasattr(scaler.mean_, '__len__') else None
    
    # 스케일러와 데이터의 피처 수가 일치하는지 확인
    use_scaler = scaler is not None
    if scaler_expected_features is not None and scaler_expected_features != len(available_features):
        log_warning(f"   ⚠️ 스케일러 피처 수 불일치: 스케일러는 {scaler_expected_features}개를 기대하지만 데이터는 {len(available_features)}개입니다. 스케일링을 건너뜁니다.")
        use_scaler = False

    try:
        # cuML scaler인지 확인
        if is_cuml_model and use_scaler and hasattr(scaler, 'transform'):
            # cuML scaler는 cuDF DataFrame을 받아야 함
            # 모델이 저장된 피처 순서대로 컬럼 정렬 (중요: 순서가 정확히 일치해야 함)
            try:
                import cudf
                X_pred_ordered = X_pred[available_features]
                X_pred_cudf = cudf.from_pandas(X_pred_ordered)
                X_pred_scaled_cudf = scaler.transform(X_pred_cudf)
                # cuML 모델은 cuDF DataFrame을 받을 수 있음
                X_pred_scaled = X_pred_scaled_cudf
            except ImportError:
                log_warning("   ⚠️ cuDF를 사용할 수 없습니다. 원본 데이터 사용")
                X_pred_ordered = X_pred[available_features]
                X_pred_scaled = cudf.from_pandas(X_pred_ordered) if is_cuml_model else X_pred
            except Exception as e:
                log_warning(f"   ⚠️ cuML 스케일링 오류: {e}, 원본 데이터 사용")
                X_pred_ordered = X_pred[available_features]
                X_pred_scaled = cudf.from_pandas(X_pred_ordered) if is_cuml_model else X_pred
        elif is_cuml_model:
            # 스케일러를 사용하지 않음 (cuML 모델은 cuDF DataFrame 필요)
            # 모델이 저장된 피처 순서대로 컬럼 정렬 (중요: 순서가 정확히 일치해야 함)
            import cudf
            X_pred_ordered = X_pred[available_features]
            X_pred_scaled = cudf.from_pandas(X_pred_ordered)
        elif use_scaler:
            # sklearn scaler (sklearn 모델)
            X_pred_scaled = scaler.transform(X_pred)
        else:
            # 스케일러를 사용하지 않음 (sklearn 모델)
            X_pred_scaled = X_pred.values
    except Exception as e:
        log_warning(f"   ⚠️ 스케일링 오류: {e}, 원본 데이터 사용")
        if is_cuml_model:
            import cudf
            # 모델이 저장된 피처 순서대로 컬럼 정렬
            X_pred_ordered = X_pred[available_features]
            X_pred_scaled = cudf.from_pandas(X_pred_ordered)
        else:
            X_pred_scaled = X_pred.values

    try:
        # 예측 시작 로그
        log_info(f"   🔄 모델 예측 실행 중... (입력 데이터: {len(X_pred_scaled):,}개)")
        
        # cuML 모델인 경우 래퍼가 자동으로 처리
        try:
            pred_proba = model.predict_proba(X_pred_scaled)
        except Exception as predict_error:
            # 예측 중 발생한 예외를 상세히 로깅
            error_detail = f"predict_proba 호출 중 오류: {type(predict_error).__name__}: {str(predict_error)}"
            log_error(f"   ❌ {error_detail}")
            log_error(f"   ❌ 입력 데이터 형태: {type(X_pred_scaled)}, 크기: {getattr(X_pred_scaled, 'shape', 'N/A')}")
            log_error(f"   ❌ 모델 타입: {type(model).__name__}")
            if hasattr(model, 'n_features_in_'):
                log_error(f"   ❌ 모델 기대 피처 수: {model.n_features_in_}")
            raise ModelPredictionError(f"모델 예측 실행 중 오류: {error_detail}", model_name="RandomForest")
        
        # 예측 결과 처리
        try:
            # 반환 형태에 따라 처리 (cuML 모델은 cuDF DataFrame 반환)
            if isinstance(pred_proba, np.ndarray):
                if pred_proba.ndim == 2:
                    y_pred_proba = pred_proba[:, 1]
                else:
                    y_pred_proba = pred_proba
            elif hasattr(pred_proba, 'iloc'):
                # cuDF DataFrame인 경우
                if hasattr(pred_proba.iloc[:, 1], 'to_pandas'):
                    y_pred_proba = pred_proba.iloc[:, 1].to_pandas().values
                elif hasattr(pred_proba.iloc[:, 1], 'to_numpy'):
                    y_pred_proba = pred_proba.iloc[:, 1].to_numpy()
                else:
                    y_pred_proba = pred_proba.iloc[:, 1].values
            else:
                # 기타 형태는 그대로 사용
                y_pred_proba = pred_proba[:, 1] if hasattr(pred_proba, '__getitem__') else pred_proba
            
            # 예측 결과 검증
            if y_pred_proba is None:
                raise ModelPredictionError("예측 결과가 None입니다.", model_name="RandomForest")
            
            if len(y_pred_proba) != len(result_df):
                log_warning(f"   ⚠️ 예측 결과 길이 불일치: 예측={len(y_pred_proba)}, 종목={len(result_df)}")
            
            # 예측 결과 통계
            avg_proba = np.mean(y_pred_proba)
            high_proba_count = np.sum(y_pred_proba > 0.7)
            log_info(f"   ✅ 예측 완료 (평균 확률: {avg_proba:.3f}, 고확률: {high_proba_count:,}개)")
            
        except Exception as process_error:
            error_detail = f"예측 결과 처리 중 오류: {type(process_error).__name__}: {str(process_error)}"
            log_error(f"   ❌ {error_detail}")
            log_error(f"   ❌ 예측 결과 타입: {type(pred_proba)}")
            raise ModelPredictionError(f"예측 결과 처리 중 오류: {error_detail}", model_name="RandomForest")
        
    except ModelPredictionError:
        # ModelPredictionError는 그대로 재발생
        raise
    except Exception as e:
        # 기타 예외는 ModelPredictionError로 변환
        error_msg = f"모델 예측 중 예상치 못한 오류 발생: {type(e).__name__}: {str(e)}"
        log_error(f"   ❌ {error_msg}")
        import traceback
        log_error(f"   ❌ 스택 트레이스:\n{traceback.format_exc()}")
        raise ModelPredictionError(error_msg, model_name="RandomForest")
    
    result_df['ml_pred_proba'] = y_pred_proba
    log_info("🎉 머신러닝 예측 완료!")
    
    return result_df

def predict_with_lgbm_model(df):
    """
    LightGBM 모델을 사용한 상승 확률 예측 함수
    
    Args:
        df: 분석할 종목 데이터가 포함된 데이터프레임
        
    Returns:
        pandas.DataFrame: 종목코드와 LGBM 예측 확률(lgbm_pred_proba)이 포함된 데이터프레임
    """
    if df.empty:
        log_warning("입력 데이터프레임이 비어있습니다 (LGBM).")
        return pd.DataFrame(columns=['종목코드', 'lgbm_pred_proba'])

    try:
        import lightgbm as lgb
    except ImportError:
        log_error("LightGBM 패키지가 설치되지 않았습니다.")
        return pd.DataFrame(columns=['종목코드', 'lgbm_pred_proba'])

    result_df = df[['종목코드']].copy()
    
    # 모델 경로
    model_dir = path_manager.data_dir
    lgbm_model_path = model_dir / 'lgbm_model.txt'
    lgbm_meta_path = model_dir / 'lgbm_model_metadata.joblib'
    
    if not lgbm_model_path.exists() or not lgbm_meta_path.exists():
        log_warning("LGBM 모델 파일이 없어 예측을 건너뜁니다.")
        result_df['lgbm_pred_proba'] = np.nan
        return result_df

    try:
        log_info(f"🤖 [LGBM] 모델 예측 중... ({len(df):,}개 종목)")
        
        # scikit-learn 버전 불일치 경고 억제 (모델 로드 시)
        with warnings.catch_warnings():
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except (ImportError, AttributeError):
                pass
            
            # 메타데이터 로드
            metadata = joblib.load(lgbm_meta_path)
        features = metadata['features']
        scaler = metadata.get('scaler')  # 스케일러 로드
        
        # 모델 로드
        model = lgb.Booster(model_file=str(lgbm_model_path))
        
        # 피처 준비 (순서 중요)
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            log_warning(f"   ⚠️ [LGBM] 피처 부족: {len(missing_features)}개")
            
        if not available_features:
            log_error("   ❌ [LGBM] 사용 가능한 피처가 없습니다.")
            result_df['lgbm_pred_proba'] = np.nan
            return result_df
            
        X_pred = df[available_features].copy()
        
        # LGBM은 NaN 처리가 가능하지만, 무한대 값은 처리 필요
        numeric_cols = X_pred.select_dtypes(include=[np.number]).columns
        X_pred[numeric_cols] = X_pred[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 스케일링 적용 (학습 시 사용한 스케일러 필수)
        if scaler:
            try:
                # NaN이 있으면 스케일러가 에러를 낼 수 있으므로, 
                # 학습 데이터의 중앙값 등으로 채워야 하지만, 
                # 여기서는 일단 0으로 채우거나 그대로 진행 (StandardScaler는 NaN 허용 안 함)
                # -> LightGBM은 NaN을 허용하지만 Scaler는 아님.
                # -> Scaler 사용 전 NaN 처리 필요.
                X_pred = X_pred.fillna(0) # 임시로 0으로 채움 (더 정교한 방법 필요할 수 있음)
                X_pred_scaled = scaler.transform(X_pred)
            except Exception as e:
                log_warning(f"   ⚠️ [LGBM] 스케일링 실패: {e}, 원본 데이터 사용")
                X_pred_scaled = X_pred
        else:
            X_pred_scaled = X_pred
        
        # 예측
        y_pred = model.predict(X_pred_scaled)
        
        result_df['lgbm_pred_proba'] = y_pred
        
        avg_proba = np.mean(y_pred)
        high_proba = np.sum(y_pred > 0.7)
        log_info(f"   ✅ [LGBM] 예측 완료 (평균: {avg_proba:.3f}, 고확률: {high_proba:,}개)")
        
        return result_df
        
    except Exception as e:
        log_error(f"   ❌ [LGBM] 예측 실패: {e}")
        import traceback
        log_error(traceback.format_exc())
        result_df['lgbm_pred_proba'] = np.nan
        return result_df
