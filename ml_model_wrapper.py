"""
cuML 앙상블 모델 래퍼 클래스
============================

cuML 앙상블 모델을 단일 모델처럼 사용할 수 있도록 래핑하는 클래스입니다.
기존 코드와의 호환성을 위해 sklearn 모델 인터페이스를 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional
import warnings
import gc

try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    warnings.warn("cuML이 설치되지 않았습니다. GPU 환경에서 실행해야 합니다.")


class EnsembleModelWrapper:
    """
    cuML 앙상블 모델을 단일 모델처럼 사용할 수 있도록 래핑하는 클래스
    
    여러 cuML RandomForest 모델의 예측을 평균하여 앙상블 예측을 수행합니다.
    sklearn 모델 인터페이스를 제공하여 기존 코드와 호환됩니다.
    """
    
    def __init__(self, models: List, scaler=None, lazy_importances=False):
        """
        Args:
            models: cuML RandomForest 모델 리스트
            scaler: cuML StandardScaler (선택사항)
            lazy_importances: True면 피처 중요도 계산을 지연 (메모리 최적화)
        """
        if not models:
            raise ValueError("모델 리스트가 비어있습니다.")
        
        self.models = models
        self.scaler = scaler
        self.n_models = len(models)
        self._lazy_importances = lazy_importances
        
        # 첫 번째 모델의 파라미터를 기본값으로 사용
        self._params = models[0].get_params() if hasattr(models[0], 'get_params') else {}
        
        # feature_importances_는 첫 번째 모델의 것을 사용 (나중에 평균으로 업데이트 가능)
        self._feature_importances_ = None
        if not lazy_importances:
            self._update_feature_importances()
    
    def _update_feature_importances(self):
        """모든 모델의 피처 중요도를 평균하여 계산"""
        try:
            importances_list = []
            
            for i, model in enumerate(self.models):
                # cuML 모델의 다른 가능한 속성/메서드 확인
                if hasattr(model, 'get_feature_importances'):
                    try:
                        imp = model.get_feature_importances()
                        if imp is not None:
                            # cuDF Series나 다른 형태를 numpy 배열로 변환
                            if hasattr(imp, 'to_pandas'):
                                imp = imp.to_pandas().values
                            elif hasattr(imp, 'values'):
                                imp = imp.values
                            elif hasattr(imp, 'to_numpy'):
                                imp = imp.to_numpy()
                            elif not isinstance(imp, np.ndarray):
                                imp = np.array(imp)
                            importances_list.append(imp)
                    except Exception:
                        pass
                
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    
                    if imp is not None:
                        # cuDF Series나 다른 형태를 numpy 배열로 변환
                        try:
                            if hasattr(imp, 'to_pandas'):
                                imp = imp.to_pandas().values
                            elif hasattr(imp, 'values'):
                                imp = imp.values
                            elif hasattr(imp, 'to_numpy'):
                                imp = imp.to_numpy()
                            elif not isinstance(imp, np.ndarray):
                                imp = np.array(imp)
                            
                            importances_list.append(imp)
                        except Exception:
                            pass
            
            if importances_list:
                # 모든 모델의 피처 중요도 평균
                self._feature_importances_ = np.mean(importances_list, axis=0)
            else:
                # 피처 중요도를 가져올 수 없는 경우
                self._feature_importances_ = None
        except Exception:
            self._feature_importances_ = None
    
    @property
    def feature_importances_(self):
        """피처 중요도 (모든 모델의 평균)"""
        if self._feature_importances_ is None:
            if self._lazy_importances:
                # 지연 로딩: 필요할 때만 계산
                self._update_feature_importances()
            else:
                self._update_feature_importances()
        return self._feature_importances_
    
    def get_params(self, deep=True):
        """모델 파라미터 반환 (첫 번째 모델의 파라미터 사용)"""
        return self._params.copy()
    
    def predict_proba(self, X):
        """
        예측 확률 계산 (모든 모델의 평균)
        
        Args:
            X: 입력 데이터 (pandas DataFrame, cuDF DataFrame, numpy array)
            
        Returns:
            numpy array: 예측 확률 [n_samples, n_classes]
        """
        # 입력 데이터를 cuDF로 변환 (필요한 경우)
        is_cudf = False
        is_cupy = False
        
        if CUML_AVAILABLE:
            if isinstance(X, cudf.DataFrame):
                # 원본 참조 끊기 위해 .copy() 사용 (메모리 누수 방지)
                X_cudf = X.copy()
                is_cudf = True
            elif isinstance(X, pd.DataFrame):
                X_cudf = cudf.from_pandas(X)
            elif isinstance(X, cp.ndarray):
                X_cudf = cudf.DataFrame(X)
                is_cupy = True
            elif isinstance(X, np.ndarray):
                X_cudf = cudf.DataFrame(X)
            else:
                # 기타 타입은 pandas로 변환 후 cuDF로 변환
                X_cudf = cudf.from_pandas(pd.DataFrame(X))
        else:
            # cuML이 없는 경우 (이론적으로는 발생하지 않아야 함)
            raise RuntimeError("cuML이 설치되지 않았습니다. GPU 환경에서 실행해야 합니다.")
        
        def predict_single_model_with_cleanup(model, X_cudf_input, model_idx, total_models):
            """
            단일 모델 예측 (독립 함수로 메모리 자동 해제)
            함수 종료 시 로컬 변수들이 자동으로 해제되어 FIL 내부 버퍼 누적 방지
            
            Args:
                model: cuML RandomForest 모델
                X_cudf_input: 입력 데이터 (cuDF DataFrame)
                model_idx: 모델 인덱스 (0부터 시작)
                total_models: 전체 모델 개수
                
            Returns:
                numpy array: 예측 확률 [n_samples, n_classes]
            """
            proba_cudf = None
            proba = None
            
            try:
                # 모델 예측
                proba_cudf = model.predict_proba(X_cudf_input)
                
                # cuDF DataFrame이나 cupy array를 numpy로 변환
                if hasattr(proba_cudf, 'to_pandas'):
                    proba = proba_cudf.to_pandas().values
                elif hasattr(proba_cudf, 'values'):
                    proba = proba_cudf.values
                elif hasattr(proba_cudf, 'to_numpy'):
                    proba = proba_cudf.to_numpy()
                elif isinstance(proba_cudf, cp.ndarray):
                    proba = cp.asnumpy(proba_cudf)
                elif not isinstance(proba_cudf, np.ndarray):
                    proba = np.array(proba_cudf)
                else:
                    proba = proba_cudf
                
                # cuML 예측 결과 즉시 삭제 (numpy 변환 완료 후)
                del proba_cudf
                safe_gpu_memory_cleanup()
                gc.collect()
                
                # cuPy 메모리 풀 강제 해제 (FIL 내부 버퍼 정리)
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                
                # cuML 메모리 정리 API 호출
                try:
                    import cuml
                    if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                        if hasattr(cuml.utils.memory_utils, 'rts'):
                            cuml.utils.memory_utils.rts.cuda_free_memory()
                except (AttributeError, ImportError):
                    pass
                
                # 추가 메모리 정리
                safe_gpu_memory_cleanup()
                gc.collect()
                
                # 함수 종료 시 로컬 변수 자동 해제 (FIL 내부 버퍼 참조 해제)
                return proba
                
            except Exception as e:
                # 예외 발생 시 생성된 객체들 명시적으로 삭제
                if proba_cudf is not None:
                    del proba_cudf
                if proba is not None:
                    del proba
                
                # 예외 발생 시 더 적극적인 메모리 정리
                safe_gpu_memory_cleanup()
                gc.collect()
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                try:
                    import cuml
                    if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                        if hasattr(cuml.utils.memory_utils, 'rts'):
                            cuml.utils.memory_utils.rts.cuda_free_memory()
                except (AttributeError, ImportError):
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
                
                # 예외를 다시 발생시켜 상위에서 처리
                raise
        
        # 모든 모델의 예측 확률 계산 (메모리 최적화: 누적 합산 방식)
        sum_proba = None
        count = 0
        
        for i, model in enumerate(self.models):
            # 모델 예측 전 메모리 정리 (첫 번째 모델이 아닌 경우)
            if i > 0:
                safe_gpu_memory_cleanup()
                gc.collect()
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                try:
                    import cuml
                    if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                        if hasattr(cuml.utils.memory_utils, 'rts'):
                            cuml.utils.memory_utils.rts.cuda_free_memory()
                except (AttributeError, ImportError):
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
            
            try:
                # 독립 함수로 모델 예측 (함수 종료 시 자동 메모리 해제)
                proba = predict_single_model_with_cleanup(model, X_cudf, i, len(self.models))
                
                # 누적 합산 (메모리 최적화: 리스트에 저장하지 않음)
                if sum_proba is None:
                    sum_proba = proba.copy()
                else:
                    sum_proba += proba
                count += 1
                
                # 중간 객체 즉시 삭제 (메모리 최적화)
                del proba
                
                # 각 모델 예측 후 추가 메모리 정리 (FIL 내부 버퍼 해제)
                safe_gpu_memory_cleanup()
                gc.collect()
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
                
            except Exception as e:
                warnings.warn(f"모델 {i+1}/{self.n_models} 예측 실패: {e}")
                # 예외 발생 시 더 적극적인 메모리 정리
                safe_gpu_memory_cleanup()
                gc.collect()
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                try:
                    import cuml
                    if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                        if hasattr(cuml.utils.memory_utils, 'rts'):
                            cuml.utils.memory_utils.rts.cuda_free_memory()
                except (AttributeError, ImportError):
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
                continue
        
        if count == 0:
            raise RuntimeError("모든 모델의 예측이 실패했습니다.")
        
        # 평균 계산
        avg_proba = sum_proba / count
        
        # 중간 객체 삭제 및 GPU 메모리 정리
        del sum_proba
        # X_cudf가 새로 생성된 경우에만 삭제 (입력이 cuDF가 아니었던 경우)
        if not is_cudf and 'X_cudf' in locals():
            del X_cudf
        safe_gpu_memory_cleanup()
        gc.collect()
        
        return avg_proba
    
    def predict(self, X):
        """
        예측 클래스 반환
        
        Args:
            X: 입력 데이터
            
        Returns:
            numpy array: 예측 클래스 [n_samples]
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def oob_score_(self):
        """Out-of-bag score (cuML은 지원하지 않으므로 None 반환)"""
        return None
    
    @property
    def n_estimators(self):
        """트리 개수 (첫 번째 모델의 값 사용)"""
        if hasattr(self.models[0], 'n_estimators'):
            return self.models[0].n_estimators
        return None
    
    @property
    def max_depth(self):
        """최대 깊이 (첫 번째 모델의 값 사용)"""
        if hasattr(self.models[0], 'max_depth'):
            return self.models[0].max_depth
        return None

def safe_gpu_memory_cleanup():
    """안전하게 GPU 메모리를 정리합니다."""
    try:
        gc.collect()
        # cuML의 메모리 정리 API는 버전에 따라 다를 수 있으므로 안전하게 처리
        try:
            import cuml
            if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                if hasattr(cuml.utils.memory_utils, 'rts'):
                    cuml.utils.memory_utils.rts.cuda_free_memory()
        except (AttributeError, ImportError):
            # API가 없는 경우 무시 (GPU 메모리는 Python GC로도 정리됨)
            pass
    except Exception as e:
        warnings.warn(f"GPU 메모리 정리 중 오류: {e}")

