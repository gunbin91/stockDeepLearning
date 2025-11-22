# GPU Accelerated Stock Predictor Training Script
#
# Features:
# - Just-in-Time (JIT) data loading to minimize memory usage.
# - Interactive data management (prompts for regeneration).
# - GPU acceleration using cuDF and cuML.
# - Hyperparameter optimization using Optuna.
# - Explicit memory management for stability.


import os
import sys
import argparse
import shutil
import glob
import subprocess
from datetime import datetime, timedelta
import gc
import warnings

# cudf의 feather reader 관련 UserWarning은 예상된 동작이므로 숨김 처리
warnings.filterwarnings("ignore", message="Using CPU via PyArrow to read feather dataset", category=UserWarning)

# 서드파티 라이브러리
import pandas as pd
import cudf
import cuml
import numpy as np
import joblib
import optuna
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.metrics import roc_auc_score
from sklearn.model_selection import KFold
import psutil
from numba import cuda

import cupy as cp

# SHAP 라이브러리 (피처 중요도 계산용)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP 라이브러리가 설치되지 않았습니다. 피처 중요도 계산을 건너뜁니다.")


# 내부 모듈

# 프로젝트 루트를 sys.path에 추가하여 다른 모듈 임포트

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_processor

from path_manager import path_manager

from logger import log_info, log_warning, log_error, log_critical



# Optuna TPE Sampler의 seed 고정을 위한 설정

optuna.samplers.TPESampler.init_rng = lambda self: np.random.RandomState()



# --- 메모리 관리 유틸리티 ---



def get_memory_usage(gpu=False):

    """현재 CPU 또는 GPU 메모리 사용량을 MB 단위로 반환합니다."""

    if gpu:

        try:

            # nvidia-smi 명령어를 직접 호출하여 가장 정확한 메모리 사용량 확인

            # memory.used [MiB]를 쿼리

            command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"

            result = subprocess.check_output(command, shell=True, encoding='utf-8')

            # 여러 GPU가 있을 경우 첫 번째 GPU의 메모리를 반환

            used_mb = int(result.strip().split('\n')[0])

            return used_mb

        except Exception:

            # nvidia-smi가 실패할 경우 0.0 반환

            return 0.0

    else:

        process = psutil.Process(os.getpid())

        return process.memory_info().rss / 1024**2



def log_memory_usage(stage_name, additional_info=None):



    """CPU 및 GPU 메모리 사용량을 로그로 출력합니다."""



    try:



        # GPU 작업이 완료될 때까지 동기화하여 정확한 메모리 측정



        cuda.synchronize()



    except Exception:



        pass  # Numba 컨텍스트가 활성화되지 않은 경우 등 예외 무시



        



    cpu_mem = get_memory_usage(gpu=False)



    gpu_mem = get_memory_usage(gpu=True)
    
    # GPU 메모리 총량도 함께 확인
    try:
        command = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True, encoding='utf-8')
        gpu_total_mb = int(result.strip().split('\n')[0])
        gpu_usage_pct = (gpu_mem / gpu_total_mb * 100) if gpu_total_mb > 0 else 0
        gpu_info = f"{gpu_mem:.1f} MB / {gpu_total_mb:.1f} MB ({gpu_usage_pct:.1f}%)"
    except Exception:
        gpu_info = f"{gpu_mem:.1f} MB"
    
    info_str = f"   💾 {stage_name} - CPU: {cpu_mem:.1f} MB, GPU: {gpu_info}"
    if additional_info:
        info_str += f" | {additional_info}"
    log_info(info_str)


def safe_gpu_memory_cleanup():

    """안전하게 GPU 메모리를 정리합니다."""

    try:

        gc.collect()

        # cuPy 메모리 풀 정리
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            # cuPy 메모리 풀 정리 실패 시 무시
            pass

        # cuML의 메모리 정리 API는 버전에 따라 다를 수 있으므로 안전하게 처리
        try:
            # cuML 24.04+ 버전
            if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                if hasattr(cuml.utils.memory_utils, 'rts'):
                    cuml.utils.memory_utils.rts.cuda_free_memory()
        except AttributeError:
            # API가 없는 경우 무시 (GPU 메모리는 Python GC로도 정리됨)
            pass

    except Exception as e:

        log_warning(f"   ⚠️ GPU 메모리 정리 중 오류: {e}")


def apply_hybrid_sampling(X, y, target_ratio=0.5, random_state=42):
    """
    하이브리드 샘플링을 적용하여 클래스 불균형을 보정합니다.
    - 소수 클래스: 오버샘플링 (복제)
    - 다수 클래스: 언더샘플링 (랜덤 제거)
    
    Args:
        X: 특징 데이터 (cudf DataFrame)
        y: 타겟 변수 (cudf Series)
        target_ratio: 목표 클래스 비율 (기본값 0.5 = 50:50)
        random_state: 재현성을 위한 시드
    
    Returns:
        X_resampled: 리샘플링된 특징 데이터
        y_resampled: 리샘플링된 타겟 변수
    """
    try:
        # cuDF Series를 pandas로 변환하여 처리
        if hasattr(y, 'to_pandas'):
            y_pandas = y.to_pandas()
            X_pandas = X.to_pandas()
        else:
            y_pandas = y
            X_pandas = X
        
        # 클래스 분포 확인
        value_counts = y_pandas.value_counts()
        n_samples = len(y_pandas)
        
        if len(value_counts) < 2:
            # 클래스가 1개만 있는 경우 샘플링 불필요
            return X, y
        
        # 소수 클래스와 다수 클래스 식별
        minority_class = value_counts.idxmin()
        majority_class = value_counts.idxmax()
        minority_count = value_counts[minority_class]
        majority_count = value_counts[majority_class]
        
        # 목표 샘플 수 계산
        target_minority = int(majority_count * target_ratio / (1 - target_ratio))
        target_majority = int(minority_count * (1 - target_ratio) / target_ratio)
        
        # 실제 목표: 더 균형잡힌 비율로 조정
        target_total = min(minority_count + majority_count, 
                          int(minority_count / target_ratio))
        target_minority = int(target_total * target_ratio)
        target_majority = target_total - target_minority
        
        # 소수 클래스 오버샘플링
        minority_indices = y_pandas[y_pandas == minority_class].index
        if len(minority_indices) < target_minority:
            # 복제가 필요한 경우
            n_repeats = target_minority // len(minority_indices)
            remainder = target_minority % len(minority_indices)
            
            minority_indices_resampled = list(minority_indices) * n_repeats
            if remainder > 0:
                rng = np.random.RandomState(random_state)
                additional = rng.choice(minority_indices, size=remainder, replace=False)
                minority_indices_resampled.extend(additional)
        else:
            # 이미 충분한 경우 랜덤 선택
            rng = np.random.RandomState(random_state)
            minority_indices_resampled = rng.choice(minority_indices, 
                                                    size=target_minority, 
                                                    replace=False).tolist()
        
        # 다수 클래스 언더샘플링
        majority_indices = y_pandas[y_pandas == majority_class].index
        if len(majority_indices) > target_majority:
            rng = np.random.RandomState(random_state)
            majority_indices_resampled = rng.choice(majority_indices, 
                                                   size=target_majority, 
                                                   replace=False).tolist()
        else:
            majority_indices_resampled = list(majority_indices)
        
        # 샘플 결합 및 셔플
        all_indices = minority_indices_resampled + majority_indices_resampled
        rng = np.random.RandomState(random_state)
        rng.shuffle(all_indices)
        
        # 리샘플링된 데이터 생성
        X_resampled_pandas = X_pandas.loc[all_indices].reset_index(drop=True)
        y_resampled_pandas = y_pandas.loc[all_indices].reset_index(drop=True)
        
        # cuDF로 변환
        if hasattr(X, 'to_pandas'):
            X_resampled = cudf.from_pandas(X_resampled_pandas)
            y_resampled = cudf.from_pandas(y_resampled_pandas)
        else:
            X_resampled = X_resampled_pandas
            y_resampled = y_resampled_pandas
        
        return X_resampled, y_resampled
        
    except Exception as e:
        log_warning(f"   ⚠️ 샘플링 실패: {e}. 원본 데이터 사용.")
        return X, y



# --- 데이터 준비 (전처리) ---



def prepare_data_and_save(data_path, start_date, end_date):

    """

    데이터를 전처리하고, 종목별로 분리하여 Feather 파일로 저장합니다.

    """

    log_info("🚀 실시간 데이터 수집 및 전처리를 시작합니다...")

    log_memory_usage("데이터 전처리 시작")

    # 기존 CPU 버전과 동일하게, 검증된 핵심 피처 목록을 하드코딩
    features = [
        'PBR', 'log_mktcap', '이익수익률', 'BPS',
        '수익률(1M)', '수익률(3M)', '52주_신고가_비율',
        'ADX_14',
        '변동성(1W)', '변동성(1M)', '변동성(3M)', 'ATRr_14', 'BBW_20_2', 'BB_Position',
        'disparity_120', 'disparity_240',
        '거래대금_MA5', '거래대금_MA20', 'OBV',
        'KOSPI_pct_1d', 'KOSPI_pct_5d', 'USDKRW_pct_1d', 'USDKRW_pct_5d',
        'VIX_pct_1d', 'VIX_pct_5d'
    ]

    try:

        # data_processor를 통해 모든 피처가 계산된 거대 데이터프레임 생성

        # 이 단계에서 메모리 사용량이 일시적으로 크게 증가함

        full_df = data_processor.get_preprocessed_data(start_date, end_date)

        log_memory_usage("전체 데이터 로딩 완료")



        if full_df is None or full_df.empty:

            log_error("데이터 전처리 중 오류가 발생하여 데이터를 생성할 수 없습니다.")

            return False

        # target 필터링 (전체 데이터에서)
        log_info("   🔍 target 필터링 중...")
        full_df = full_df[full_df['target'].notna()].copy()
        if full_df.empty:
            log_error("target 필터링 후 데이터가 없습니다.")
            return False

        # imputation_values 계산 (전체 데이터 기준)
        log_info("   📊 결측치 대체값(중앙값) 계산 중...")
        numeric_features_df = full_df[features].select_dtypes(include=np.number)
        imputation_values = numeric_features_df.median().to_dict()
        log_info(f"   ✅ 결측치 대체값 계산 완료 ({len(features)}개 피처)")

        # imputation_values 파일 저장
        imputation_values_dir = os.path.dirname(os.path.expanduser(data_path))
        imputation_values_path = os.path.join(imputation_values_dir, "imputation_values.joblib")
        log_info(f"   💾 imputation_values 파일 저장 중... ({imputation_values_path})")
        try:
            joblib.dump(imputation_values, imputation_values_path)
            log_info("   ✅ imputation_values 파일 저장 완료.")
        except Exception as e:
            log_warning(f"   ⚠️ imputation_values 파일 저장 실패: {e}. 데이터 저장은 계속 진행됩니다.")

        os.makedirs(data_path, exist_ok=True)

        tickers = full_df['종목코드'].unique()

        log_info(f"   📂 총 {len(tickers)}개 종목의 데이터를 전처리하여 개별 파일로 저장합니다...")

        for i, ticker in enumerate(tickers):

            ticker_df = full_df[full_df['종목코드'] == ticker].copy()

            # 전처리: 피처 선택, 타입 변환, fillna
            X = ticker_df[features].astype(np.float32)
            y = ticker_df['target'].astype(np.int32)
            X = X.fillna(imputation_values)

            # X와 y를 하나의 DataFrame으로 합치기
            preprocessed_df = X.copy()
            preprocessed_df['target'] = y

            file_path = os.path.join(data_path, f"{ticker}.feather")

            # pandas를 사용하여 feather 파일로 저장 (전처리 완료된 데이터)
            preprocessed_df.to_feather(file_path)

            # 메모리 정리
            del ticker_df, X, y, preprocessed_df

            if (i + 1) % 200 == 0:

                log_info(f"      ... {i+1}/{len(tickers)} 개 종목 저장 완료")



        log_info(f"   ✅ 총 {len(tickers)}개 종목의 전처리된 데이터 저장이 완료되었습니다.")

        

        # 거대 데이터프레임 메모리에서 명시적으로 해제

        del full_df, numeric_features_df

        gc.collect()

        log_memory_usage("개별 파일 저장 후 메모리 정리")

        

        return True



    except Exception as e:

        log_critical("데이터 준비 과정에서 심각한 오류 발생", exception=e)

        return False



# --- 모델 훈련 및 평가 ---



def load_fold_data(file_paths, features, imputation_values):



    """



    주어진 파일 경로 리스트에서 전처리된 데이터를 로드하여 반환합니다.



    (전처리 완료된 데이터를 로드하므로 concat만 수행, 속도 향상)



    """



    if not file_paths:



        return None, None



    



    try:



        # 제너레이터 표현식을 사용하여 메모리 효율성 개선



        df_generator = (cudf.read_feather(f) for f in file_paths)



        df = cudf.concat(df_generator)



        



        # 전처리 완료된 데이터이므로 필터링 불필요



        if df.empty:



            return None, None







        # features와 target만 추출 (이미 전처리 완료된 데이터, 타입 변환 불필요)
        X = df[features]



        y = df['target']







        # fillna 불필요 (이미 전처리 완료된 데이터)
        
        # 데이터 크기 로깅 (간단하게)
        try:
            x_mem = 0
            if hasattr(X, 'memory_usage'):
                mem_usage = X.memory_usage(deep=True)
                if hasattr(mem_usage, 'sum'):
                    x_mem = mem_usage.sum() / 1024**2  # MB
                elif isinstance(mem_usage, (int, float)):
                    x_mem = mem_usage / 1024**2  # MB
            
            y_mem = 0
            if hasattr(y, 'memory_usage'):
                mem_usage = y.memory_usage(deep=True)
                if hasattr(mem_usage, 'sum'):
                    y_mem = mem_usage.sum() / 1024**2  # MB
                elif isinstance(mem_usage, (int, float)):
                    y_mem = mem_usage / 1024**2  # MB
            elif hasattr(y, '__len__'):
                # Series나 배열인 경우 대략적 계산
                y_mem = len(y) * 4 / 1024**2  # int32 = 4 bytes
        except Exception:
            pass  # 로깅 실패는 무시



        



        # 원본 df 명시적 삭제 (메모리 관리 개선)



        del df



        gc.collect()



            



        return X, y



    except Exception as e:



        log_error(f"데이터 로딩 중 오류 발생: {e}")



        return None, None



def objective(trial, fold_data_cache, features, imputation_values, max_depth_list, rng):

    """

    Optuna를 위한 objective 함수 (청크 단위 학습 방식).

    fold_data_cache: 모든 Fold의 데이터 캐시 (main 함수에서 미리 로드)
    각 fold를 3개 청크로 나눠서 학습하고, 앙상블 모델로 예측하여 점수 계산

    """

    trial_start_time = datetime.now()
    log_info(f"\n{'='*60}")
    log_info(f"🚀 Optuna Trial #{trial.number} 시작")
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_categorical('max_depth', max_depth_list),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),  # 하한 상향: 2-50 → 5-50 (과적합 방지)
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),  # 하한 상향: 1-50 → 2-50 (노이즈 대응)
        'max_samples': trial.suggest_categorical('max_samples', [0.7, 0.8, 0.9, 1.0]),  # 0.6 제거: 정보 손실 방지
        'split_criterion': trial.suggest_categorical('split_criterion', [0, 1]),
        'random_state': 42,
        'n_streams': 1,
    }

    log_info(f"   📋 파라미터: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, max_samples={params['max_samples']}")

    if not fold_data_cache:
        log_error("Fold 데이터 캐시가 비어있습니다. Trial을 중단합니다.")
        return 0.0

    fold_scores = []

    # --- 각 Fold 처리 (캐시된 데이터 사용) ---
    for fold in range(len(fold_data_cache)):
        fold_start_time = datetime.now()
        
        # Fold 시작 전 GPU 메모리 정리 (이전 fold의 잔여 메모리 해제)
        if fold > 0:
            safe_gpu_memory_cleanup()
            gc.collect()

        # 캐시된 데이터 사용 (중복 로드 방지)
        X_train_all, y_train_all, X_val, y_val = fold_data_cache[fold]
        
        if X_train_all is None or X_val is None:
            log_warning(f"   ⚠️ Fold #{fold+1} 데이터가 없습니다. 건너뜁니다.")
            continue

        # 샘플링 제거: 실제 시장 분포를 그대로 사용 (보수적 예측)
        # 스케일러 학습 (각 Fold마다 새로 생성)
        step_start = datetime.now()
        scaler = cuStandardScaler()
        scaler.fit(X_train_all)
        scaler_fit_time = (datetime.now() - step_start).total_seconds()

        # 훈련 데이터 스케일링
        step_start = datetime.now()
        X_train_scaled = scaler.transform(X_train_all)
        train_scale_time = (datetime.now() - step_start).total_seconds()
        
        # 원본 훈련 데이터 즉시 삭제 (스케일링 완료 후 불필요, 메모리 최적화)
        del X_train_all
        safe_gpu_memory_cleanup()
        gc.collect()

        # 검증 데이터 스케일링
        step_start = datetime.now()
        X_val_scaled = scaler.transform(X_val)
        val_scale_time = (datetime.now() - step_start).total_seconds()
        
        # 원본 검증 데이터 즉시 삭제 (스케일링 완료 후 불필요, 메모리 최적화)
        del X_val
        safe_gpu_memory_cleanup()
        gc.collect()

        # 청크 단위 학습 (3개 청크로 분할)
        n_chunks = 3
        chunk_size = len(X_train_scaled) // n_chunks
        chunk_models = []
        fit_time = 0.0
        
        step_start = datetime.now()
        for chunk_idx in range(n_chunks):
            # 청크 인덱스 계산 (데이터 정합성 보장)
            start_idx = chunk_idx * chunk_size
            if chunk_idx == n_chunks - 1:
                # 마지막 청크는 나머지 모든 데이터 포함
                end_idx = len(X_train_scaled)
            else:
                end_idx = (chunk_idx + 1) * chunk_size
            
            # 청크 데이터 추출
            X_chunk = X_train_scaled.iloc[start_idx:end_idx]
            y_chunk = y_train_all.iloc[start_idx:end_idx]
            
            # 청크별 모델 학습
            chunk_model = cuRF(**params)
            chunk_model.fit(X_chunk, y_chunk)
            chunk_models.append(chunk_model)
            
            # 청크 데이터 즉시 삭제 (메모리 최적화)
            del X_chunk, y_chunk
            safe_gpu_memory_cleanup()
            gc.collect()
        
        fit_time = (datetime.now() - step_start).total_seconds()
        
        # X_train_scaled 즉시 삭제 (모델 학습 완료 후 불필요)
        del X_train_scaled, y_train_all
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # 앙상블 모델로 예측 (GPU 메모리 최적화 및 예외 처리)
        step_start = datetime.now()
        
        # 1. 예측 전 GPU 메모리 강제 해제 (FIL 모델 생성 전 메모리 확보)
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # 2. 앙상블 예측 수행 (MemoryError 발생 시 fold 건너뛰기)
        try:
            from ml_model_wrapper import EnsembleModelWrapper
            ensemble_model = EnsembleModelWrapper(chunk_models, scaler, lazy_importances=True)
            
            # 검증 데이터에 대해 앙상블 예측
            pred_proba = ensemble_model.predict_proba(X_val_scaled)
            
            # 앙상블 모델 및 청크 모델들 삭제
            del ensemble_model
            for cm in chunk_models:
                del cm
            del chunk_models
            safe_gpu_memory_cleanup()
            gc.collect()
            
        except MemoryError as e:
            # GPU 메모리 부족으로 예측 실패
            log_error(f"   ❌ GPU 메모리 부족으로 예측 실패: {e}")
            log_warning(f"   ⚠️ Fold #{fold+1}를 건너뛰고 다음 fold로 진행합니다.")
            # GPU 메모리 강제 해제
            if 'chunk_models' in locals():
                for cm in chunk_models:
                    del cm
                del chunk_models
            if 'X_val_scaled' in locals():
                del X_val_scaled
            safe_gpu_memory_cleanup()
            gc.collect()
            # 해당 fold는 건너뛰고 다음 fold로 진행
            continue
        
        pred_time = (datetime.now() - step_start).total_seconds()
        
        # 예측 결과 추출
        if isinstance(pred_proba, np.ndarray):
            if len(pred_proba.shape) == 2 and pred_proba.shape[1] > 1:
                y_pred_proba = pred_proba[:, 1]
            else:
                y_pred_proba = pred_proba.flatten()
        else:
            y_pred_proba = pred_proba
        
        # 3. 예측 결과 및 검증 데이터 삭제
        del X_val_scaled, pred_proba
        safe_gpu_memory_cleanup()
        gc.collect()

        # ROC-AUC 점수 계산
        score = roc_auc_score(y_val, y_pred_proba)

        fold_scores.append(score)
        
        # y_pred_proba 즉시 삭제 (ROC-AUC 계산 완료 후 불필요, 메모리 최적화)
        del y_pred_proba
        safe_gpu_memory_cleanup()
        gc.collect()

        fold_duration = (datetime.now() - fold_start_time).total_seconds()
        
        # Fold 완료 로그 (한 줄로 통합)
        preprocessing_time = scaler_fit_time + train_scale_time + val_scale_time
        log_info(f"   ✅ Fold #{fold+1}/3 | Score: {score:.4f} | 총: {fold_duration:.1f}초 "
                f"(전처리: {preprocessing_time:.1f}초, 학습: {fit_time:.1f}초, 예측: {pred_time:.1f}초, 청크: {n_chunks}개)")

        # Pruning 체크: 각 fold 후 중간 점수를 보고 나쁜 trial 조기 종료
        trial.report(score, step=fold)
        if trial.should_prune():
            trial_duration = (datetime.now() - trial_start_time).total_seconds()
            log_info(f"   ⏹️ Trial #{trial.number} 조기 종료 (Pruning) | 총 소요시간: {trial_duration/60:.1f}분 ({trial_duration:.1f}초)")
            log_info(f"{'='*60}")
            # Pruning 시 캐시 데이터는 유지 (다른 Trial에서 사용)
            # GPU 메모리 정리 후 종료
            safe_gpu_memory_cleanup()
            gc.collect()
            raise optuna.TrialPruned()

        # 4. Fold 정리 (모든 변수 삭제)
        # X_train_all, X_val, X_val_scaled, pred_proba, chunk_models, y_pred_proba는 이미 위에서 삭제됨
        del scaler, y_val
        safe_gpu_memory_cleanup()
        gc.collect()

    if not fold_scores:
        log_error("모든 Fold에서 학습에 실패했습니다. Trial을 중단합니다.")
        return 0.0

    mean_score = np.mean(fold_scores)
    trial_duration = (datetime.now() - trial_start_time).total_seconds()
    log_info(f"✅ Trial #{trial.number} 완료 | 평균 Score: {mean_score:.4f} | "
            f"총 소요시간: {trial_duration/60:.1f}분 ({trial_duration:.1f}초)")
    log_info(f"{'='*60}")

    return mean_score

def train_final_ensemble_model(fold_data_cache, features, imputation_values, best_params, rng, optimization_results=None, training_config=None):
    """
    전체 데이터를 사용하여 최종 단일 모델을 훈련하고 저장합니다.
    
    fold_data_cache: 모든 Fold의 데이터 캐시 (main 함수에서 미리 로드)
    """
    log_info("\n--- 🚂 최적 파라미터로 최종 단일 모델 훈련 시작 ---")
    
    try:
        # 캐시된 데이터로 전체 데이터 구성
        log_info("   📊 전체 데이터 구성 중...")
        compose_start = datetime.now()
        X_all_list = []
        y_all_list = []
        
        for fold in range(len(fold_data_cache)):
            X_train, y_train, X_val, y_val = fold_data_cache[fold]
            if X_train is None or X_val is None:
                log_warning(f"   ⚠️ Fold #{fold+1} 데이터가 없습니다. 건너뜁니다.")
                continue
            # 모든 Fold의 훈련 데이터와 검증 데이터를 합침
            X_all_list.append(X_train)
            X_all_list.append(X_val)
            y_all_list.append(y_train)
            y_all_list.append(y_val)
        
        # cuDF concat으로 전체 데이터 합치기
        X_all = cudf.concat(X_all_list, ignore_index=True)
        y_all = cudf.concat(y_all_list, ignore_index=True)
        del X_all_list, y_all_list
        gc.collect()
        compose_time = (datetime.now() - compose_start).total_seconds()
        log_info(f"   ✅ 전체 데이터 구성 완료: {len(X_all):,}행 ({compose_time:.1f}초)")
        
        # fold_data_cache의 원본 데이터 참조 해제 (X_all 생성 완료 후 더 이상 불필요, 메모리 최적화)
        # 주의: fold_data_cache는 함수 파라미터이므로 내용만 삭제하고 딕셔너리 구조는 유지
        for fold_key in list(fold_data_cache.keys()):
            fold_data = fold_data_cache[fold_key]
            if fold_data:
                X_train, y_train, X_val, y_val = fold_data
                del X_train, y_train, X_val, y_val
                fold_data_cache[fold_key] = None
        safe_gpu_memory_cleanup()
        gc.collect()
        
        final_scaler = cuStandardScaler()
        
        # 전체 데이터로 스케일러를 fit
        step_start = datetime.now()
        final_scaler.fit(X_all)
        scaler_fit_time = (datetime.now() - step_start).total_seconds()

        # 스케일링
        step_start = datetime.now()
        X_all_scaled = final_scaler.transform(X_all)
        scale_time = (datetime.now() - step_start).total_seconds()
        
        # 원본 데이터 즉시 삭제 (스케일링 완료 후 불필요, 메모리 최적화)
        del X_all
        safe_gpu_memory_cleanup()
        gc.collect()
        
        log_info(f"   ✅ 전처리 완료 (스케일러 학습: {scaler_fit_time:.1f}초, 스케일링: {scale_time:.1f}초)")
        
        # 청크 단위 학습으로 앙상블 모델 생성 (Knowledge Distillation을 위한 teacher 모델)
        n_chunks = 6
        chunk_size = len(X_all_scaled) // n_chunks
        chunk_models = []
        ensemble_fit_time = 0.0
        
        step_start = datetime.now()
        for chunk_idx in range(n_chunks):
            # 청크 인덱스 계산 (데이터 정합성 보장)
            start_idx = chunk_idx * chunk_size
            if chunk_idx == n_chunks - 1:
                # 마지막 청크는 나머지 모든 데이터 포함
                end_idx = len(X_all_scaled)
            else:
                end_idx = (chunk_idx + 1) * chunk_size
            
            # 청크 데이터 추출 (원본 참조 끊기 위해 .copy() 사용)
            X_chunk = X_all_scaled.iloc[start_idx:end_idx].copy()
            y_chunk = y_all.iloc[start_idx:end_idx].copy()
            
            # 청크별 모델 학습
            chunk_model = cuRF(**best_params)
            chunk_model.fit(X_chunk, y_chunk)
            chunk_models.append(chunk_model)
            
            # 청크 데이터 즉시 삭제 (메모리 최적화)
            del X_chunk, y_chunk
            # 모델 학습 후 GPU 메모리 정리 강화
            safe_gpu_memory_cleanup()
            gc.collect()
            safe_gpu_memory_cleanup()  # 이중 정리로 메모리 확보
        
        ensemble_fit_time = (datetime.now() - step_start).total_seconds()
        log_info(f"   ✅ 앙상블 모델 학습 완료 ({ensemble_fit_time:.1f}초, {n_chunks}개 청크)")
        
        # y_all 삭제 (앙상블 모델 학습 완료 후 더 이상 불필요, 예측 단계에서 사용 안 함)
        del y_all
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # Knowledge Distillation: 앙상블 모델의 예측을 soft label로 사용하여 단일 모델 학습
        from ml_model_wrapper import EnsembleModelWrapper
        ensemble_model = EnsembleModelWrapper(chunk_models, final_scaler, lazy_importances=True)
        
        # 앙상블 모델로 전체 데이터에 대해 청크 단위 예측 (soft label 생성, 메모리 최적화)
        log_info("   📊 앙상블 모델로 soft label 생성 중...")
        step_start = datetime.now()
        
        # 예측 전 GPU 메모리 정리
        safe_gpu_memory_cleanup()
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # 청크 단위 예측 (메모리 최적화: 누적 합산 방식 사용)
        n_pred_chunks = 60  # 예측용 청크 수 (메모리 사용량 감소, 30 → 60으로 증가)
        pred_chunk_size = len(X_all_scaled) // n_pred_chunks
        soft_labels = None  # 누적 합산 방식으로 변경 (리스트 대신)
        total_samples = 0  # 총 샘플 수 추적
        
        try:
            for pred_chunk_idx in range(n_pred_chunks):
                # 청크 인덱스 계산 (데이터 정합성 보장)
                pred_start_idx = pred_chunk_idx * pred_chunk_size
                if pred_chunk_idx == n_pred_chunks - 1:
                    # 마지막 청크는 나머지 모든 데이터 포함
                    pred_end_idx = len(X_all_scaled)
                else:
                    pred_end_idx = (pred_chunk_idx + 1) * pred_chunk_size
                
                # 청크 데이터 추출 (원본 참조 끊기 위해 .copy() 사용)
                X_pred_chunk = X_all_scaled.iloc[pred_start_idx:pred_end_idx].copy()
                
                # 예측 전 GPU 메모리 정리 (각 청크마다)
                safe_gpu_memory_cleanup()
                gc.collect()
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
                
                # 청크별 앙상블 예측 (MemoryError 발생 시 처리)
                try:
                    chunk_soft_labels = ensemble_model.predict_proba(X_pred_chunk)
                except (MemoryError, RuntimeError) as e:
                    error_msg = str(e)
                    if "out_of_memory" in error_msg or "MemoryError" in error_msg or "cudaErrorMemoryAllocation" in error_msg:
                        log_error(f"   ❌ GPU 메모리 부족으로 청크 #{pred_chunk_idx+1} 예측 실패: {e}")
                        # 메모리 정리 강화 (FIL 내부 버퍼 포함)
                        safe_gpu_memory_cleanup()
                        gc.collect()
                        try:
                            cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                        safe_gpu_memory_cleanup()
                        gc.collect()
                        # 재시도
                        try:
                            chunk_soft_labels = ensemble_model.predict_proba(X_pred_chunk)
                        except Exception as retry_e:
                            log_error(f"   ❌ 재시도도 실패: {retry_e}")
                            # 재시도 실패 시 해당 청크는 건너뛰고 NaN으로 채움
                            chunk_size_actual = pred_end_idx - pred_start_idx
                            chunk_soft_labels = np.full(chunk_size_actual, 0.5, dtype=np.float32)  # 중립값으로 채움
                            log_warning(f"   ⚠️ 청크 #{pred_chunk_idx+1}는 중립값(0.5)으로 대체됩니다.")
                    else:
                        # MemoryError가 아닌 다른 예외는 그대로 전파
                        raise
                
                # 예측 결과 처리 (양성 클래스 확률 추출)
                if isinstance(chunk_soft_labels, np.ndarray):
                    if len(chunk_soft_labels.shape) == 2 and chunk_soft_labels.shape[1] > 1:
                        chunk_soft_labels = chunk_soft_labels[:, 1]  # 양성 클래스 확률
                    else:
                        chunk_soft_labels = chunk_soft_labels.flatten()
                
                # 누적 합산 방식으로 변경 (메모리 누적 방지)
                chunk_size_actual = len(chunk_soft_labels)
                if soft_labels is None:
                    # 첫 번째 청크: 직접 할당
                    soft_labels = chunk_soft_labels.copy()
                    total_samples = chunk_size_actual
                else:
                    # 이후 청크: 누적 합산 (순서 보장)
                    soft_labels = np.concatenate([soft_labels, chunk_soft_labels])
                    total_samples += chunk_size_actual
                
                # 청크 데이터 및 중간 결과 즉시 삭제 (메모리 최적화)
                del X_pred_chunk, chunk_soft_labels
                
                # 각 청크 예측 후 GPU 메모리 정리 강화 (FIL 내부 버퍼 포함)
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
            
            # 모든 청크 결과 확인
            if soft_labels is None or total_samples == 0:
                raise RuntimeError("soft label 생성 실패: 모든 청크 예측 실패")
            
            # 최종 결과 확인 (데이터 정합성 검증)
            if len(soft_labels) != len(X_all_scaled):
                log_warning(f"   ⚠️ soft label 길이 불일치: {len(soft_labels)} != {len(X_all_scaled)}")
                # 길이 맞추기 (부족한 경우 중립값으로 채움)
                if len(soft_labels) < len(X_all_scaled):
                    missing = len(X_all_scaled) - len(soft_labels)
                    soft_labels = np.concatenate([soft_labels, np.full(missing, 0.5, dtype=np.float32)])
                else:
                    soft_labels = soft_labels[:len(X_all_scaled)]
            
            safe_gpu_memory_cleanup()
            gc.collect()
            
        except Exception as e:
            # 예외 발생 시 메모리 정리 강화
            if 'soft_labels' in locals():
                del soft_labels
            if 'X_pred_chunk' in locals():
                del X_pred_chunk
            if 'chunk_soft_labels' in locals():
                del chunk_soft_labels
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
            log_error(f"   ❌ soft label 생성 중 오류 발생: {e}")
            raise
        
        # soft label을 이진 클래스로 변환 (0.6 기준)
        y_soft = (soft_labels >= 0.6).astype(np.int32)
        y_soft_cudf = cudf.Series(y_soft)
        
        # soft_labels 즉시 삭제 (y_soft_cudf 생성 완료 후 불필요)
        del soft_labels, y_soft
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # 앙상블 모델 및 청크 모델들 삭제 (메모리 최적화)
        del ensemble_model
        for cm in chunk_models:
            del cm
        del chunk_models
        # 모델 삭제 후 GPU 메모리 정리 강화
        safe_gpu_memory_cleanup()
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        safe_gpu_memory_cleanup()
        gc.collect()
        
        soft_label_time = (datetime.now() - step_start).total_seconds()
        
        # 단일 모델 학습 (soft label 사용)
        step_start = datetime.now()
        final_model = cuRF(**best_params)
        final_model.fit(X_all_scaled, y_soft_cudf)
        fit_time = (datetime.now() - step_start).total_seconds()
        
        # soft label 삭제 (y_all은 이미 예측 전에 삭제됨)
        del y_soft_cudf
        safe_gpu_memory_cleanup()
        gc.collect()
        
        log_info(f"   ✅ 모델 학습 완료 (앙상블: {ensemble_fit_time:.1f}초, soft label: {soft_label_time:.1f}초, 단일 모델: {fit_time:.1f}초)")
        
        # SHAP 계산 전 GPU 메모리 정리 (대용량 메모리 사용 전 정리)
        safe_gpu_memory_cleanup()
        gc.collect()
        
        # SHAP를 사용한 피처 중요도 계산 (단일 모델용)
        feature_importances = None
        if SHAP_AVAILABLE:
            try:
                log_info("   📊 SHAP 피처 중요도 계산 중...")
                shap_start = datetime.now()
                
                # 샘플 데이터 선택 (메모리 절약을 위해 최대 1000개 또는 전체의 10%)
                sample_size = min(1000, max(100, len(X_all_scaled) // 10))
                if sample_size < len(X_all_scaled):
                    # 랜덤 샘플링
                    sample_indices = np.random.choice(len(X_all_scaled), sample_size, replace=False)
                    X_sample_cudf = X_all_scaled.iloc[sample_indices]
                else:
                    X_sample_cudf = X_all_scaled
                
                # cuDF를 pandas로 변환 (SHAP는 CPU 데이터 필요)
                X_sample_pd = X_sample_cudf.to_pandas()
                
                # cuML 모델의 예측 함수 래핑 (SHAP 호환)
                def model_predict_wrapper(X):
                    """cuML 모델 예측 함수 래퍼 (SHAP 호환)"""
                    X_cudf = cudf.from_pandas(pd.DataFrame(X, columns=features))
                    try:
                        proba = final_model.predict_proba(X_cudf)
                        if isinstance(proba, cudf.DataFrame):
                            return proba.iloc[:, 1].to_pandas().values
                        elif hasattr(proba, 'iloc'):
                            return proba.iloc[:, 1].values
                        else:
                            return proba[:, 1]
                    finally:
                        del X_cudf
                
                # SHAP KernelExplainer 사용
                background_size = min(50, len(X_sample_pd))
                background_indices = np.random.choice(len(X_sample_pd), background_size, replace=False)
                X_background = X_sample_pd.iloc[background_indices].values
                
                explainer = shap.KernelExplainer(model_predict_wrapper, X_background)
                shap_values = explainer.shap_values(X_sample_pd.values, nsamples=100)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importances = list(zip(features, mean_abs_shap))
                feature_importances.sort(key=lambda x: x[1], reverse=True)
                
                shap_time = (datetime.now() - shap_start).total_seconds()
                log_info(f"   ✅ SHAP 피처 중요도 계산 완료 ({shap_time:.1f}초)")
                
                # 중간 객체 메모리 해제
                del X_sample_cudf, X_sample_pd, X_background, shap_values, mean_abs_shap, explainer
                safe_gpu_memory_cleanup()
                gc.collect()
                
            except Exception as e:
                # SHAP 계산 실패는 경고만 출력 (학습은 계속 진행)
                log_warning(f"   ⚠️ SHAP 피처 중요도 계산 실패: {type(e).__name__}: {str(e)}")
                feature_importances = None
        else:
            log_warning("   ⚠️ SHAP 라이브러리가 없어 피처 중요도를 계산할 수 없습니다.")
            feature_importances = None

        # 전체 데이터 메모리 해제 (SHAP 계산 완료 후)
        del X_all_scaled
        safe_gpu_memory_cleanup()
        gc.collect()
        
        if final_model is None:
            log_error("   최종 모델 훈련에 실패했습니다.")
            return

        log_info("   💾 최종 단일 모델 및 전처리기 저장 중...")
        model_path = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
        metadata_path = str(path_manager.data_dir / 'cuml_ensemble_model_metadata.joblib')
        
        # training_config에 모델 개수 저장 (단일 모델)
        if training_config:
            training_config['n_final_models'] = 1
        else:
            training_config = {'n_final_models': 1}
        
        # 피처 중요도 저장 전 확인
        if feature_importances is None:
            log_warning("   ⚠️ 피처 중요도가 None입니다. 모델 파일과 메타데이터에 저장되지 않습니다.")
        else:
            log_info(f"   💾 피처 중요도 저장 준비: {len(feature_importances)}개 피처")
        
        # 전체 모델 파일 저장 (예측에 사용)
        joblib.dump({
            'model': final_model,  # 단일 모델 저장
            'features': features,
            'scaler': final_scaler,
            'imputation_values': imputation_values,
            'best_params': best_params,
            'model_type': 'single_model',  # 단일 모델 타입
            'optimization_results': optimization_results or {},
            'training_config': training_config or {},
            'feature_importances': feature_importances  # SHAP 값으로 계산된 피처 중요도
        }, model_path, compress=3)
        
        # 메타데이터 파일 저장 (모델 분석 페이지용, 메모리 최적화)
        # 모델 객체는 저장하지 않고 정보만 저장
        joblib.dump({
            'features': features,
            'best_params': best_params,
            'model_type': 'single_model',
            'optimization_results': optimization_results or {},
            'training_config': training_config or {},
            'feature_importances': feature_importances,  # SHAP 값으로 계산된 피처 중요도 (메타데이터에도 저장)
            'parameter_explanations': {
                'n_estimators': 'RandomForest가 만들 트리의 개수',
                'max_depth': '각 트리의 최대 깊이 (과적합 방지)',
                'min_samples_split': '노드 분할에 필요한 최소 샘플 수',
                'min_samples_leaf': '리프 노드의 최소 샘플 수',
                'max_samples': '각 트리가 사용할 샘플 비율',
                'split_criterion': '분할 기준 (0: Gini, 1: Entropy)'
            }
        }, metadata_path, compress=3)
        
        log_info(f"   ✅ 최종 단일 모델이 '{model_path}' 경로에 저장되었습니다.")
        log_info(f"   ✅ 모델 메타데이터가 '{metadata_path}' 경로에 저장되었습니다 (메모리 최적화).")
        if feature_importances:
            log_info(f"   ✅ 피처 중요도가 모델 파일과 메타데이터에 저장되었습니다 ({len(feature_importances)}개 피처).")
        else:
            log_warning("   ⚠️ 피처 중요도가 저장되지 않았습니다. SHAP 계산이 실패했거나 SHAP 라이브러리가 설치되지 않았을 수 있습니다.")
    
    except Exception as e:
        log_critical("최종 모델 훈련 또는 저장 중 오류 발생", exception=e)
        log_warning("   최종 모델을 저장하지 못했습니다.")
    finally:
        # 함수 내에서 생성된 모든 변수 정리
        if 'final_model' in locals(): del final_model
        if 'final_scaler' in locals(): del final_scaler
        safe_gpu_memory_cleanup(); gc.collect()


# --- 메인 실행 로직 ---

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="GPU 가속 모델 훈련 스크립트")
    parser.add_argument('--n_iter', type=int, default=100, help='Optuna 탐색 횟수')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[10, 15, 20, 25, 30], help='max_depth 후보 리스트 (과적합 방지를 위해 10-30 권장)')
    args = parser.parse_args()
    
    # WSL 홈 디렉토리 내에 데이터 저장 경로 설정
    data_path = os.path.expanduser("~/stock_data/processed_feather")
    
    # --- 1. 데이터 준비 단계 ---
    run_preparation = False
    if os.path.exists(data_path) and len(os.listdir(data_path)) > 0:
        log_info(f"✅ 전처리된 데이터를 찾았습니다. 기존 데이터를 사용하여 학습을 시작합니다: {data_path}")
        run_preparation = False
    else:
        log_info(f"📂 전처리된 데이터가 없습니다. 데이터 생성을 시작합니다: {data_path}")
        run_preparation = True
        
        # 사용자로부터 학습 기간(년) 입력 받기
        while True:
            try:
                years_input = input("\n   최근 몇 년치 데이터로 학습하시겠습니까? (기본값: 10년, Enter 입력 시 기본값 사용): ").strip()
                if not years_input:
                    years = 10
                    log_info(f"   기본값 10년을 사용합니다.")
                    break
                years = int(years_input)
                if years <= 0:
                    print("   ⚠️ 1 이상의 숫자를 입력해주세요.")
                    continue
                if years > 50:
                    print("   ⚠️ 50년 이하의 숫자를 입력해주세요.")
                    continue
                log_info(f"   최근 {years}년치 데이터로 학습합니다.")
                break
            except ValueError:
                print("   ⚠️ 올바른 숫자를 입력해주세요.")
                continue
            except KeyboardInterrupt:
                log_critical("사용자가 입력을 취소했습니다. 프로그램을 종료합니다.")
                sys.exit(1)

    if run_preparation:
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        log_info(f"   데이터 수집 기간: {start_date} ~ {end_date} ({years}년)")
        
        # 데이터 재생성 시 기존 imputation_values 파일 삭제
        imputation_values_dir = os.path.dirname(os.path.expanduser(data_path))
        imputation_values_path = os.path.join(imputation_values_dir, "imputation_values.joblib")
        if os.path.exists(imputation_values_path):
            try:
                os.remove(imputation_values_path)
                log_info(f"   🗑️ 기존 imputation_values 파일 삭제 완료 (데이터 재생성으로 인해)")
            except Exception as e:
                log_warning(f"   ⚠️ 기존 imputation_values 파일 삭제 실패: {e}")
        
        if not prepare_data_and_save(data_path, start_date, end_date):
            log_critical("데이터 준비에 실패하여 프로그램을 종료합니다.")
            sys.exit(1)

    # --- 2. 학습 설정 단계 ---
    log_info("\n--- ⚙️ 학습 설정 시작 ---")
    file_paths = glob.glob(os.path.join(data_path, "*.feather"))
    if not file_paths:
        log_critical("학습할 데이터 파일이 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    # file_paths 정렬 (fold 분할 일관성 보장)
    file_paths = sorted(file_paths)
    
    log_info(f"   총 {len(file_paths)}개 종목의 데이터를 학습에 사용합니다.")

    # 기존 CPU 버전과 동일하게, 검증된 핵심 피처 목록을 하드코딩
    features = [
        'PBR', 'log_mktcap', '이익수익률', 'BPS',
        '수익률(1M)', '수익률(3M)', '52주_신고가_비율',
        'ADX_14',
        '변동성(1W)', '변동성(1M)', '변동성(3M)', 'ATRr_14', 'BBW_20_2', 'BB_Position',
        'disparity_120', 'disparity_240',
        '거래대금_MA5', '거래대금_MA20', 'OBV',
        'KOSPI_pct_1d', 'KOSPI_pct_5d', 'USDKRW_pct_1d', 'USDKRW_pct_5d',
        'VIX_pct_1d', 'VIX_pct_5d'
    ]
    
    # imputation_values 파일 경로 설정 (data_path와 같은 레벨)
    imputation_values_dir = os.path.dirname(os.path.expanduser(data_path))
    imputation_values_path = os.path.join(imputation_values_dir, "imputation_values.joblib")
    
    # imputation_values 파일 존재 확인 및 로드/계산
    if os.path.exists(imputation_values_path):
        log_info("   ✅ imputation_values 파일을 찾았습니다. 로드 중...")
        try:
            imputation_values = joblib.load(imputation_values_path)
            log_info("   ✅ imputation_values 파일 로드 완료.")
        except Exception as e:
            log_warning(f"   ⚠️ imputation_values 파일 로드 실패: {e}. 재계산을 진행합니다.")
            # 파일 로드 실패 시 재계산
            imputation_values = None
    else:
        imputation_values = None
    
    # 파일이 없거나 로드 실패한 경우 계산
    if imputation_values is None:
        log_info("   전체 데이터로 중앙값 계산 중...")
        all_dfs = []
        batch_size = 20  # 메모리 효율을 위해 배치로 로드
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            batch_df = pd.concat([pd.read_feather(f) for f in batch_files])
            all_dfs.append(batch_df)
            del batch_df
            gc.collect()
            if (i + batch_size) % 500 == 0:
                log_info(f"      ... {min(i + batch_size, len(file_paths))}/{len(file_paths)} 파일 처리 중...")
        
        log_info(f"   전체 {len(file_paths)}개 파일 로딩 완료. 중앙값 계산 중...")
        all_df = pd.concat(all_dfs)
        del all_dfs
        gc.collect()
        
        numeric_features_df = all_df[features].select_dtypes(include=np.number)
        imputation_values = numeric_features_df.median().to_dict()
        
        log_info(f"   사용될 피처 개수: {len(features)}개 (기존 CPU 버전과 동일)")
        log_info("   ✅ 결측치 대체값(중앙값) 계산 완료 (전체 데이터 기준).")
        del all_df, numeric_features_df
        gc.collect()
        
        # 계산된 imputation_values 파일로 저장
        log_info(f"   💾 imputation_values 파일 저장 중... ({imputation_values_path})")
        try:
            joblib.dump(imputation_values, imputation_values_path)
            log_info("   ✅ imputation_values 파일 저장 완료.")
        except Exception as e:
            log_warning(f"   ⚠️ imputation_values 파일 저장 실패: {e}. 학습은 계속 진행됩니다.")

    # 재현성을 위한 RandomState 객체 생성
    rng = np.random.RandomState(42)

    # --- 3. Optuna 하이퍼파라미터 튜닝 ---
    study_name = "cuml_randomforest_optimization"
    # Pruning을 통해 나쁜 trial을 조기 종료하여 시간 절약
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # 처음 5개 trial은 pruning하지 않음
        n_warmup_steps=1,    # 첫 번째 fold만 완료하면 바로 pruning 판단 시작
        interval_steps=1      # 매 fold마다 pruning 체크
    )
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner
    )

    # --- 모든 Fold의 데이터를 미리 로드하여 캐싱 (중복 로드 방지) ---
    log_info(f"\n--- 📁 Fold 데이터 로딩 및 구성 중 ---")
    
    # fold_cache 디렉토리 경로 설정
    fold_cache_dir = os.path.join(os.path.dirname(os.path.expanduser(data_path)), "fold_cache")
    os.makedirs(fold_cache_dir, exist_ok=True)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    file_indices = np.arange(len(file_paths))
    fold_data_cache = {}  # {fold_idx: (X_train, y_train, X_val, y_val)}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(file_indices)):
        fold_cache_path = os.path.join(fold_cache_dir, f"fold_{fold}_data.joblib")
        
        # fold 데이터 파일 존재 확인
        if os.path.exists(fold_cache_path):
            log_info(f"   ✅ Fold #{fold+1}/3 캐시 파일을 찾았습니다. 로드 중...")
            load_start = datetime.now()
            try:
                fold_data = joblib.load(fold_cache_path)
                X_train, y_train, X_val, y_val = fold_data
                load_time = (datetime.now() - load_start).total_seconds()
                
                if X_train is None or X_val is None:
                    log_warning(f"   ⚠️ Fold #{fold+1} 캐시 파일 데이터가 유효하지 않습니다. 재로딩합니다.")
                    raise ValueError("Invalid cache data")
                
                fold_data_cache[fold] = (X_train, y_train, X_val, y_val)
                log_info(f"   ✅ Fold #{fold+1}/3 캐시 로드 완료: 훈련 {len(X_train):,}행, 검증 {len(X_val):,}행 ({load_time:.1f}초)")
            except Exception as e:
                log_warning(f"   ⚠️ Fold #{fold+1} 캐시 파일 로드 실패: {e}. 재로딩합니다.")
                # 캐시 파일이 손상된 경우 삭제 후 재로딩
                try:
                    os.remove(fold_cache_path)
                except:
                    pass
                fold_data = None
        else:
            fold_data = None
        
        # 캐시 파일이 없거나 로드 실패한 경우 기존 방식으로 로드
        if fold_data is None:
            val_files = [file_paths[i] for i in val_idx]
            train_files = [file_paths[i] for i in train_idx]
            
            log_info(f"   Fold #{fold+1}/3 데이터 로딩 중...")
            load_start = datetime.now()
            X_train, y_train = load_fold_data(train_files, features, imputation_values)
            X_val, y_val = load_fold_data(val_files, features, imputation_values)
            load_time = (datetime.now() - load_start).total_seconds()
            
            if X_train is None or X_val is None:
                log_warning(f"   ⚠️ Fold #{fold+1} 데이터 로딩 실패. 건너뜁니다.")
                continue
            
            fold_data_cache[fold] = (X_train, y_train, X_val, y_val)
            log_info(f"   ✅ Fold #{fold+1}/3 로딩 완료: 훈련 {len(X_train):,}행, 검증 {len(X_val):,}행 ({load_time:.1f}초)")
            
            # fold 데이터 파일로 저장
            log_info(f"   💾 Fold #{fold+1}/3 데이터 파일 저장 중...")
            try:
                joblib.dump((X_train, y_train, X_val, y_val), fold_cache_path)
                log_info(f"   ✅ Fold #{fold+1}/3 데이터 파일 저장 완료.")
            except Exception as e:
                log_warning(f"   ⚠️ Fold #{fold+1} 데이터 파일 저장 실패: {e}. 학습은 계속 진행됩니다.")
    
    log_info(f"   ✅ Fold 데이터 로딩 완료 (총 {len(fold_data_cache)}개 Fold)")
    
    if not fold_data_cache:
        log_critical("Fold 데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    log_info(f"\n--- 🤖 Optuna 하이퍼파라미터 최적화 시작 (n_trials={args.n_iter}) ---")

    try:
        study.optimize(
            lambda trial: objective(trial, fold_data_cache, features, imputation_values, args.max_depth, rng),
            n_trials=args.n_iter
        )
    finally:
        # 캐시는 최종 모델 훈련까지 유지 (중복 로딩 방지)
        pass

    log_info(f"\n--- 🏆 최적화 결과 | 최고 점수: {study.best_value:.4f} | 최적 파라미터: {study.best_params} ---")

    # 최적화 결과 및 학습 설정 구성
    optimization_results = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'total_combinations_tested': len(study.trials),
        'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    }
    
    training_config = {
        'n_iter': args.n_iter,
        'max_depth_candidates': args.max_depth,
        'cv_folds': 3,
        'scoring': 'roc_auc',
        'search_method': 'Optuna (TPE Sampler)',
        'n_streams': 1,
        'n_mini_batches': 1  # Optuna trial에서 단일 모델 사용
    }

    # --- 4. 최종 모델 훈련 및 저장 (캐시 데이터 재사용) ---
    try:
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_streams'] = 1  # GPU 병렬 처리 개선 (속도 향상)
        train_final_ensemble_model(fold_data_cache, features, imputation_values, best_params, rng, optimization_results, training_config)
    except Exception as e:
        log_critical("최종 모델 훈련 또는 저장 중 오류 발생", exception=e)
    finally:
        # 최종 모델 훈련 완료 후 데이터 캐시 정리 (함수 내에서 이미 삭제했지만 안전을 위해 재정리)
        for fold_key in list(fold_data_cache.keys()):
            fold_data = fold_data_cache[fold_key]
            if fold_data:
                try:
                    X_train, y_train, X_val, y_val = fold_data
                    del X_train, y_train, X_val, y_val
                except (ValueError, TypeError):
                    # 이미 삭제되었거나 None인 경우
                    pass
                fold_data_cache[fold_key] = None
        fold_data_cache.clear()
        safe_gpu_memory_cleanup()
        gc.collect()

    log_info("\n🎉 모든 작업이 성공적으로 완료되었습니다. 🎉")


if __name__ == '__main__':
    main()
