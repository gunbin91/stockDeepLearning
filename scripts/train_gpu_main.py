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

from logger import log_info, log_warning, log_error, log_critical

import data_processor

from path_manager import path_manager

# 언더샘플링은 직접 구현하므로 외부 라이브러리 불필요



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

        # 학습용 데이터 생성 시 팩터 점수 계산 건너뛰기 (백테스팅에서만 필요)
        full_df = data_processor.get_preprocessed_data(start_date, end_date, skip_factor_scores=True)

        log_memory_usage("전체 데이터 로딩 완료")

        if full_df is None or full_df.empty:
            log_error("데이터 전처리 중 오류가 발생하여 데이터를 생성할 수 없습니다.")
            return False

        log_info(f"   📊 전처리된 원본 데이터: {len(full_df):,}행")

        # target 필터링 (전체 데이터에서)
        log_info("   🔍 target 필터링 중...")
        full_df = full_df[full_df['target'].notna()].copy()
        log_info(f"   📊 target 필터링 후 데이터: {len(full_df):,}행")
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
    Optuna를 위한 objective 함수 (단일 모델 학습 방식).
    fold_data_cache: 모든 Fold의 데이터 캐시.
    각 fold에서 단일 모델을 학습하고 검증 점수를 계산합니다.
    """
    trial_start_time = datetime.now()
    log_info(f"\n{'='*60}")
    log_info(f"🚀 Optuna Trial #{trial.number} 시작")
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_categorical('max_depth', max_depth_list),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
        'max_samples': trial.suggest_categorical('max_samples', [0.7, 0.8, 0.9, 1.0]),
        'max_features': trial.suggest_float('max_features', 0.4, 1.0),
        'split_criterion': trial.suggest_categorical('split_criterion', [0, 1]),
        'random_state': 42,
        'n_streams': 1,
    }

    log_info(f"   📋 파라미터: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, max_samples={params['max_samples']}, max_features={params['max_features']}")

    if not fold_data_cache:
        log_error("Fold 데이터 캐시가 비어있습니다. Trial을 중단합니다.")
        return 0.0

    fold_scores = []
    # --- 각 Fold 처리 (캐시된 데이터 사용) ---
    for fold in range(len(fold_data_cache)):
        fold_start_time = datetime.now()
        
        if fold > 0:
            safe_gpu_memory_cleanup(); gc.collect()

        X_train_all, y_train_all, X_val, y_val = fold_data_cache[fold]
        
        if X_train_all is None or X_val is None:
            log_warning(f"   ⚠️ Fold #{fold+1} 데이터가 없습니다. 건너뜁니다.")
            continue

        # 스케일러 학습 및 데이터 변환
        step_start = datetime.now()
        scaler = cuStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_all)
        X_val_scaled = scaler.transform(X_val)
        preprocessing_time = (datetime.now() - step_start).total_seconds()
        
        del X_train_all, X_val
        safe_gpu_memory_cleanup(); gc.collect()

        # 단일 모델 학습
        step_start = datetime.now()
        model = cuRF(**params)
        model.fit(X_train_scaled, y_train_all)
        fit_time = (datetime.now() - step_start).total_seconds()
        
        del X_train_scaled, y_train_all
        safe_gpu_memory_cleanup(); gc.collect()
        
        # 예측
        step_start = datetime.now()
        try:
            pred_proba_cudf = model.predict_proba(X_val_scaled)
            y_pred_proba = pred_proba_cudf.iloc[:, 1]
            pred_time = (datetime.now() - step_start).total_seconds()
        except Exception as e:
            log_error(f"   ❌ Fold #{fold+1} 예측 실패: {e}")
            del model, X_val_scaled, y_val
            safe_gpu_memory_cleanup(); gc.collect()
            continue # 다음 fold로 진행

        # ROC-AUC 점수 계산
        score = roc_auc_score(y_val, y_pred_proba)
        fold_scores.append(score)
        
        # 사용 완료된 객체 정리
        del scaler, model, X_val_scaled, y_val, y_pred_proba, pred_proba_cudf
        safe_gpu_memory_cleanup(); gc.collect()

        fold_duration = (datetime.now() - fold_start_time).total_seconds()
        log_info(f"   ✅ Fold #{fold+1}/3 | Score: {score:.4f} | 총: {fold_duration:.1f}초 "
                 f"(전처리: {preprocessing_time:.1f}초, 학습: {fit_time:.1f}초, 예측: {pred_time:.1f}초)")

        # Pruning 체크
        trial.report(score, step=fold)
        if trial.should_prune():
            trial_duration = (datetime.now() - trial_start_time).total_seconds()
            log_info(f"   ⏹️ Trial #{trial.number} 조기 종료 (Pruning) | 총 소요시간: {trial_duration/60:.1f}분 ({trial_duration:.1f}초)")
            log_info(f"{'='*60}")
            safe_gpu_memory_cleanup(); gc.collect()
            raise optuna.TrialPruned()

    if not fold_scores:
        log_error("모든 Fold에서 학습에 실패했습니다. Trial을 중단합니다.")
        return 0.0

    mean_score = np.mean(fold_scores)
    trial_duration = (datetime.now() - trial_start_time).total_seconds()
    log_info(f"✅ Trial #{trial.number} 완료 | 평균 Score: {mean_score:.4f} | "
            f"총 소요시간: {trial_duration/60:.1f}분 ({trial_duration:.1f}초)")
    log_info(f"{'='*60}")

    return mean_score

def train_final_ensemble_model(fold_data_cache, features, imputation_values, best_params, rng, optimization_results=None, training_config=None, data_path=None):
    """
    전체 데이터를 사용하여 최종 단일 모델을 훈련하고 저장합니다.
    (메모리 최적화 버전: 모델을 순차적으로 학습/예측하여 GPU 메모리 사용량 최소화)
    """
    log_info("\n--- 🚂 최적 파라미터로 최종 단일 모델 훈련 시작 ---")
    
    try:
        # 1. 원본 데이터 로드 (fold_cache 파일에서 직접 로드하여 데이터 누실 방지)
        log_info("   [DATA] 원본 데이터 로드 중 (fold_cache 파일에서)...")
        compose_start = datetime.now()
        
        # fold_cache 디렉토리 경로 계산
        if data_path is None:
            # data_path가 제공되지 않은 경우, fold_data_cache에서 추론 불가능하므로 에러
            raise ValueError("data_path가 제공되지 않았습니다. fold_cache 파일을 로드할 수 없습니다.")
        
        fold_cache_dir = os.path.join(os.path.dirname(os.path.expanduser(data_path)), "fold_cache")
        fold_cache_path = os.path.join(fold_cache_dir, "fold_0_data.joblib")
        
        if not os.path.exists(fold_cache_path):
            raise FileNotFoundError(f"fold_cache 파일을 찾을 수 없습니다: {fold_cache_path}")
        
        # 원본 데이터 로드 (샘플링 전 데이터)
        fold_data = joblib.load(fold_cache_path)
        X_train_original, y_train_original, X_val_original, y_val_original = fold_data
        
        if X_train_original is None or X_val_original is None:
            raise ValueError("fold_cache 파일의 데이터가 유효하지 않습니다.")
        
        log_info(f"   [OK] 원본 데이터 로드 완료: Train {len(X_train_original):,}행, Val {len(X_val_original):,}행")
        
        # 원본 train + 원본 val 합치기 (전체 데이터 복원)
        X_all = cudf.concat([X_train_original, X_val_original], ignore_index=True)
        y_all = cudf.concat([y_train_original, y_val_original], ignore_index=True)
        
        # 원본 데이터 메모리 해제
        del X_train_original, y_train_original, X_val_original, y_val_original, fold_data
        safe_gpu_memory_cleanup(); gc.collect()
        
        # 원본 데이터 클래스 분포 확인
        y_all_pandas = y_all.to_pandas()
        value_counts_original = y_all_pandas.value_counts()
        minority_class_label = value_counts_original.idxmin()
        majority_class_label = value_counts_original.idxmax()
        n_minority_original = value_counts_original[minority_class_label]
        n_majority_original = value_counts_original[majority_class_label]
        minority_class_name = "상승" if minority_class_label == 1 else "하락"
        majority_class_name = "상승" if majority_class_label == 1 else "하락"
        
        log_info(f"   [OK] 전체 데이터 구성 완료: {len(X_all):,}행 ({(datetime.now() - compose_start).total_seconds():.1f}초)")
        log_info(f"   [DATA] 원본 데이터 클래스 분포:")
        log_info(f"      - 소수 클래스 ({minority_class_label}, {minority_class_name}): {n_minority_original:,}개")
        log_info(f"      - 다수 클래스 ({majority_class_label}, {majority_class_name}): {n_majority_original:,}개")
        
        del y_all_pandas

        # 2. 전체 데이터 언더샘플링 적용 (클래스 불균형 해결)
        log_info("   [SAMPLING] 전체 데이터 언더샘플링 진행 중...")
        sampling_start = datetime.now()
        
        if n_majority_original > n_minority_original:
            # cuDF에서 직접 boolean 인덱싱으로 위치 인덱스 추출
            majority_mask = (y_all == majority_class_label).to_pandas().values
            minority_mask = (y_all == minority_class_label).to_pandas().values
            
            # 위치 인덱스 생성 (0부터 시작)
            all_indices = np.arange(len(y_all))
            majority_indices = all_indices[majority_mask]
            minority_indices = all_indices[minority_mask]
            
            # 소수 클래스 개수만큼 랜덤 샘플링
            rng_final = np.random.RandomState(42)  # 최종 학습용 고정 시드
            sampled_majority_indices = rng_final.choice(majority_indices, size=n_minority_original, replace=False)
            
            # 언더샘플링된 인덱스 결합
            balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
            
            # 셔플 (numpy 배열에서 직접)
            rng_final.shuffle(balanced_indices)
            
            # cuPy 배열로 변환하여 cuDF iloc에 사용
            balanced_indices_cupy = cp.asarray(balanced_indices)
            
            # 언더샘플링된 데이터 생성
            X_all_resampled = X_all.iloc[balanced_indices_cupy].reset_index(drop=True)
            y_all_resampled = y_all.iloc[balanced_indices_cupy].reset_index(drop=True)
            
            # 결과 확인
            y_all_resampled_pandas = y_all_resampled.to_pandas()
            value_counts_resampled = y_all_resampled_pandas.value_counts()
            n_minority_resampled = value_counts_resampled[minority_class_label]
            n_majority_resampled = value_counts_resampled[majority_class_label]
            del y_all_resampled_pandas, majority_mask, minority_mask, all_indices, majority_indices, minority_indices, balanced_indices, balanced_indices_cupy
            
            # 원본 데이터 삭제
            del X_all, y_all
            safe_gpu_memory_cleanup(); gc.collect()
            
            # 샘플링된 데이터로 교체
            X_all = X_all_resampled
            y_all = y_all_resampled
            del X_all_resampled, y_all_resampled
            
            log_info(f"   [OK] 언더샘플링 완료: 소수 클래스 {n_minority_original:,}개 → {n_minority_resampled:,}개, 다수 클래스 {n_majority_original:,}개 → {n_majority_resampled:,}개")
            log_info(f"   [DATA] 샘플링 후 데이터: {len(X_all):,}행 ({(datetime.now() - sampling_start).total_seconds():.1f}초)")
        else:
            log_info(f"   ℹ️ 클래스 불균형이 없어 샘플링을 건너뜁니다.")
            log_info(f"   [DATA] 최종 학습 데이터: {len(X_all):,}행")

        # 3. 전체 데이터 전처리
        step_start = datetime.now()
        log_info("   [PREPROC] 데이터 스케일링 중...")
        final_scaler = cuStandardScaler()
        final_scaler.fit(X_all)
        X_all_scaled = final_scaler.transform(X_all)
        del X_all
        safe_gpu_memory_cleanup(); gc.collect()
        log_info(f"   [OK] 전처리 완료 ({(datetime.now() - step_start).total_seconds():.1f}초)")

        # 4. Teacher 모델 학습/예측 및 Soft Label 생성 (보류)
        # [보류] Teacher 모델/예측 Soft라벨 방식 보류 => 언더샘플링 방식으로 변경
        # OOM 방지 및 메모리 효율성을 위해 Teacher 모델 학습/예측 로직은 주석 처리하고,
        # 바로 언더샘플링 방식으로 최종 모델을 학습합니다.
        log_info("   [INFO] Teacher 모델/예측 Soft라벨 방식 보류 => 언더샘플링 방식으로 변경")
        
        # ===== 주석 처리된 Teacher 모델 학습/예측 로직 (보류) =====
        # soft_label_start = datetime.now()
        # 
        # n_chunks = 6
        # chunk_size = len(X_all_scaled) // n_chunks
        # # CPU에 예측 확률 누적 (메모리 최적화)
        # accumulated_probas = np.zeros(len(X_all_scaled), dtype=np.float32)
        # ensemble_fit_time = 0.0
        # 
        # for chunk_idx in range(n_chunks):
        #     log_info(f"\n   --- Teacher 모델 {chunk_idx + 1}/{n_chunks} 학습 및 예측 ---")
        #     
        #     # --- 가. Teacher 모델 학습 ---
        #     fit_start = datetime.now()
        #     start_idx = chunk_idx * chunk_size
        #     end_idx = len(X_all_scaled) if chunk_idx == n_chunks - 1 else (chunk_idx + 1) * chunk_size
        #     
        #     X_chunk = X_all_scaled.iloc[start_idx:end_idx]
        #     y_chunk = y_all.iloc[start_idx:end_idx]
        # 
        #     chunk_model = cuRF(**best_params)
        #     chunk_model.fit(X_chunk, y_chunk)
        #     del X_chunk, y_chunk
        #     safe_gpu_memory_cleanup(); gc.collect()
        #     
        #     fit_duration = (datetime.now() - fit_start).total_seconds()
        #     ensemble_fit_time += fit_duration
        #     log_info(f"      [OK] 모델 학습 완료 ({fit_duration:.1f}초)")
        # 
        #     # --- 나. 전체 데이터에 대한 예측 (배치 처리) ---
        #     predict_start = datetime.now()
        #     log_info("      [PRED] 전체 데이터셋에 대한 예측 진행 중...")
        #     
        #     # 예측을 위한 배치 크기 설정 (GPU 메모리 상황에 맞게 조절)
        #     pred_batch_size = 500000 
        #     num_batches = (len(X_all_scaled) + pred_batch_size - 1) // pred_batch_size
        #     
        #     probas_for_model = np.zeros(len(X_all_scaled), dtype=np.float32)
        # 
        #     for i in range(num_batches):
        #         batch_start_idx = i * pred_batch_size
        #         batch_end_idx = min((i + 1) * pred_batch_size, len(X_all_scaled))
        #         
        #         X_batch = X_all_scaled.iloc[batch_start_idx:batch_end_idx]
        #         
        #         try:
        #             # predict_proba는 cudf.DataFrame [class_0_prob, class_1_prob] 반환
        #             batch_probas = chunk_model.predict_proba(X_batch)
        #             # to_numpy()로 CPU로 바로 가져옴
        #             probas_for_model[batch_start_idx:batch_end_idx] = batch_probas.iloc[:, 1].to_numpy()
        #         except Exception as e:
        #             log_error(f"      [FAIL] 배치 {i+1}/{num_batches} 예측 실패: {e}")
        #             # 실패 시 중립값(0.5)으로 채워넣기
        #             probas_for_model[batch_start_idx:batch_end_idx] = 0.5
        #         finally:
        #             del X_batch, batch_probas
        #             safe_gpu_memory_cleanup()
        #     
        #     # 현재 모델의 예측 확률을 누적
        #     accumulated_probas += probas_for_model
        #     
        #     predict_duration = (datetime.now() - predict_start).total_seconds()
        #     log_info(f"      [OK] 예측 완료 ({predict_duration:.1f}초)")
        # 
        #     # --- 다. Teacher 모델 및 관련 객체 즉시 삭제 ---
        #     del chunk_model, probas_for_model
        #     log_info("      [CLEAN] Teacher 모델 메모리 해제 완료")
        #     safe_gpu_memory_cleanup(); gc.collect()
        # 
        # # --- 라. Soft Label 계산 완료 및 메모리 정리 ---
        # # Soft Label은 이제 최종 모델 학습에 직접 사용되지 않으므로 계산 없이 바로 삭제합니다.
        # del accumulated_probas
        # safe_gpu_memory_cleanup(); gc.collect()
        # log_info(f"\n   [OK] 참고용 Soft Label 계산 로직 완료. 이제 OOM 방지 및 클래스 균형을 위해 언더샘플링을 진행합니다. (총 소요시간: {(datetime.now() - soft_label_start).total_seconds():.1f}초)")
        # ===== 주석 처리 완료 =====

        # 5. 최종 단일 모델 학습 (언더샘플링된 전체 데이터 사용)
        log_info("   [TRAIN] 최종 단일 모델 학습 시작...")
        step_start = datetime.now()
        final_model = cuRF(**best_params)
        # 이미 샘플링된 데이터로 최종 모델 학습
        final_model.fit(X_all_scaled, y_all)
        fit_time = (datetime.now() - step_start).total_seconds()
        
        log_info(f"   [OK] 최종 모델 학습 완료 (학습 시간: {fit_time:.1f}초)")
        
        # 6. SHAP를 사용한 피처 중요도 계산
        feature_importances = None
        if SHAP_AVAILABLE:
            log_info("   [SHAP] 피처 중요도 계산 중...")
            shap_start = datetime.now()
            X_sample_cudf = None
            try:
                # 샘플링된 데이터의 일부를 사용
                sample_size = min(1000, max(100, len(X_all_scaled) // 10))
                sample_indices = cp.random.choice(len(X_all_scaled), sample_size, replace=False)
                X_sample_cudf = X_all_scaled.iloc[sample_indices.get()]
                
                # SHAP는 numpy 배열을 선호
                X_sample_np = X_sample_cudf.to_numpy()

                def model_predict_proba_wrapper(X_np):
                    X_cudf = cudf.DataFrame(X_np, columns=features)
                    try:
                        probas = final_model.predict_proba(X_cudf)
                        return probas.iloc[:, 1].to_numpy()
                    finally:
                        del X_cudf

                explainer = shap.KernelExplainer(model_predict_proba_wrapper, shap.sample(X_sample_np, 50))
                shap_values = explainer.shap_values(X_sample_np, nsamples=100)
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importances = sorted(zip(features, mean_abs_shap), key=lambda x: x[1], reverse=True)
                
                log_info(f"   [OK] SHAP 계산 완료 ({(datetime.now() - shap_start).total_seconds():.1f}초)")
                del X_sample_np, explainer, shap_values, mean_abs_shap
                
            except Exception as e:
                log_warning(f"   [WARN] SHAP 피처 중요도 계산 실패: {e}")
            finally:
                if X_sample_cudf is not None:
                    del X_sample_cudf
                safe_gpu_memory_cleanup(); gc.collect()
        else:
            log_warning("   [WARN] SHAP 라이브러리가 없어 피처 중요도를 계산할 수 없습니다.")

        # 7. 최종 모델 저장
        if final_model:
            log_info("   [SAVE] 최종 단일 모델 및 전처리기 저장 중...")
            model_path = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
            metadata_path = str(path_manager.data_dir / 'cuml_ensemble_model_metadata.joblib')
            
            # 최종 training_config 구성 (n_final_models 추가)
            final_training_config = {**(training_config or {}), 'n_final_models': 1}
            
            # 모델 파일 저장
            joblib.dump({
                'model': final_model,
                'features': features,
                'scaler': final_scaler,
                'imputation_values': imputation_values,
                'best_params': best_params,
                'model_type': 'single_model',
                'optimization_results': optimization_results or {},
                'training_config': final_training_config,
                'feature_importances': feature_importances
            }, model_path, compress=3)
            log_info(f"   [OK] 최종 모델 저장 완료: {model_path}")
            
            # 메타데이터 파일 저장 (웹페이지에서 빠른 로드를 위해)
            try:
                parameter_explanations = {
                    'n_estimators': 'RandomForest가 만들 트리의 개수',
                    'max_depth': '각 트리의 최대 깊이 (과적합 방지)',
                    'min_samples_split': '노드 분할에 필요한 최소 샘플 수',
                    'min_samples_leaf': '리프 노드의 최소 샘플 수',
                    'max_samples': '각 트리가 사용할 샘플 비율',
                    'max_features': '각 분할에서 사용할 최대 피처 비율',
                    'split_criterion': '분할 기준 (0: Gini, 1: Entropy)'
                }
                
                metadata_to_save = {
                    'features': features,
                    'best_params': best_params,
                    'model_type': 'single_model',
                    'optimization_results': optimization_results or {},
                    'training_config': final_training_config,
                    'feature_importances': feature_importances,
                    'parameter_explanations': parameter_explanations
                }
                
                joblib.dump(metadata_to_save, metadata_path, compress=3)
                log_info(f"   [OK] 메타데이터 파일 저장 완료: {metadata_path}")
            except Exception as e:
                log_warning(f"   [WARN] 메타데이터 파일 저장 실패 (선택사항): {e}")
        else:
            log_error("   [FAIL] 최종 모델이 생성되지 않아 저장할 수 없습니다.")

    except Exception as e:
        log_critical("최종 모델 훈련 또는 저장 중 심각한 오류 발생", exception=e)
    finally:
        # 모든 주요 변수 정리
        if 'final_model' in locals(): del final_model
        if 'final_scaler' in locals(): del final_scaler
        if 'X_all_scaled' in locals(): del X_all_scaled
        if 'y_all' in locals(): del y_all
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

    # --- Trial 전 언더샘플링 적용 (모든 fold의 train 데이터에 적용) ---
    log_info(f"\n--- 🔄 Trial 전 언더샘플링 적용 중 ---")
    step_start = datetime.now()
    
    for fold in range(len(fold_data_cache)):
        X_train, y_train, X_val, y_val = fold_data_cache[fold]
        
        # 클래스 분포 확인
        y_train_pandas = y_train.to_pandas()
        value_counts = y_train_pandas.value_counts()
        
        if len(value_counts) < 2:
            log_warning(f"   ⚠️ Fold #{fold+1}에 클래스가 1개만 있어 샘플링을 건너뜁니다.")
            continue
        
        minority_class_label = value_counts.idxmin()
        majority_class_label = value_counts.idxmax()
        n_minority = value_counts[minority_class_label]
        n_majority = value_counts[majority_class_label]
        
        # 클래스 레이블을 의미있는 문자열로 변환
        minority_class_name = "상승" if minority_class_label == 1 else "하락"
        majority_class_name = "상승" if majority_class_label == 1 else "하락"
        
        log_info(f"   Fold #{fold+1}/3 클래스 분포:")
        log_info(f"      - 소수 클래스 ({minority_class_label}, {minority_class_name}) 샘플 수: {n_minority:,}개")
        log_info(f"      - 다수 클래스 ({majority_class_label}, {majority_class_name}) 샘플 수: {n_majority:,}개")
        
        # 언더샘플링: 다수 클래스를 소수 클래스 개수만큼만 남김
        if n_majority > n_minority:
            # cuDF에서 직접 boolean 인덱싱으로 위치 인덱스 추출
            # cuDF Series는 위치 기반 인덱싱을 사용하므로, boolean mask로 필터링 후 인덱스 추출
            majority_mask = (y_train == majority_class_label).to_pandas().values
            minority_mask = (y_train == minority_class_label).to_pandas().values
            
            # 위치 인덱스 생성 (0부터 시작)
            all_indices = np.arange(len(y_train))
            majority_indices = all_indices[majority_mask]
            minority_indices = all_indices[minority_mask]
            
            # 소수 클래스 개수만큼 랜덤 샘플링
            rng = np.random.RandomState(42 + fold)  # fold별로 다른 시드 사용
            sampled_majority_indices = rng.choice(majority_indices, size=n_minority, replace=False)
            
            # 언더샘플링된 인덱스 결합
            balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
            
            # 셔플 (numpy 배열에서 직접)
            rng.shuffle(balanced_indices)
            
            # cuPy 배열로 변환하여 cuDF iloc에 사용
            balanced_indices_cupy = cp.asarray(balanced_indices)
            
            # 언더샘플링된 데이터 생성
            X_train_resampled = X_train.iloc[balanced_indices_cupy].reset_index(drop=True)
            y_train_resampled = y_train.iloc[balanced_indices_cupy].reset_index(drop=True)
            
            # 결과 확인 (삭제 전에 확인)
            y_train_resampled_pandas = y_train_resampled.to_pandas()
            value_counts_resampled = y_train_resampled_pandas.value_counts()
            n_minority_resampled = value_counts_resampled[minority_class_label]
            n_majority_resampled = value_counts_resampled[majority_class_label]
            del y_train_resampled_pandas
            
            # fold_data_cache 업데이트 (검증 데이터는 그대로 유지)
            fold_data_cache[fold] = (X_train_resampled, y_train_resampled, X_val, y_val)
            
            # 원본 데이터 삭제
            del X_train, y_train, balanced_indices, balanced_indices_cupy
            safe_gpu_memory_cleanup(); gc.collect()
            
            log_info(f"      ✅ 언더샘플링 완료: 소수 클래스 {n_minority:,}개 → {n_minority_resampled:,}개, 다수 클래스 {n_majority:,}개 → {n_majority_resampled:,}개")
        else:
            log_info(f"      ℹ️ 클래스 불균형이 없어 샘플링을 건너뜁니다.")
    
    log_info(f"   ✅ Trial 전 언더샘플링 완료: 총 {len(fold_data_cache)}개 Fold | 소요시간: {(datetime.now() - step_start).total_seconds():.1f}초")
    
    # 주의: 샘플링된 데이터는 메모리에서만 사용하고, fold_cache 파일은 원본 데이터로 유지합니다.
    # 다음 실행 시 원본 데이터를 로드하여 일관성 있는 샘플링을 보장합니다.

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
        'n_mini_batches': 1,  # Optuna trial에서 단일 모델 사용
        # 모델 목표 정보 (메타데이터)
        'target_days': 10,  # 거래일 기준
        'target_percentage': 8,  # 퍼센트
        'target_description': '10거래일 사이 한번이라도 8% 상승이 있었는지 확인'
    }

    # --- 4. 최종 모델 훈련 및 저장 (캐시 데이터 재사용) ---
    try:
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_streams'] = 1  # GPU 병렬 처리 개선 (속도 향상)
        train_final_ensemble_model(fold_data_cache, features, imputation_values, best_params, rng, optimization_results, training_config, data_path)
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
