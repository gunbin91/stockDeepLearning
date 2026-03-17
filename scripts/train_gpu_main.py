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
from concurrent.futures import ThreadPoolExecutor, as_completed

# cudf의 feather reader 관련 UserWarning은 예상된 동작이므로 숨김 처리
warnings.filterwarnings("ignore", message="Using CPU via PyArrow to read feather dataset", category=UserWarning)

# 서드파티 라이브러리
import pandas as pd

# pandas 호환성 패치: cuDF가 pandas.api.types.is_interval을 사용하는데 최신 pandas에서는 제거됨
def apply_pandas_compatibility_patch():
    """pandas 최신 버전과 cuDF 호환성을 위한 패치 적용"""
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
        # 패치 적용 실패해도 계속 진행 (cuDF가 직접 처리할 수 있음)
        pass

# scikit-learn 호환성 패치: cuML이 BaseEstimator._get_default_requests를 사용하는데 최신 scikit-learn에서는 제거됨
def apply_sklearn_compatibility_patch():
    """scikit-learn 최신 버전과 cuML 호환성을 위한 패치 적용"""
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
        # 패치 적용 실패해도 계속 진행
        pass

# 호환성 패치 적용 (cudf/cuml import 전에 실행)
apply_pandas_compatibility_patch()
apply_sklearn_compatibility_patch()

import cudf
import cuml
import numpy as np
import joblib
import optuna
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
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
    """안전하게 GPU 메모리를 정리하고 VRAM 파편화를 최소화합니다."""
    try:
        # 1. Python GC 강제 실행 (여러 번 실행하여 순환 참조 해제)
        for _ in range(3):
            gc.collect()
        
        # 2. CUDA 컨텍스트 동기화 (진행 중인 작업 완료 대기)
        try:
            cuda.synchronize()
        except Exception:
            pass  # Numba 컨텍스트가 없는 경우 무시
        
        # 3. cuPy 메모리 풀 완전 정리
        try:
            # 기본 메모리 풀 정리
            cp.get_default_memory_pool().free_all_blocks()
            # 피니드 메모리 풀도 정리 (있는 경우)
            if hasattr(cp, 'get_default_pinned_memory_pool'):
                try:
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception:
            pass  # cuPy 메모리 풀 정리 실패 시 무시
        
        # 4. cuDF 내부 캐시 정리 (있는 경우)
        try:
            if hasattr(cudf, '_cached_data'):
                cudf._cached_data.clear()
        except Exception:
            pass
        
        # 5. cuML의 메모리 정리 API (버전에 따라 다를 수 있음)
        try:
            # cuML 24.04+ 버전
            if hasattr(cuml, 'utils') and hasattr(cuml.utils, 'memory_utils'):
                if hasattr(cuml.utils.memory_utils, 'rts'):
                    cuml.utils.memory_utils.rts.cuda_free_memory()
        except (AttributeError, Exception):
            pass  # API가 없거나 실패 시 무시
        
        # 6. 추가 GC 실행 (메모리 정리 후)
        gc.collect()
        
    except Exception as e:
        log_warning(f"   ⚠️ GPU 메모리 정리 중 오류: {e}")


def check_gpu_memory_fragmentation():
    """
    GPU 메모리 파편화 정도를 확인합니다.
    Returns:
        tuple: (is_fragmented: bool, fragmentation_info: dict)
    """
    try:
        # GPU 메모리 사용량 확인
        gpu_mem_used = get_memory_usage(gpu=True)
        
        # GPU 메모리 총량 확인
        try:
            command = "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
            result = subprocess.check_output(command, shell=True, encoding='utf-8')
            parts = result.strip().split('\n')[0].split(', ')
            gpu_total_mb = int(parts[0])
            gpu_free_mb = int(parts[1])
        except Exception:
            return False, {}
        
        # 메모리 사용률 계산
        usage_pct = (gpu_mem_used / gpu_total_mb * 100) if gpu_total_mb > 0 else 0
        
        # 파편화 추정: 사용률이 높은데 큰 메모리 할당이 실패할 가능성
        # 실제로는 테스트 할당을 해봐야 정확하지만, 여기서는 사용률 기반으로 추정
        is_fragmented = False
        fragmentation_reason = ""
        
        # 사용률이 80% 이상이면 파편화 가능성 높음
        if usage_pct > 80:
            is_fragmented = True
            fragmentation_reason = f"메모리 사용률이 높음 ({usage_pct:.1f}%)"
        # 사용 가능한 메모리가 적은데 사용률이 높으면 파편화 가능성
        elif gpu_free_mb < 1000 and usage_pct > 60:
            is_fragmented = True
            fragmentation_reason = f"사용 가능 메모리 부족 ({gpu_free_mb}MB) 및 사용률 높음 ({usage_pct:.1f}%)"
        
        info = {
            'total_mb': gpu_total_mb,
            'used_mb': gpu_mem_used,
            'free_mb': gpu_free_mb,
            'usage_pct': usage_pct,
            'is_fragmented': is_fragmented,
            'reason': fragmentation_reason
        }
        
        return is_fragmented, info
        
    except Exception as e:
        log_warning(f"   ⚠️ GPU 메모리 파편화 확인 중 오류: {e}")
        return False, {}


def calculate_dynamic_batch_size(model_params, val_size):
    """
    모델 복잡도에 따른 동적 배치 크기 계산 (VRAM 8GB 기준)
    
    max_samples·max_features·min_samples_leaf을 반영해
    트리 수/깊이가 크고 샘플을 많이 쓰는 모델일수록 더 작은 배치로 조정.
    
    Returns:
        batch_size: 배치 크기 (None이면 배치 처리 불필요)
    """
    n_est = model_params.get('n_estimators', 300)
    max_d = model_params.get('max_depth', 20)
    max_samples = model_params.get('max_samples', 1.0) or 1.0
    max_features = model_params.get('max_features', 1.0) or 1.0
    min_leaf = model_params.get('min_samples_leaf', 32) or 32
    
    # 복잡도 점수: 트리 규모 * 입력 크기
    # - max_samples, max_features가 클수록, leaf가 작을수록 점수 ↑
    # - leaf가 0/None이면 32로 보호
    min_leaf = max(1, min_leaf)
    complexity_score = (
        n_est * max_d * max_samples * max_features * (32.0 / min_leaf) * val_size
    )
    
    # 기본 분기 (8GB 기준)
    if complexity_score < 2_000_000_000:          # 20억 미만
        batch_size = None                          # 배치 불필요
    elif complexity_score < 5_000_000_000:        # 50억 미만
        batch_size = 250_000
    elif complexity_score < 10_000_000_000:       # 100억 미만
        batch_size = 150_000
    else:                                         # 100억 이상
        batch_size = 100_000
    
    # 위험 플래그: 두 개 이상이면 한 단계 추가 축소
    risk_flags = 0
    if max_samples >= 0.9:
        risk_flags += 1
    if min_leaf < 32:
        risk_flags += 1
    if (n_est * max_d) > 7_000:
        risk_flags += 1
    
    if risk_flags >= 2 and batch_size is not None:
        if batch_size > 180_000:
            batch_size = 150_000
        elif batch_size > 120_000:
            batch_size = 100_000
    
    return batch_size


def enhanced_gpu_memory_cleanup(force_defrag=False):
    """
    향상된 GPU 메모리 정리 함수 (파편화 모니터링 포함)
    
    Args:
        force_defrag: True이면 파편화 확인 후 강제 정리 수행
    """
    # 기본 메모리 정리
    safe_gpu_memory_cleanup()
    
    # 파편화 확인
    if force_defrag:
        is_fragmented, frag_info = check_gpu_memory_fragmentation()
        
        if is_fragmented:
            log_warning(f"   ⚠️ GPU 메모리 파편화 감지: {frag_info.get('reason', '알 수 없음')}")
            log_info(f"   💾 메모리 상태: 사용 {frag_info.get('used_mb', 0):.0f}MB / 총 {frag_info.get('total_mb', 0):.0f}MB ({frag_info.get('usage_pct', 0):.1f}%)")
            
            # 추가 정리 수행
            try:
                # 추가 GC 실행
                for _ in range(5):
                    gc.collect()
                
                # cuPy 메모리 풀 강제 정리
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                
                # CUDA 동기화
                try:
                    cuda.synchronize()
                except Exception:
                    pass
                
                log_info("   ✅ 추가 메모리 정리 완료")
            except Exception as e:
                log_warning(f"   ⚠️ 추가 메모리 정리 중 오류: {e}")


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
    
    웜업 기간을 고려하여 실제 수집 시작일보다 1년 전부터 데이터를 수집합니다.
    결측치는 저장하지 않고, 나중에 Fold별로 처리합니다.

    """

    log_info("🚀 실시간 데이터 수집 및 전처리를 시작합니다...")

    log_memory_usage("데이터 전처리 시작")

    # 웜업 기간 설정: 가장 긴 window (250일) + 여유 (115일) = 총 365일
    WARMUP_DAYS = 365
    
    # 실제 수집 시작일: 사용자 입력 시작일보다 1년 전부터 수집
    start_date_obj = pd.to_datetime(start_date)
    actual_start_date = (start_date_obj - timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    
    log_info(f"   📅 웜업 기간: {WARMUP_DAYS}일")
    log_info(f"   📅 사용자 입력 시작일: {start_date}")
    log_info(f"   📅 실제 데이터 수집 시작일: {actual_start_date} (웜업 기간 포함)")

    # 기존 CPU 버전과 동일하게, 검증된 핵심 피처 목록을 공용 설정에서 로드
    # 제거된 피처: PBR, USDKRW_pct_1d, KOSPI_pct_1d, 이익수익률, 수익률(3M), 수익률(1M), ATRr_14
    # 제거된 피처: KOSPI_disparity_240, USDKRW_pct_5d, VIX_pct_1d, VIX_pct_5d, KOSPI_pct_5d
    # 추가된 피처: KOSPI_disparity_60
    # 추가된 피처: disparity_60
    # 제거된 피처: BBW_20_2, 거래량 변동성 계수, 변동성 기울기, 거래량 변동성 계수 기울기
    # 제거된 피처: 등락율, 주가_기울기(1W/1M/3M), disparity_5/10, 변동성(5M), KOSPI_변동성(3D/5M), KOSPI_disparity_5
    # 제거된 피처: 변동성(3D/3M/1M/1W), KOSPI_변동성(3M/1W), KOSPI_disparity_120/20, BPS, 거래대금_MA20/MA5, BB_Position
    # 제거된 피처: disparity_120, disparity_240
    # 변경된 피처: RSI_14 -> RSI_Signal_Oscillator
    # 추가된 피처: Relative_Strength_20 (KOSPI 수익률 상대강도), 시총_회전율(1W) (5일 거래대금 기준)
    # 제거된 피처: 변동성(1W), 변동성(3M) (2024년 12월)
    features = [
        'log_mktcap',
        '52주_신고가_비율',
        'ADX_14',
        'disparity_120',  # 120일 이격도
        'disparity_240',  # 240일 이격도
        'disparity_20',   # 20일 이격도
        'KOSPI_disparity_20',  # KOSPI 20일 이격도
        # 추가된 피처
        'Trend_Pullback_Score',  # 추세+눌림 점수
        'Position_Range_60',
        # 'KOSPI_변동성(1M)',  # 2024년 12월 제거
        # 변동성(1W), 변동성(3M) 제거됨 (2024년 12월)
        'MA20_Slope',  # 20일 이동평균선 기울기
        'MA120_Slope',  # 120일 이동평균선 기울기
        'MA240_Slope',  # 240일 이동평균선 기울기
        'KOSPI_MA20_Slope',  # KOSPI 20일 이동평균선 기울기
        # 'PBR_log',  # PBR 로그 변환 (2024년 12월 제거)
        # 새로 추가된 피처
        'RVOL',  # 상대 거래량 (Relative Volume)
        'RVOL(1W)',  # 5일/20일 상대 거래량
        '시총 회전율(1W)',  # 시총 회전율 1주 (5일 평균 거래대금 / 시가총액 * 100)
        '시총 회전율(3M)',  # 시총 회전율 3개월 (60일 평균 거래대금 / 시가총액 * 100)
        'RSI_Signal_Oscillator',  # RSI 신호 오실레이터 (RSI_14 - RSI_14.rolling(9).mean())
        'ATRr_5',  # ATR 비율 5일 (기준 - 1W)
        'ATRr_20',  # ATR 비율 20일 (기준 - 1M)
        # 'ATRr_60',  # ATR 비율 60일 (제거)
        # ATR_Ratio_Short, ATR_Ratio_Trend 제거됨 (2024년 12월)
        # 'Eff_Ratio_10'  # 효율성 비율 10일 (2024년 12월 제거)
        
        # 2024년 12월 신규 추가 피처 (3종)
        'Log_Return_20',     # 로그 수익률 1개월 (20일)
        'HV_Volatility_20',  # HV 변동성 1개월 (일별 로그 수익률의 20일 표준편차)
        'HV_Volatility_60',  # HV 변동성 3개월 (일별 로그 수익률의 60일 표준편차)
        'HV_Volatility_5',   # HV 변동성 1주 (일별 로그 수익률의 5일 표준편차)
        'VWAP_Disparity_5',  # VWAP 괴리율 1주 (5일 기준)
        # Gap 피처 제거
        # 신규 추가
        'Max_Drawdown_20',  # 최근 20일 최대 낙폭 (%)
        'CLV',  # Close Location Value (종가 위치 지수, 캔들 내 매수/매도 힘의 우위)
    ]

    try:

        # data_processor를 통해 모든 피처가 계산된 거대 데이터프레임 생성
        # 웜업 기간 포함하여 수집 (나중에 로드 시 필터링)
        # 이 단계에서 메모리 사용량이 일시적으로 크게 증가함

        # 학습용 데이터 생성 시 팩터 점수 계산 건너뛰기 (백테스팅에서만 필요)
        # 학습 데이터는 100억 미만 종목도 포함 (100억 = 10,000,000,000원)
        full_df = data_processor.get_preprocessed_data(actual_start_date, end_date, skip_factor_scores=True, min_marcap=10_000_000_000)

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

        # 결측치 처리는 나중에 Fold별로 수행하므로 여기서는 제거하지 않음
        # 날짜 컬럼이 있는지 확인 (나중에 날짜 필터링에 필요)
        if 'date' not in full_df.columns:
            log_error("❌ 심각한 오류: 'date' 컬럼이 데이터에 없습니다.")
            raise ValueError("'date' 컬럼이 데이터에 없습니다.")
        
        log_info("   ✅ 결측치는 Fold별로 처리하므로 여기서는 저장하지 않습니다.")

        os.makedirs(data_path, exist_ok=True)

        tickers = full_df['종목코드'].unique()

        log_info(f"   📂 총 {len(tickers)}개 종목의 데이터를 전처리하여 개별 파일로 저장합니다...")

        saved_count = 0
        failed_count = 0
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                ticker_df = full_df[full_df['종목코드'] == ticker].copy()

                if ticker_df.empty:
                    log_warning(f"   ⚠️ 종목 {ticker}의 데이터가 비어있습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    continue

                # 전처리: 모든 피처 저장 (학습 시 features 리스트로 필터링)
                # 숫자형 피처만 선택 (target 제외)
                numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
                if 'target' in numeric_cols:
                    numeric_cols.remove('target')
                
                if not numeric_cols:
                    log_warning(f"   ⚠️ 종목 {ticker}에 숫자형 피처가 없습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    continue
                
                # 모든 숫자형 피처 저장 (features 리스트에 없는 피처도 포함)
                X_all = ticker_df[numeric_cols].astype(np.float32)
                
                if 'target' not in ticker_df.columns:
                    log_warning(f"   ⚠️ 종목 {ticker}에 target 컬럼이 없습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    del X_all
                    continue
                
                y = ticker_df['target'].astype(np.int32)
                
                # 결측치는 나중에 Fold별로 처리하므로 여기서는 채우지 않음
                # date 컬럼도 함께 저장 (나중에 날짜 필터링에 필요)
                if 'date' not in ticker_df.columns:
                    log_warning(f"   ⚠️ 종목 {ticker}에 'date' 컬럼이 없습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    del X_all, y
                    continue
                
                # 모든 피처와 target, date, 종목코드, 시가총액을 하나의 DataFrame으로 합치기
                preprocessed_df = X_all.copy()
                preprocessed_df['target'] = y
                preprocessed_df['date'] = pd.to_datetime(ticker_df['date'])
                # 백테스팅에 필요한 메타데이터 추가
                if '종목코드' in ticker_df.columns:
                    preprocessed_df['종목코드'] = ticker_df['종목코드'].values
                else:
                    preprocessed_df['종목코드'] = ticker
                if '시가총액' in ticker_df.columns:
                    preprocessed_df['시가총액'] = ticker_df['시가총액'].values
                if '종목명' in ticker_df.columns:
                    preprocessed_df['종목명'] = ticker_df['종목명'].values

                file_path = os.path.join(data_path, f"{ticker}.feather")

                # pandas를 사용하여 feather 파일로 저장 (모든 피처 포함)
                try:
                    preprocessed_df.to_feather(file_path)
                    saved_count += 1
                except Exception as save_error:
                    log_error(f"   ❌ 종목 {ticker} feather 파일 저장 실패: {save_error}")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    # 저장 실패해도 메모리 정리는 계속 진행
                
                # 메모리 정리 (X_all로 수정)
                del ticker_df, X_all, y, preprocessed_df

                if (i + 1) % 200 == 0:
                    log_info(f"      ... {i+1}/{len(tickers)} 개 종목 저장 완료 (성공: {saved_count}, 실패: {failed_count})")
                    
            except Exception as e:
                log_error(f"   ❌ 종목 {ticker} 처리 중 오류 발생: {e}")
                log_error(f"   ❌ 오류 상세: {type(e).__name__}: {str(e)}")
                import traceback
                log_error(f"   ❌ 스택 트레이스:\n{traceback.format_exc()}")
                failed_count += 1
                failed_tickers.append(ticker)
                # 메모리 정리 시도
                try:
                    if 'ticker_df' in locals():
                        del ticker_df
                    if 'X_all' in locals():
                        del X_all
                    if 'y' in locals():
                        del y
                    if 'preprocessed_df' in locals():
                        del preprocessed_df
                except:
                    pass
                continue

        if failed_count > 0:
            log_warning(f"   ⚠️ {failed_count}개 종목 저장 실패: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        log_info(f"   ✅ 총 {saved_count}/{len(tickers)}개 종목의 전처리된 데이터 저장이 완료되었습니다.")




        

        # 거대 데이터프레임 메모리에서 명시적으로 해제

        del full_df

        gc.collect()

        log_memory_usage("개별 파일 저장 후 메모리 정리")

        

        return True



    except Exception as e:

        log_critical("데이터 준비 과정에서 심각한 오류 발생", exception=e)

        return False



# --- 모델 훈련 및 평가 ---

def get_date_range_from_files(file_paths):
    """
    모든 feather 파일에서 날짜 범위를 추출합니다.
    
    Args:
        file_paths: feather 파일 경로 리스트
    
    Returns:
        (min_date, max_date) 튜플 (pd.Timestamp 객체)
    """
    min_date = None
    max_date = None
    
    # 샘플 파일 몇 개만 로드하여 날짜 범위 확인 (성능 최적화)
    sample_size = min(10, len(file_paths))
    sample_files = file_paths[:sample_size]
    
    log_info(f"   📅 날짜 범위 확인 중... (샘플 {sample_size}개 파일 확인)")
    
    for file_path in sample_files:
        try:
            df = cudf.read_feather(file_path)
            if 'date' not in df.columns:
                log_warning(f"   ⚠️ 파일 {file_path}에 'date' 컬럼이 없습니다.")
                del df
                continue
            
            # cuDF Series를 pandas로 명시적 변환 후 날짜 처리
            date_series_pandas = df['date'].to_pandas()
            date_series_pandas = pd.to_datetime(date_series_pandas)
            file_min = pd.Timestamp(date_series_pandas.min())
            file_max = pd.Timestamp(date_series_pandas.max())
            
            if min_date is None or file_min < min_date:
                min_date = file_min
            if max_date is None or file_max > max_date:
                max_date = file_max
            
            del df
        except Exception as e:
            log_warning(f"   ⚠️ 파일 {file_path} 날짜 범위 확인 실패: {e}")
            continue
    
    if min_date is None or max_date is None:
        raise ValueError("날짜 범위를 확인할 수 없습니다. feather 파일에 'date' 컬럼이 있는지 확인하세요.")
    
    log_info(f"   ✅ 날짜 범위 확인 완료: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    
    return min_date, max_date

def get_trading_dates_from_files(file_paths, max_files=30):
    """
    샘플 feather 파일에서 거래일(date) 캘린더를 추출합니다.
    - 폴드 경계 purge(미래 10거래일 라벨 누수 방지)를 위해 사용
    - 성능을 위해 모든 파일을 읽지 않고 샘플링합니다.
    """
    if not file_paths:
        return pd.DatetimeIndex([])
    
    # 샘플링: 앞/중간/뒤에서 일부 추출 (균형)
    sample_paths = []
    head_n = min(max_files // 3, len(file_paths))
    tail_n = min(max_files // 3, len(file_paths))
    mid_n = min(max_files - head_n - tail_n, max(0, len(file_paths) - head_n - tail_n))
    
    sample_paths.extend(file_paths[:head_n])
    if mid_n > 0 and len(file_paths) > (head_n + tail_n):
        mid_start = max(0, len(file_paths) // 2 - mid_n // 2)
        sample_paths.extend(file_paths[mid_start:mid_start + mid_n])
    sample_paths.extend(file_paths[-tail_n:])
    
    # 중복 제거
    sample_paths = list(dict.fromkeys(sample_paths))
    
    all_dates = []
    for p in sample_paths:
        try:
            df = pd.read_feather(p, columns=['date'])
            if 'date' not in df.columns:
                continue
            dates = pd.to_datetime(df['date'], errors='coerce')
            dates = dates.dropna()
            if not dates.empty:
                all_dates.append(dates)
        except Exception:
            continue
    
    if not all_dates:
        return pd.DatetimeIndex([])
    
    unique_dates = pd.to_datetime(pd.concat(all_dates, ignore_index=True).unique())
    unique_dates = pd.DatetimeIndex(unique_dates).sort_values()
    return unique_dates

def get_purged_train_end_exclusive(trading_dates: pd.DatetimeIndex, val_start, purge_trading_days: int):
    """
    Val 시작일 기준으로 purge_trading_days(거래일)만큼 떨어진 Train end(exclusive)를 계산합니다.
    Train/Val을 [start, end) 형태로 사용할 때 안전한 경계값을 반환합니다.
    """
    if trading_dates is None or len(trading_dates) == 0:
        return None
    if purge_trading_days <= 0:
        return pd.Timestamp(val_start)
    
    val_start_ts = pd.Timestamp(val_start)
    idx = trading_dates.searchsorted(val_start_ts, side='left')  # val_start 이상 첫 거래일 index
    # idx가 purge_trading_days보다 작으면 충분한 과거 데이터가 없음
    if idx - purge_trading_days <= 0:
        return None
    # Train은 < boundary (exclusive). boundary일부터 val_start 전까지 purge 구간이 됨.
    return pd.Timestamp(trading_dates[idx - purge_trading_days])

def calculate_expanding_fold_ranges(file_paths, warmup_days=250, val_period_days=365, n_folds=3, purge_trading_days=10):
    """
    Expanding Window 방식으로 Fold 범위를 계산합니다.
    
    Args:
        file_paths: feather 파일 경로 리스트
        warmup_days: 웜업 기간 (일)
        val_period_days: 검증 기간 (일)
        n_folds: Fold 개수
    
    Returns:
        fold_ranges 리스트: 각 요소는 {'fold': int, 'train_start': Timestamp, 'train_end': Timestamp, 
                                      'val_start': Timestamp, 'val_end': Timestamp} 딕셔너리
    """
    # 전체 날짜 범위 확인
    min_date, max_date = get_date_range_from_files(file_paths)
    trading_dates = get_trading_dates_from_files(file_paths)
    if trading_dates is None or len(trading_dates) == 0:
        raise ValueError("거래일(date) 캘린더를 추출할 수 없습니다. feather 파일에 'date' 컬럼이 있는지 확인하세요.")
    
    # 실제 학습 시작일: 웜업 기간 제외
    actual_start_date = min_date + timedelta(days=warmup_days)
    # [중요] 날짜 필터링은 [start, end) 형태(< end)로 사용하므로 end는 exclusive 경계로 둡니다.
    # max_date(마지막 거래일)을 포함하려면 +1일을 더한 값을 end 경계로 사용해야 합니다.
    actual_end_date = max_date + timedelta(days=1)
    
    log_info(f"   📅 실제 학습 기간: {actual_start_date.strftime('%Y-%m-%d')} ~ {actual_end_date.strftime('%Y-%m-%d')}")
    
    # Expanding Window 방식으로 Fold 범위 계산
    # 검증 기간을 고정하고, Train은 누적되는 방식
    fold_ranges = []
    
    # 마지막 n_folds개 검증 기간을 역순으로 계산
    # 예: Fold 0: 마지막 3번째 검증 기간, Fold 1: 마지막 2번째 검증 기간, Fold 2: 마지막 검증 기간
    for fold_idx in range(n_folds):
        # 검증 기간 계산 (역순)
        val_end = actual_end_date - timedelta(days=val_period_days * (n_folds - 1 - fold_idx))
        val_start = val_end - timedelta(days=val_period_days)
        
        # Train 기간: actual_start_date ~ val_start (누적, 단 val_start 직전 purge 적용)
        train_start = actual_start_date
        train_end = None
        
        # 검증 기간이 실제 학습 기간을 벗어나지 않도록 조정
        if val_start < actual_start_date:
            log_warning(f"   ⚠️ Fold #{fold_idx+1} 검증 기간이 학습 시작일보다 이전입니다. 조정합니다.")
            val_start = actual_start_date
            val_end = val_start + timedelta(days=val_period_days)
        
        # [핵심] 타깃이 "향후 10거래일"을 참조하므로, 폴드 경계에서 라벨 누수를 막기 위해
        # val_start 기준 purge_trading_days(기본 10거래일)만큼 Train 끝을 앞당깁니다.
        train_end = get_purged_train_end_exclusive(trading_dates, val_start, purge_trading_days)
        if train_end is None:
            log_warning(f"   ⚠️ Fold #{fold_idx+1} purge({purge_trading_days}거래일) 적용 후 Train 기간이 유효하지 않습니다. 건너뜁니다.")
            continue
        
        if train_end <= train_start:
            log_warning(f"   ⚠️ Fold #{fold_idx+1} Train 기간이 유효하지 않습니다. 건너뜁니다.")
            continue
        
        fold_ranges.append({
            'fold': fold_idx,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end
        })
        
        log_info(f"   📊 Fold #{fold_idx+1}/{n_folds}:")
        log_info(f"      Train: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
        log_info(f"      Val: {val_start.strftime('%Y-%m-%d')} ~ {val_end.strftime('%Y-%m-%d')}")
    
    return fold_ranges

def _process_single_file(file_path, features, start_date, end_date, include_crash_pattern=False):
    """
    단일 파일을 처리하는 헬퍼 함수 (워커 스레드에서 실행)
    
    Args:
        file_path: feather 파일 경로
        features: 사용할 피처 리스트
        start_date: 시작 날짜 (pd.Timestamp)
        end_date: 종료 날짜 (pd.Timestamp)
        include_crash_pattern: True이면 급락 패턴 정보도 계산 (샘플링용)
    
    Returns:
        pandas.DataFrame 또는 None (처리 실패 시)
    """
    try:
        # CPU에서 pandas로 파일 읽기 (스레드 안전)
        df = pd.read_feather(file_path)
        
        # 날짜 컬럼 확인
        if 'date' not in df.columns:
            return None
        
        # 날짜 변환 및 필터링 (CPU에서 처리)
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] < end_date)
        df = df.loc[mask]
        
        if df.empty:
            return None
        
        # target 컬럼 확인
        if 'target' not in df.columns:
            return None
        
        # 급락 패턴 계산 (샘플링용)
        # 원본 데이터를 다시 읽어서 종가 정보로 계산
        if include_crash_pattern:
            try:
                # 종목코드 추출
                ticker = os.path.basename(file_path).replace('.feather', '')
                
                # 원본 데이터에서 종가 정보 읽기 (data_processor를 통해)
                # 간단하게: 원본 feather 파일에는 종가가 없으므로
                # data_processor.get_preprocessed_data를 다시 호출하는 것은 비효율적
                # 대신 샘플링 시점에 별도로 처리
                # 여기서는 일단 계산하지 않음
                pass
            except Exception:
                pass
        
        # Forward Fill (CPU에서 처리)
        # features 컬럼만 ffill 적용
        available_features = [f for f in features if f in df.columns]
        if available_features:
            df[available_features] = df[available_features].ffill()
        
        # 무한대(Inf) 값 처리 (스케일러 고장 방지)
        # 숫자형 컬럼만 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 필요한 컬럼만 반환 (메모리 절약)
        # 백테스팅에 필요한 메타데이터도 포함
        required_cols = ['date', 'target'] + [f for f in features if f in df.columns]
        # 백테스팅에 필요한 메타데이터 추가 (있는 경우만)
        for meta_col in ['종목코드', '시가총액', '종목명']:
            if meta_col in df.columns and meta_col not in required_cols:
                required_cols.append(meta_col)
        df = df[required_cols].copy()
        
        return df
        
    except Exception as e:
        log_warning(f"   ⚠️ 파일 {file_path} 처리 실패: {e}")
        return None

def _load_crash_pattern_data(file_paths, start_date, end_date, max_workers=4):
    """
    급락 패턴 식별을 위해 원본 데이터에서 종가 정보를 로드하는 함수
    
    Args:
        file_paths: feather 파일 경로 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
        max_workers: 병렬 워커 수
    
    Returns:
        dict: {인덱스: bool} - 급락 패턴 여부 (10일 내 -20% 이하로 떨어진 경우 True)
    """
    crash_pattern_map = {}
    
    def _load_single_crash_pattern(file_path):
        """단일 파일에서 급락 패턴 계산"""
        try:
            df = pd.read_feather(file_path)
            if 'date' not in df.columns:
                return {}
            
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] < end_date)
            df = df.loc[mask]
            
            if df.empty:
                return {}
            
            # 종가 정보가 있는지 확인 (feather 파일에는 없을 수 있음)
            # 원본 데이터를 다시 읽어야 함
            # 일단 target만 사용하고, 나중에 개선
            return {}
        except Exception:
            return {}
    
    # 병렬 처리로 급락 패턴 계산
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_load_single_crash_pattern, file_path): file_path
            for file_path in file_paths
        }
        
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                crash_pattern_map.update(result)
            except Exception:
                continue
    
    return crash_pattern_map

def _process_batch(batch_files, features, start_date, end_date, max_workers=4):
    """
    배치 단위로 파일들을 병렬 처리하는 함수
    
    Args:
        batch_files: 파일 경로 리스트 (배치)
        features: 사용할 피처 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
        max_workers: 병렬 워커 수
    
    Returns:
        pandas DataFrame 리스트 (처리된 데이터)
    """
    batch_dfs = []
    
    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 파일에 대해 작업 제출
        future_to_file = {
            executor.submit(_process_single_file, file_path, features, start_date, end_date): file_path
            for file_path in batch_files
        }
        
        # 완료된 작업 처리
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    batch_dfs.append(df)
            except Exception as e:
                log_warning(f"   ⚠️ 파일 {file_path} 처리 중 오류: {e}")
                continue
    
    return batch_dfs

def load_data_period(file_paths, features, start_date, end_date, imputation_map=None, batch_size=50, max_workers=4):
    """
    날짜 기반 데이터 로드 및 결측치 처리 (하이브리드 병렬 처리)
    
    배치 처리 + ThreadPoolExecutor를 사용하여 파일 I/O 병렬화
    
    Args:
        file_paths: feather 파일 경로 리스트
        features: 사용할 피처 리스트
        start_date: 시작 날짜 (pd.Timestamp 또는 datetime 객체)
        end_date: 종료 날짜 (pd.Timestamp 또는 datetime 객체, exclusive)
        imputation_map: Train에서 계산한 중앙값 딕셔너리 (None이면 Train 모드)
        batch_size: 배치 크기 (기본값: 50)
        max_workers: 병렬 워커 수 (기본값: 4)
    
    Returns:
        (X, y, imputation_map, meta_data) 튜플
        - Train 모드: (X_train, y_train, calculated_map, meta_data)
        - Val 모드: (X_val, y_val, None, meta_data)
        - meta_data: pandas DataFrame with 'date' and '종목코드' columns (None if not available)
    """
    if not file_paths:
        return None, None, None, None
    
    # 날짜 타입 변환
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # 배치로 나누기
    total_files = len(file_paths)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    log_info(f"   📦 데이터 로딩 시작: 총 {total_files}개 파일, {num_batches}개 배치 (배치 크기: {batch_size}, 워커: {max_workers})")
    
    all_dfs = []
    processed_count = 0
    failed_count = 0
    
    try:
        # 각 배치 처리
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_files)
            batch_files = file_paths[batch_start:batch_end]
            
            # GPU 메모리 확인 (배치 처리 전)
            if batch_idx > 0 and batch_idx % 10 == 0:
                gpu_mem = get_memory_usage(gpu=True)
                if gpu_mem > 7000:  # 8GB 중 7GB 이상 사용 시 경고
                    log_warning(f"   ⚠️ GPU 메모리 사용량 높음: {gpu_mem:.1f}MB. 메모리 정리 중...")
                    safe_gpu_memory_cleanup()
                    gc.collect()
            
            # 배치 병렬 처리 (CPU에서 pandas로 처리)
            batch_dfs = _process_batch(batch_files, features, start_date, end_date, max_workers)
            
            if not batch_dfs:
                failed_count += len(batch_files)
                continue
            
            # 배치 결과를 cuDF로 변환 (메인 스레드에서 GPU 전송)
            try:
                # pandas DataFrame들을 합치기
                batch_pandas_df = pd.concat(batch_dfs, ignore_index=True)
                del batch_dfs
                gc.collect()
                
                # cuDF로 변환 (GPU 메모리로 전송)
                batch_cudf = cudf.from_pandas(batch_pandas_df)
                del batch_pandas_df
                gc.collect()
                
                all_dfs.append(batch_cudf)
                processed_count += len(batch_files)
                
                # 진행 상황 로깅
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    progress_pct = ((batch_idx + 1) / num_batches) * 100
                    log_info(f"   📊 진행률: {batch_idx + 1}/{num_batches} 배치 완료 ({progress_pct:.1f}%) - 처리: {processed_count}개, 실패: {failed_count}개")
                
            except Exception as e:
                log_warning(f"   ⚠️ 배치 {batch_idx + 1} GPU 전송 실패: {e}")
                failed_count += len(batch_files)
                # 메모리 정리
                try:
                    del batch_dfs, batch_pandas_df, batch_cudf
                except:
                    pass
                safe_gpu_memory_cleanup()
                gc.collect()
                continue
        
        if not all_dfs:
            log_warning("   ⚠️ 처리된 데이터가 없습니다.")
            return None, None, None, None
        
        if failed_count > 0:
            log_warning(f"   ⚠️ {failed_count}개 파일 처리 실패 (총 {total_files}개 중)")
        
        log_info(f"   ✅ 파일 로딩 완료: {processed_count}개 성공, {failed_count}개 실패")
        
        # 모든 배치 데이터 병합
        log_info("   🔄 데이터 병합 중...")
        final_df = cudf.concat(all_dfs, ignore_index=True)
        del all_dfs
        safe_gpu_memory_cleanup()
        gc.collect()
        
        log_info(f"   ✅ 병합 완료: {len(final_df):,}행")
        
        # 무한대(Inf) 값 처리 안전장치 (스케일러 고장 방지)
        # 숫자형 컬럼만 처리
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count_before = (final_df[numeric_cols] == np.inf).sum().sum() + (final_df[numeric_cols] == -np.inf).sum().sum()
            if inf_count_before > 0:
                log_warning(f"   ⚠️ 무한대 값 {inf_count_before:,}개 발견. NaN으로 변환합니다.")
                final_df[numeric_cols] = final_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # features와 target 추출
        missing_features = [f for f in features if f not in final_df.columns]
        if missing_features:
            error_msg = f"❌ 심각한 오류: load_data_period에서 필요한 피처가 데이터에 없습니다: {missing_features}"
            log_critical(error_msg)
            log_critical(f"   사용 가능한 컬럼: {list(final_df.columns)[:30]}")
            log_critical(f"   기대하는 features ({len(features)}개): {features}")
            log_critical(f"   누락된 피처 ({len(missing_features)}개): {missing_features}")
            raise ValueError(error_msg)
        
        if 'target' not in final_df.columns:
            error_msg = "❌ 심각한 오류: load_data_period에서 'target' 컬럼이 데이터에 없습니다."
            log_critical(error_msg)
            log_critical(f"   사용 가능한 컬럼: {list(final_df.columns)[:30]}")
            raise ValueError(error_msg)
        
        X = final_df[features]
        y = final_df['target']
        
        # Train-Only Imputation
        if imputation_map is None:
            # [Train 모드] 중앙값 계산
            log_info("   📊 Train 모드: 결측치 대체값 계산 중...")
            numeric_features_df = X.select_dtypes(include=[np.number])
            calculated_map = numeric_features_df.median().to_dict()
            
            # Train 데이터에 적용
            X = X.fillna(calculated_map)
            
            del final_df, numeric_features_df
            safe_gpu_memory_cleanup()
            gc.collect()
            
            log_info(f"   ✅ Train 데이터 처리 완료: {len(X):,}행")
            return X, y, calculated_map
        else:
            # [Val 모드] 외부에서 받은 맵 적용
            log_info("   📊 Val 모드: 결측치 대체값 적용 중...")
            X = X.fillna(imputation_map)
            
            del final_df
            safe_gpu_memory_cleanup()
            gc.collect()
            
            log_info(f"   ✅ Val 데이터 처리 완료: {len(X):,}행")
            return X, y, None
    
    except Exception as e:
        log_error(f"데이터 로딩 중 오류 발생: {e}")
        import traceback
        log_error(f"   상세 오류:\n{traceback.format_exc()}")
        # 메모리 정리
        try:
            del all_dfs, final_df
        except:
            pass
        safe_gpu_memory_cleanup()
        gc.collect()
        return None, None, None

def objective(trial, fold_data_cache, features, max_depth_list, rng):
    """
    Optuna를 위한 objective 함수 (단일 모델 학습 방식).
    fold_data_cache: 모든 Fold의 데이터 캐시 (이미 결측치 처리 완료).
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
        'max_features': trial.suggest_categorical('max_features', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
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
            enhanced_gpu_memory_cleanup(force_defrag=True)
            gc.collect()

        fold_data = fold_data_cache[fold]
        if len(fold_data) >= 4:
            X_train_all, y_train_all, X_val, y_val = fold_data[:4]
        else:
            log_warning(f"   ⚠️ Fold #{fold+1} 데이터 형식이 올바르지 않습니다. 건너뜁니다.")
            continue
        
        if X_train_all is None or X_val is None:
            log_warning(f"   ⚠️ Fold #{fold+1} 데이터가 없습니다. 건너뜁니다.")
            continue

        # Trial 전에 이미 언더샘플링이 적용되었으므로, 여기서는 샘플링을 건너뜁니다.
        # (중복 샘플링 방지: Trial 전 언더샘플링으로 이미 클래스 균형이 맞춰져 있음)
        X_train_resampled = X_train_all
        y_train_resampled = y_train_all
        
        # [중요] is_crash 컬럼이 X_train에 포함되어 있으면 제거 (정답 누설 방지)
        if 'is_crash' in X_train_resampled.columns:
            X_train_resampled = X_train_resampled.drop(columns=['is_crash'])
            log_info(f"   ✅ is_crash 컬럼 제거 완료 (정답 누설 방지)")
        if 'is_crash' in X_val.columns:
            X_val = X_val.drop(columns=['is_crash'])
        
        # 스케일링 전 데이터 검증 및 정리
        # NaN/Inf 값 확인 및 처리
        numeric_cols_train = X_train_resampled.select_dtypes(include=[np.number]).columns
        numeric_cols_val = X_val.select_dtypes(include=[np.number]).columns
        
        median_values = None  # Val 데이터 NaN 대체용
        
        if len(numeric_cols_train) > 0:
            # 무한대 값 처리
            inf_count_train = (X_train_resampled[numeric_cols_train] == np.inf).sum().sum() + (X_train_resampled[numeric_cols_train] == -np.inf).sum().sum()
            if inf_count_train > 0:
                log_warning(f"   ⚠️ Fold #{fold+1} Train 데이터에 무한대 값 {inf_count_train:,}개 발견. NaN으로 변환합니다.")
                X_train_resampled[numeric_cols_train] = X_train_resampled[numeric_cols_train].replace([np.inf, -np.inf], np.nan)
            
            # NaN 개수 확인 및 중앙값 계산
            nan_count_train = X_train_resampled[numeric_cols_train].isna().sum().sum()
            if nan_count_train > 0:
                log_warning(f"   ⚠️ Fold #{fold+1} Train 데이터에 NaN {nan_count_train:,}개 발견. 중앙값으로 대체합니다.")
                # 중앙값 계산 (Val 데이터 대체용으로도 사용)
                median_values = X_train_resampled[numeric_cols_train].median()
                X_train_resampled[numeric_cols_train] = X_train_resampled[numeric_cols_train].fillna(median_values)
        
        if len(numeric_cols_val) > 0:
            # 검증 데이터도 동일하게 처리
            inf_count_val = (X_val[numeric_cols_val] == np.inf).sum().sum() + (X_val[numeric_cols_val] == -np.inf).sum().sum()
            if inf_count_val > 0:
                log_warning(f"   ⚠️ Fold #{fold+1} Val 데이터에 무한대 값 {inf_count_val:,}개 발견. NaN으로 변환합니다.")
                X_val[numeric_cols_val] = X_val[numeric_cols_val].replace([np.inf, -np.inf], np.nan)
            
            # 검증 데이터는 Train의 중앙값으로 대체 (데이터 누출 방지)
            nan_count_val = X_val[numeric_cols_val].isna().sum().sum()
            if nan_count_val > 0:
                if median_values is not None:
                    log_warning(f"   ⚠️ Fold #{fold+1} Val 데이터에 NaN {nan_count_val:,}개 발견. Train 중앙값으로 대체합니다.")
                    X_val[numeric_cols_val] = X_val[numeric_cols_val].fillna(median_values[numeric_cols_val])
                else:
                    log_warning(f"   ⚠️ Fold #{fold+1} Val 데이터에 NaN {nan_count_val:,}개 발견. Val 중앙값으로 대체합니다.")
                    val_median = X_val[numeric_cols_val].median()
                    X_val[numeric_cols_val] = X_val[numeric_cols_val].fillna(val_median)
        
        # 스케일러 학습 및 데이터 변환 (샘플링된 데이터로 학습)
        step_start = datetime.now()
        try:
            scaler = cuStandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_val_scaled = scaler.transform(X_val)
        except Exception as scale_e:
            log_error(f"   ❌ Fold #{fold+1} 스케일링 실패: {scale_e}")
            import traceback
            log_error(f"   상세 오류:\n{traceback.format_exc()}")
            continue  # 다음 fold로 진행
        
        preprocessing_time = (datetime.now() - step_start).total_seconds()
        
        del X_train_all, X_val, X_train_resampled
        enhanced_gpu_memory_cleanup(force_defrag=False)
        gc.collect()

        # 단일 모델 학습
        step_start = datetime.now()
        model = cuRF(**params)
        try:
            model.fit(X_train_scaled, y_train_resampled)
        except Exception as fit_e:
            log_error(f"   ❌ Fold #{fold+1} 모델 학습 실패: {fit_e}")
            import traceback
            log_error(f"   상세 오류:\n{traceback.format_exc()}")
            continue  # 다음 fold로 진행
        
        fit_time = (datetime.now() - step_start).total_seconds()
        
        del X_train_scaled, y_train_resampled
        enhanced_gpu_memory_cleanup(force_defrag=False)
        gc.collect()
        
        # 예측 (VRAM 파편화 방지를 위한 배치 처리)
        step_start = datetime.now()
        batch_info_str = ""  # 배치 정보 문자열 초기화
        try:
            # 예측 전 파편화 확인 및 정리
            is_fragmented, frag_info = check_gpu_memory_fragmentation()
            if is_fragmented:
                log_warning(f"   ⚠️ Fold #{fold+1} 예측 전 파편화 감지: {frag_info.get('reason', '알 수 없음')}")
                enhanced_gpu_memory_cleanup(force_defrag=True)
            
            # 모델 복잡도 기반 동적 배치 크기 결정
            val_size = len(X_val_scaled)
            batch_size = calculate_dynamic_batch_size(params, val_size)
            
            if batch_size is not None:
                # 배치 처리 실행
                num_batches = (val_size + batch_size - 1) // batch_size
                batch_info_str = f" (배치: {num_batches}개, 크기: {batch_size:,})"
                
                pred_proba_list = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, val_size)
                    X_batch = X_val_scaled.iloc[start_idx:end_idx]
                    
                    try:
                        batch_proba = model.predict_proba(X_batch)
                        pred_proba_list.append(batch_proba.iloc[:, 1])
                        del X_batch, batch_proba
                        # 배치 간 메모리 정리
                        if batch_idx < num_batches - 1:  # 마지막 배치가 아니면 정리
                            enhanced_gpu_memory_cleanup(force_defrag=False)
                    except Exception as e:
                        log_error(f"   ❌ Fold #{fold+1} 배치 {batch_idx+1}/{num_batches} 예측 실패: {e}")
                        # 실패한 배치는 중립값(0.5)으로 채움
                        neutral_proba = cudf.Series([0.5] * (end_idx - start_idx))
                        pred_proba_list.append(neutral_proba)
                        del X_batch
                        enhanced_gpu_memory_cleanup(force_defrag=False)
                
                # 배치 결과 합치기
                y_pred_proba = cudf.concat(pred_proba_list, ignore_index=True)
                del pred_proba_list
            else:
                # 배치 처리 불필요 (한 번에 처리)
                pred_proba_cudf = model.predict_proba(X_val_scaled)
                y_pred_proba = pred_proba_cudf.iloc[:, 1]
                del pred_proba_cudf
            
            pred_time = (datetime.now() - step_start).total_seconds()
        except Exception as e:
            log_error(f"   ❌ Fold #{fold+1} 예측 실패: {e}")
            del model, X_val_scaled, y_val
            enhanced_gpu_memory_cleanup(force_defrag=True)
            continue # 다음 fold로 진행

        # ROC-AUC 점수 계산 (명시적 타입 변환)
        try:
            # cuDF Series를 numpy 배열로 명시적 변환
            if hasattr(y_val, 'to_numpy'):
                y_true_np = y_val.to_numpy()
            elif hasattr(y_val, 'to_pandas'):
                y_true_np = y_val.to_pandas().values
            else:
                y_true_np = np.array(y_val)
            
            if hasattr(y_pred_proba, 'to_numpy'):
                y_pred_np = y_pred_proba.to_numpy()
            elif hasattr(y_pred_proba, 'to_pandas'):
                y_pred_np = y_pred_proba.to_pandas().values
            else:
                y_pred_np = np.array(y_pred_proba)
            
            # cuML의 roc_auc_score 사용 (GPU 가속)
            score = roc_auc_score(y_true_np, y_pred_np)
            
        except Exception as score_e:
            log_error(f"   ❌ ROC-AUC 점수 계산 실패: {score_e}")
            score = 0.0  # 실패 시 기본값
        
        fold_scores.append(score)
        
        # 사용 완료된 객체 정리
        # 변수 정리 (배치 처리 경로에서는 pred_proba_cudf가 없을 수 있음)
        del scaler, model, X_val_scaled, y_val, y_pred_proba
        try:
            del pred_proba_cudf
        except NameError:
            # 배치 처리 경로에서는 pred_proba_cudf가 생성되지 않음
            pass
        enhanced_gpu_memory_cleanup(force_defrag=False)
        gc.collect()

        fold_duration = (datetime.now() - fold_start_time).total_seconds()
        log_info(f"   ✅ Fold #{fold+1}/3 | Score: {score:.4f} | 총: {fold_duration:.1f}초 "
                 f"(전처리: {preprocessing_time:.1f}초, 학습: {fit_time:.1f}초, 예측: {pred_time:.1f}초){batch_info_str}")

        # Pruning 체크
        trial.report(score, step=fold)
        if trial.should_prune():
            trial_duration = (datetime.now() - trial_start_time).total_seconds()
            log_info(f"   ⏹️ Trial #{trial.number} 조기 종료 (Pruning) | 총 소요시간: {trial_duration/60:.1f}분 ({trial_duration:.1f}초)")
            log_info(f"{'='*60}")
            enhanced_gpu_memory_cleanup(force_defrag=True)
            gc.collect()
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

def train_final_ensemble_model(fold_data_cache, features, best_params, rng, optimization_results=None, training_config=None, data_path=None):
    """
    전체 데이터를 사용하여 최종 단일 모델을 훈련하고 저장합니다.
    (메모리 최적화 버전: 모델을 순차적으로 학습/예측하여 GPU 메모리 사용량 최소화)
    결측치 처리는 fold_cache에서 로드한 데이터가 이미 처리된 상태입니다.
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
        
        # 저장할 때 6개를 저장했으므로 (imputation_map, meta_data 포함), 6개를 받아야 함
        if len(fold_data) == 6:
            X_train_original, y_train_original, X_val_original, y_val_original, last_fold_imputer, train_meta_data_final = fold_data
        elif len(fold_data) == 5:
            # 이전 버전 호환성
            X_train_original, y_train_original, X_val_original, y_val_original, last_fold_imputer = fold_data
            train_meta_data_final = None
        else:
            raise ValueError(f"fold_cache 파일 형식이 맞지 않습니다. (요소 개수: {len(fold_data)}, 기대값: 5 또는 6)")
        
        # pandas DataFrame을 cuDF DataFrame으로 변환 (저장 시 pandas로 변환했으므로)
        if isinstance(X_train_original, pd.DataFrame):
            X_train_original = cudf.from_pandas(X_train_original)
        if isinstance(X_val_original, pd.DataFrame):
            X_val_original = cudf.from_pandas(X_val_original)
        if isinstance(y_train_original, (pd.Series, np.ndarray)):
            y_train_original = cudf.Series(y_train_original) if not isinstance(y_train_original, cudf.Series) else y_train_original
        if isinstance(y_val_original, (pd.Series, np.ndarray)):
            y_val_original = cudf.Series(y_val_original) if not isinstance(y_val_original, cudf.Series) else y_val_original
        
        if X_train_original is None or X_val_original is None:
            raise ValueError("fold_cache 파일의 데이터가 유효하지 않습니다.")
        
        log_info(f"   [OK] 원본 데이터 로드 완료: Train {len(X_train_original):,}행, Val {len(X_val_original):,}행")
        
        # 원본 train + 원본 val 합치기 (전체 데이터 복원)
        X_all = cudf.concat([X_train_original, X_val_original], ignore_index=True)
        y_all = cudf.concat([y_train_original, y_val_original], ignore_index=True)
        
        # features 리스트에 있는 피처만 선택 (feather 파일에는 모든 피처가 저장되어 있음)
        missing_features = [f for f in features if f not in X_all.columns]
        if missing_features:
            error_msg = f"❌ 심각한 오류: features 리스트에 있는 피처가 데이터에 없습니다: {missing_features}"
            log_critical(error_msg)
            log_critical(f"   사용 가능한 컬럼: {list(X_all.columns)[:20]}...")
            log_critical(f"   기대하는 features: {features}")
            raise ValueError(error_msg)
        
        # features 리스트에 있는 피처만 선택
        X_all = X_all[features]
        
        log_info(f"   ✅ 데이터 피처 선택 완료: {len(features)}개 피처 사용")
        
        # [중요] 최종 모델을 위한 Imputation Value 재계산
        # 이유: X_all(전체 데이터) 기준으로 최신 중앙값을 다시 계산해서 저장해야,
        # 나중에 내일 주가를 예측할 때 그 기준으로 결측치를 채울 수 있음.
        log_info("   📊 최종 모델용 결측치 대체값(중앙값) 계산 중...")
        numeric_features_df = X_all.select_dtypes(include=[np.number])
        final_imputation_values = numeric_features_df.median().to_dict()
        log_info(f"   ✅ 최종 모델용 결측치 대체값 계산 완료 ({len(final_imputation_values)}개 피처)")
        
        # 원본 데이터 메모리 해제
        del X_train_original, y_train_original, X_val_original, y_val_original, fold_data, last_fold_imputer, numeric_features_df
        enhanced_gpu_memory_cleanup(force_defrag=False)
        gc.collect()
        
        # 원본 데이터 클래스 분포 확인
        y_all_pandas = y_all.to_pandas()
        value_counts_original = y_all_pandas.value_counts()
        minority_class_label = value_counts_original.idxmin()
        majority_class_label = value_counts_original.idxmax()
        n_minority_original = value_counts_original[minority_class_label]
        n_majority_original = value_counts_original[majority_class_label]
        minority_class_name = "상승" if minority_class_label == 1 else "급락"
        majority_class_name = "상승" if majority_class_label == 1 else "급락"
        
        log_info(f"   [OK] 전체 데이터 구성 완료: {len(X_all):,}행 ({(datetime.now() - compose_start).total_seconds():.1f}초)")
        log_info(f"   [DATA] 원본 데이터 클래스 분포:")
        log_info(f"      - 소수 클래스 ({minority_class_label}, {minority_class_name}): {n_minority_original:,}개")
        log_info(f"      - 다수 클래스 ({majority_class_label}, {majority_class_name}): {n_majority_original:,}개")
        
        del y_all_pandas

        # 2. 전체 데이터 언더샘플링 적용 (1:1 비율, Trial 전 언더샘플링과 동일한 로직)
        log_info("   ⚖️ 최종 학습 전 언더샘플링 적용 (1:1 비율, 급락 패턴 우선 선택)...")
        sampling_start = datetime.now()
        
        # 클래스 분포 확인
        y_all_pandas = y_all.to_pandas()
        value_counts = y_all_pandas.value_counts()
        
        if len(value_counts) >= 2:
            # 소수 클래스와 다수 클래스 식별
            minority_class_label_check = value_counts.idxmin()
            majority_class_label_check = value_counts.idxmax()
            n_minority_check = value_counts[minority_class_label_check]
            n_majority_check = value_counts[majority_class_label_check]
            
            # 언더샘플링: 다수 클래스를 소수 클래스 크기만큼 랜덤 선택 (1:1 비율)
            # [중요] 소수 클래스는 전체 사용 (소실 방지)
            if n_majority_check > n_minority_check:
                # 위치 인덱스 생성 (0부터 시작)
                all_indices = np.arange(len(y_all))
                y_all_values = y_all_pandas.values
                
                # 소수 클래스 인덱스 (전체 사용)
                minority_indices = all_indices[y_all_values == minority_class_label_check]
                
                # 다수 클래스 인덱스 (샘플링 대상)
                majority_indices = all_indices[y_all_values == majority_class_label_check]
                
                log_info(f"      🔀 다수 클래스 랜덤 셔플 및 샘플링 중 (다양성 확보)...")
                
                # [다양성 확보] 다수 클래스 셔플
                majority_indices_shuffled = majority_indices.copy()
                rng.shuffle(majority_indices_shuffled)
                
                # 1:1 비율로 샘플링
                target_majority_size = n_minority_check * 1
                selected_majority_indices = majority_indices_shuffled[:target_majority_size]
                
                # 인덱스 결합
                balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
                
                # [시계열 정합성] 시간 순서 유지 (과거 -> 미래)
                balanced_indices.sort()
                
                # 언더샘플링된 데이터 생성
                X_all_resampled = X_all.iloc[balanced_indices].reset_index(drop=True)
                y_all_resampled = y_all.iloc[balanced_indices].reset_index(drop=True)
                
                # 샘플링 결과 확인
                y_all_resampled_pandas = y_all_resampled.to_pandas()
                value_counts_resampled = y_all_resampled_pandas.value_counts()
                n_minority_resampled = value_counts_resampled[minority_class_label]
                n_majority_resampled = value_counts_resampled[majority_class_label]
                del y_all_resampled_pandas
                
                # 원본 데이터 삭제
                del X_all, y_all, balanced_indices
                enhanced_gpu_memory_cleanup(force_defrag=True)
                gc.collect()
                
                # 샘플링된 데이터로 교체
                X_all = X_all_resampled
                y_all = y_all_resampled
                del X_all_resampled, y_all_resampled
                
                log_info(f"   [OK] 언더샘플링 완료 (1:1 비율):")
                log_info(f"      - 소수 클래스 ({minority_class_label_check}): {n_minority_original:,}개 → {n_minority_resampled:,}개 (100% 사용)")
                log_info(f"      - 다수 클래스 ({majority_class_label_check}): {n_majority_original:,}개 → {n_majority_resampled:,}개")
                log_info(f"   [DATA] 샘플링 후 데이터: {len(X_all):,}행 ({(datetime.now() - sampling_start).total_seconds():.1f}초)")
            else:
                log_info(f"   ℹ️ 클래스 불균형이 없어 샘플링을 건너뜁니다.")
                del y_all_pandas
        else:
            log_warning(f"   ⚠️ 클래스가 1개만 있어 샘플링을 건너뜁니다.")
            del y_all_pandas
        
        # 3. 전체 데이터 전처리
        step_start = datetime.now()
        
        # [삭제] is_crash 컬럼 제거 (이제 더 이상 사용하지 않음)
        # if 'is_crash' in X_all.columns:
        #    X_all = X_all.drop(columns=['is_crash'])
        
        log_info("   [PREPROC] 데이터 스케일링 중...")
        final_scaler = cuStandardScaler()
        final_scaler.fit(X_all)
        X_all_scaled = final_scaler.transform(X_all)
        del X_all
        enhanced_gpu_memory_cleanup(force_defrag=True)
        gc.collect()
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
        permutation_importances = None
        X_sample_cudf = None
        y_sample = None
        actual_features = None  # 실제 사용된 피처 이름 (SHAP/순열 중요도 계산용)
        
        if SHAP_AVAILABLE:
            log_info("   [SHAP] 피처 중요도 계산 중...")
            shap_start = datetime.now()
            try:
                # 샘플링된 데이터의 일부를 사용
                sample_size = min(1000, max(100, len(X_all_scaled) // 10))
                sample_indices = cp.random.choice(len(X_all_scaled), sample_size, replace=False)
                sample_indices_np = sample_indices.get()
                X_sample_cudf = X_all_scaled.iloc[sample_indices_np]
                y_sample = y_all.iloc[sample_indices_np]
                
                # features 리스트를 직접 사용 (스케일링 후 컬럼 이름이 유지되지 않을 수 있음)
                # X_all_scaled는 features 순서대로 스케일링되었으므로 features 리스트를 사용
                actual_features = features
                
                # SHAP는 numpy 배열을 선호
                X_sample_np = X_sample_cudf.to_numpy()

                def model_predict_proba_wrapper(X_np):
                    # features 리스트를 사용하여 DataFrame 생성 (순서 중요)
                    X_cudf = cudf.DataFrame(X_np, columns=actual_features)
                    try:
                        probas = final_model.predict_proba(X_cudf)
                        return probas.iloc[:, 1].to_numpy()
                    finally:
                        del X_cudf

                explainer = shap.KernelExplainer(model_predict_proba_wrapper, shap.sample(X_sample_np, 50))
                shap_values = explainer.shap_values(X_sample_np, nsamples=100)
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                # features 리스트와 SHAP 값 매칭 (피처 이름 보장)
                if len(actual_features) == len(mean_abs_shap):
                    feature_importances = sorted(zip(actual_features, mean_abs_shap), key=lambda x: x[1], reverse=True)
                else:
                    log_warning(f"   [WARN] 피처 수 불일치: features={len(actual_features)}, SHAP={len(mean_abs_shap)}")
                    feature_importances = None
                
                log_info(f"   [OK] SHAP 계산 완료 ({(datetime.now() - shap_start).total_seconds():.1f}초)")
                del X_sample_np, explainer, shap_values, mean_abs_shap
                
            except Exception as e:
                log_warning(f"   [WARN] SHAP 피처 중요도 계산 실패: {e}")
        else:
            log_warning("   [WARN] SHAP 라이브러리가 없어 피처 중요도를 계산할 수 없습니다.")
        
        # 6-1. 순열 중요도 계산 (SHAP 계산 후, 같은 샘플 데이터 사용)
        if X_sample_cudf is not None and y_sample is not None:
            # features 리스트를 직접 사용 (스케일링 후 컬럼 이름이 유지되지 않을 수 있음)
            if actual_features is None:
                actual_features = features
            
            log_info("   [PERM] 순열 중요도 계산 중...")
            perm_start = datetime.now()
            try:
                # cuML 모델은 sklearn API와 호환되지만, permutation_importance는 numpy 배열을 기대함
                # 샘플 데이터를 numpy 배열로 명시적 변환 (cuDF의 암묵적 변환 방지)
                X_sample_np = X_sample_cudf.to_pandas().values
                y_sample_np = np.array(y_sample.to_pandas()) if hasattr(y_sample, 'to_pandas') else np.array(y_sample.values)
                
                # cuML 모델의 classes_ 속성이 cuDF Series일 수 있으므로, 모델 래퍼 생성
                # permutation_importance가 classes_에 접근할 때 numpy 배열로 변환되도록 함
                class CuMLModelWrapper:
                    """cuML 모델을 sklearn 호환 래퍼로 감싸서 classes_ 속성 문제 해결"""
                    # sklearn이 classifier로 인식하도록 설정
                    _estimator_type = 'classifier'
                    
                    def __init__(self, cuml_model):
                        self.cuml_model = cuml_model
                        # classes_를 numpy 배열로 변환하여 저장
                        if hasattr(cuml_model, 'classes_'):
                            if hasattr(cuml_model.classes_, 'to_numpy'):
                                self.classes_ = cuml_model.classes_.to_numpy()
                            elif hasattr(cuml_model.classes_, 'to_pandas'):
                                self.classes_ = cuml_model.classes_.to_pandas().values
                            else:
                                self.classes_ = np.array(cuml_model.classes_)
                        else:
                            # classes_가 없으면 y에서 추론
                            self.classes_ = np.array([0, 1])
                    
                    def fit(self, X, y):
                        """sklearn 검증을 통과하기 위한 fit 메서드 (실제로는 호출되지 않음)"""
                        # permutation_importance는 이미 학습된 모델을 사용하므로 fit은 필요 없음
                        # 하지만 sklearn 검증을 통과하기 위해 필요
                        return self
                    
                    def predict_proba(self, X):
                        # X가 numpy 배열이면 cuDF DataFrame으로 변환
                        if isinstance(X, np.ndarray):
                            # actual_features 사용 (클로저에서 가져옴)
                            X_cudf = cudf.DataFrame(X, columns=actual_features)
                            probas = self.cuml_model.predict_proba(X_cudf)
                            # numpy 배열로 반환
                            return probas.to_pandas().values
                        else:
                            # 이미 cuDF나 pandas인 경우
                            if isinstance(X, pd.DataFrame):
                                X_cudf = cudf.DataFrame(X)
                            else:
                                X_cudf = X
                            probas = self.cuml_model.predict_proba(X_cudf)
                            return probas.to_pandas().values
                    
                    def predict(self, X):
                        # X가 numpy 배열이면 cuDF DataFrame으로 변환
                        if isinstance(X, np.ndarray):
                            # actual_features 사용 (클로저에서 가져옴)
                            X_cudf = cudf.DataFrame(X, columns=actual_features)
                            preds = self.cuml_model.predict(X_cudf)
                            # numpy 배열로 반환
                            return preds.to_pandas().values if hasattr(preds, 'to_pandas') else np.array(preds)
                        else:
                            # 이미 cuDF나 pandas인 경우
                            if isinstance(X, pd.DataFrame):
                                X_cudf = cudf.DataFrame(X)
                            else:
                                X_cudf = X
                            preds = self.cuml_model.predict(X_cudf)
                            return preds.to_pandas().values if hasattr(preds, 'to_pandas') else np.array(preds)
                
                # 모델 래퍼 생성
                wrapped_model = CuMLModelWrapper(final_model)
                
                # 순열 중요도 계산 (numpy 배열 사용)
                # _estimator_type='classifier'로 설정했으므로 자동으로 predict_proba 사용
                perm_result = permutation_importance(
                    wrapped_model,
                    X_sample_np,
                    y_sample_np,
                    n_repeats=10,
                    random_state=42,
                    scoring='roc_auc',
                    n_jobs=1  # cuML 모델은 멀티프로세싱 지원 안 함
                )
                
                # 평균 중요도 추출 및 정렬
                mean_perm_importance = perm_result.importances_mean
                # features 리스트와 순열 중요도 매칭 (피처 이름 보장)
                if len(actual_features) == len(mean_perm_importance):
                    permutation_importances = sorted(zip(actual_features, mean_perm_importance), key=lambda x: x[1], reverse=True)
                else:
                    log_warning(f"   [WARN] 피처 수 불일치: features={len(actual_features)}, 순열 중요도={len(mean_perm_importance)}")
                    permutation_importances = None
                
                log_info(f"   [OK] 순열 중요도 계산 완료 ({(datetime.now() - perm_start).total_seconds():.1f}초)")
                del X_sample_np, y_sample_np, wrapped_model, perm_result, mean_perm_importance
                
            except Exception as e:
                log_warning(f"   [WARN] 순열 중요도 계산 실패: {e}")
                import traceback
                log_warning(f"   [WARN] 순열 중요도 계산 실패 상세: {traceback.format_exc()}")
            finally:
                # 샘플 데이터 정리
                if X_sample_cudf is not None:
                    del X_sample_cudf
                if y_sample is not None:
                    del y_sample
                safe_gpu_memory_cleanup()
                gc.collect()

        # 7. 최종 모델 저장
        if final_model:
            log_info("   [SAVE] 최종 단일 모델 및 전처리기 저장 중...")
            model_path = path_manager.data_dir / 'cuml_ensemble_model.joblib'
            metadata_path = path_manager.data_dir / 'cuml_ensemble_model_metadata.joblib'
            
            # 모델 피처 수 검증 (저장 전 확인)
            model_expected_features = None
            try:
                if hasattr(final_model, 'n_features_in_'):
                    model_expected_features = final_model.n_features_in_
                elif hasattr(final_model, 'n_features_'):
                    model_expected_features = final_model.n_features_
            except Exception:
                pass
            
            if model_expected_features is not None and model_expected_features != len(features):
                error_msg = f"❌ 심각한 오류: 모델 내부는 {model_expected_features}개 피처를 기대하지만, 저장하려는 features 리스트는 {len(features)}개입니다. 모델 저장을 중단합니다."
                log_critical(error_msg)
                log_critical(f"   저장하려는 features: {features}")
                raise ValueError(error_msg)
            
            log_info(f"   ✅ 모델 피처 수 검증 완료: 모델 내부 {model_expected_features}개, features 리스트 {len(features)}개 (일치)")
            
            # 최종 training_config 구성 (n_final_models 추가)
            final_training_config = {**(training_config or {}), 'n_final_models': 1}
            
            # 모델 파일 저장
            # 디렉토리 생성 (파일 저장 전)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            # final_imputation_values: 전체 데이터(X_all) 기준으로 계산된 중앙값 (실전 추론 시 사용)
            joblib.dump({
                'model': final_model,
                'features': features,
                'scaler': final_scaler,
                'imputation_values': final_imputation_values,  # 전체 데이터 기준 중앙값 (실전 추론용)
                'best_params': best_params,
                'model_type': 'single_model',
                'optimization_results': optimization_results or {},
                'training_config': final_training_config,
                'feature_importances': feature_importances,  # SHAP 값
                'permutation_importances': permutation_importances  # 순열 중요도
            }, str(model_path), compress=3)
            log_info(f"   [OK] 최종 모델 저장 완료: {model_path}")
            
            # 메타데이터 파일 저장 (웹페이지에서 빠른 로드를 위해)
            try:
                # 디렉토리 생성 (파일 저장 전)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
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
                    'feature_importances': feature_importances,  # SHAP 값
                    'permutation_importances': permutation_importances,  # 순열 중요도
                    'parameter_explanations': parameter_explanations
                }
                
                joblib.dump(metadata_to_save, str(metadata_path), compress=3)
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
        enhanced_gpu_memory_cleanup(force_defrag=True)
        gc.collect()



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
        # 학습 데이터는 오늘에서 3개월 전까지만 수집 (백테스팅에는 영향 없음)
        end_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        log_info(f"   데이터 수집 기간: {start_date} ~ {end_date} ({years}년, 최근 3개월 제외)")
        
        # 데이터 재생성 시 기존 캐시 파일 모두 삭제 (날짜 기반 분할로 변경되므로)
        imputation_values_dir = os.path.dirname(os.path.expanduser(data_path))
        imputation_values_path = os.path.join(imputation_values_dir, "imputation_values.joblib")
        fold_cache_dir = os.path.join(imputation_values_dir, "fold_cache")
        
        # imputation_values 파일 삭제
        if os.path.exists(imputation_values_path):
            try:
                os.remove(imputation_values_path)
                log_info(f"   🗑️ 기존 imputation_values 파일 삭제 완료 (데이터 재생성으로 인해)")
            except Exception as e:
                log_warning(f"   ⚠️ 기존 imputation_values 파일 삭제 실패: {e}")
        
        # fold_cache 디렉토리 전체 삭제 (날짜 기반 분할로 변경되므로)
        if os.path.exists(fold_cache_dir):
            try:
                shutil.rmtree(fold_cache_dir)
                log_info(f"   🗑️ 기존 fold_cache 디렉토리 삭제 완료 (날짜 기반 분할로 재생성)")
            except Exception as e:
                log_warning(f"   ⚠️ 기존 fold_cache 디렉토리 삭제 실패: {e}")
        
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

    # 기존 CPU 버전과 동일하게, 검증된 핵심 피처 목록을 공용 설정에서 로드
    # 제거된 피처: PBR, USDKRW_pct_1d, KOSPI_pct_1d, 이익수익률, 수익률(3M), 수익률(1M), ATRr_14
    # 제거된 피처: KOSPI_disparity_240, USDKRW_pct_5d, VIX_pct_1d, VIX_pct_5d, KOSPI_pct_5d
    # 추가된 피처: KOSPI_disparity_60
    # 추가된 피처: disparity_60
    # 제거된 피처: BBW_20_2, 거래량 변동성 계수, 변동성 기울기, 거래량 변동성 계수 기울기
    # 제거된 피처: 등락율, 주가_기울기(1W/1M/3M), disparity_5/10, 변동성(5M), KOSPI_변동성(3D/5M), KOSPI_disparity_5
    # 제거된 피처: 변동성(3D/3M/1M/1W), KOSPI_변동성(3M/1W), KOSPI_disparity_120/20, BPS, 거래대금_MA20/MA5, BB_Position
    # 제거된 피처: disparity_120, disparity_240
    # 제거된 피처: 변동성(1W), 변동성(3M) (2024년 12월)
    features = [
        'log_mktcap',
        '52주_신고가_비율',
        'ADX_14',
        'disparity_120',  # 120일 이격도
        'disparity_240',  # 240일 이격도
        'disparity_20',   # 20일 이격도
        'KOSPI_disparity_20',  # KOSPI 20일 이격도
        # 추가된 피처
        'Trend_Pullback_Score',
        'Position_Range_60',
        # 'KOSPI_변동성(1M)',  # 2024년 12월 제거
        # 변동성(1W), 변동성(3M) 제거됨 (2024년 12월)
        'MA20_Slope',  # 20일 이동평균선 기울기
        'MA120_Slope',  # 120일 이동평균선 기울기
        'MA240_Slope',  # 240일 이동평균선 기울기
        'KOSPI_MA20_Slope',  # KOSPI 20일 이동평균선 기울기
        # 'PBR_log',  # PBR 로그 변환 (2024년 12월 제거)
        # 새로 추가된 피처
        'RVOL',  # 상대 거래량 (Relative Volume)
        '시총 회전율(1W)',  # 시총 회전율 1주 (5일 평균 거래대금 / 시가총액 * 100)
        '시총 회전율(3M)',  # 시총 회전율 3개월 (60일 평균 거래대금 / 시가총액 * 100)
        'RSI_Signal_Oscillator',  # RSI 신호 오실레이터 (RSI_14 - RSI_14.rolling(9).mean())
        'ATRr_5',  # ATR 비율 5일 (기준 - 1W)
        'ATRr_20',  # ATR 비율 20일 (기준 - 1M)
        'ATRr_60',  # ATR 비율 60일 (기준 - 3M)
        # ATR_Ratio_Short, ATR_Ratio_Trend 제거됨 (2024년 12월)
        # 'Eff_Ratio_10'  # 효율성 비율 10일 (2024년 12월 제거)
        
        # 2024년 12월 신규 추가 피처 (3종)
        'HV_Volatility_5',   # HV 변동성 1주 (일별 로그 수익률의 5일 표준편차)
        'HV_Volatility_20',  # HV 변동성 1개월 (일별 로그 수익률의 20일 표준편차)
        'HV_Volatility_60',  # HV 변동성 3개월 (일별 로그 수익률의 60일 표준편차)
        'VWAP_Disparity_5',  # VWAP 괴리율 1주 (5일 기준)
        # Gap 피처 제거
        # 신규 추가
        'Max_Drawdown_20',  # 최근 20일 최대 낙폭 (%)
        'CLV',  # Close Location Value (종가 위치 지수, 캔들 내 매수/매도 힘의 우위)
    ]
    
    # 전체 데이터 중앙값 계산 제거 (Fold별로 Train 데이터만으로 계산)
    # imputation_values는 더 이상 사용하지 않음 (호환성을 위해 None으로 설정)
    imputation_values = None
    log_info("   ℹ️ 결측치 처리는 Fold별로 Train 데이터만으로 계산합니다.")

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

    # --- 모든 Fold의 데이터를 미리 로드하여 캐싱 (날짜 기반 Expanding Window) ---
    log_info(f"\n--- 📁 Fold 데이터 로딩 및 구성 중 (날짜 기반 Expanding Window) ---")
    
    # fold_cache 디렉토리 경로 설정
    fold_cache_dir = os.path.join(os.path.dirname(os.path.expanduser(data_path)), "fold_cache")
    os.makedirs(fold_cache_dir, exist_ok=True)
    
    # 날짜 기반 Expanding Window Fold 범위 계산
    fold_ranges = calculate_expanding_fold_ranges(file_paths, warmup_days=250, val_period_days=365, n_folds=3)
    
    if not fold_ranges:
        log_critical("Fold 범위 계산에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    fold_data_cache = {}  # {fold_idx: (X_train, y_train, X_val, y_val)}
    
    for fold_info in fold_ranges:
        fold_idx = fold_info['fold']
        fold_cache_path = os.path.join(fold_cache_dir, f"fold_{fold_idx}_data.joblib")
        
        # fold 데이터 파일 존재 확인
        if os.path.exists(fold_cache_path):
            log_info(f"   ✅ Fold #{fold_idx+1}/3 캐시 파일을 찾았습니다. 로드 중...")
            load_start = datetime.now()
            try:
                fold_data = joblib.load(fold_cache_path)
                # 캐시 형식 확인 (날짜 기반이므로 imputation_map, meta_data도 포함될 수 있음)
                if len(fold_data) == 4:
                    X_train, y_train, X_val, y_val = fold_data
                    train_meta_data = None
                elif len(fold_data) == 5:
                    X_train, y_train, X_val, y_val, _ = fold_data  # imputation_map 무시
                    train_meta_data = None
                elif len(fold_data) == 6:
                    X_train, y_train, X_val, y_val, _, train_meta_data = fold_data  # imputation_map, meta_data
                else:
                    raise ValueError("캐시 형식이 맞지 않습니다.")
                
                # pandas DataFrame을 cuDF DataFrame으로 변환 (저장 시 pandas로 변환했으므로)
                if isinstance(X_train, pd.DataFrame):
                    X_train = cudf.from_pandas(X_train)
                if isinstance(X_val, pd.DataFrame):
                    X_val = cudf.from_pandas(X_val)
                if isinstance(y_train, (pd.Series, np.ndarray)):
                    y_train = cudf.Series(y_train) if not isinstance(y_train, cudf.Series) else y_train
                if isinstance(y_val, (pd.Series, np.ndarray)):
                    y_val = cudf.Series(y_val) if not isinstance(y_val, cudf.Series) else y_val
                
                load_time = (datetime.now() - load_start).total_seconds()
                
                if X_train is None or X_val is None:
                    log_warning(f"   ⚠️ Fold #{fold_idx+1} 캐시 파일 데이터가 유효하지 않습니다. 재로딩합니다.")
                    raise ValueError("Invalid cache data")
                
                # features 리스트에 있는 피처가 모두 있는지 확인
                missing_features = [f for f in features if f not in X_train.columns]
                if missing_features:
                    log_warning(f"   ⚠️ Fold #{fold_idx+1} 캐시 파일에 필요한 피처가 없습니다: {missing_features}. 재로딩합니다.")
                    raise ValueError(f"Missing features in cache: {missing_features}")
                
                # features 리스트에 있는 피처만 선택
                X_train = X_train[features]
                X_val = X_val[features]
                
                fold_data_cache[fold_idx] = (X_train, y_train, X_val, y_val, train_meta_data)
                log_info(f"   ✅ Fold #{fold_idx+1}/3 캐시 로드 완료: 훈련 {len(X_train):,}행, 검증 {len(X_val):,}행 ({load_time:.1f}초)")
            except Exception as e:
                log_warning(f"   ⚠️ Fold #{fold_idx+1} 캐시 파일 로드 실패: {e}. 재로딩합니다.")
                # 캐시 파일이 손상되었거나 피처가 불일치하는 경우 삭제 후 재로딩
                try:
                    os.remove(fold_cache_path)
                    log_info(f"   🗑️ Fold #{fold_idx+1} 캐시 파일 삭제 완료 (재생성을 위해)")
                except:
                    pass
                fold_data = None
        else:
            fold_data = None
        
        # 캐시 파일이 없거나 로드 실패한 경우 날짜 기반으로 로드
        if fold_data is None:
            log_info(f"   Fold #{fold_idx+1}/3 데이터 로딩 중...")
            log_info(f"      Train: {fold_info['train_start'].strftime('%Y-%m-%d')} ~ {fold_info['train_end'].strftime('%Y-%m-%d')}")
            log_info(f"      Val: {fold_info['val_start'].strftime('%Y-%m-%d')} ~ {fold_info['val_end'].strftime('%Y-%m-%d')}")
            
            load_start = datetime.now()
            
            # Train 로드 (맵 생성)
            X_train, y_train, train_imputation_map = load_data_period(
                file_paths, features,
                fold_info['train_start'], fold_info['train_end'],
                imputation_map=None  # Train 모드
            )
            
            # Val 로드 (Train에서 만든 맵 적용)
            X_val, y_val, _ = load_data_period(
                file_paths, features,
                fold_info['val_start'], fold_info['val_end'],
                imputation_map=train_imputation_map  # Val 모드
            )
            
            if X_train is None or X_val is None:
                log_warning(f"   ⚠️ Fold #{fold_idx+1} 데이터 로딩 실패. 건너뜁니다.")
                continue
            
            load_time = (datetime.now() - load_start).total_seconds()
            
            fold_data_cache[fold_idx] = (X_train, y_train, X_val, y_val)
            
            log_info(f"   ✅ Fold #{fold_idx+1}/3 로딩 완료: 훈련 {len(X_train):,}행, 검증 {len(X_val):,}행 ({load_time:.1f}초)")
            
            # fold 데이터 파일로 저장 (imputation_map도 함께 저장)
            log_info(f"   💾 Fold #{fold_idx+1}/3 데이터 파일 저장 중...")
            try:
                # cuDF DataFrame을 pandas DataFrame으로 변환 (joblib 호환성)
                # StringDtype 등 joblib이 직렬화하지 못하는 dtype 문제 해결
                if hasattr(X_train, 'to_pandas'):
                    X_train_pd = X_train.to_pandas()
                    # StringDtype을 object dtype으로 변환 (joblib 호환성)
                    for col in X_train_pd.columns:
                        if hasattr(X_train_pd[col].dtype, 'name') and 'string' in str(X_train_pd[col].dtype).lower():
                            X_train_pd[col] = X_train_pd[col].astype('object')
                else:
                    X_train_pd = X_train
                    # StringDtype을 object dtype으로 변환
                    for col in X_train_pd.columns:
                        if hasattr(X_train_pd[col].dtype, 'name') and 'string' in str(X_train_pd[col].dtype).lower():
                            X_train_pd[col] = X_train_pd[col].astype('object')
                
                if hasattr(X_val, 'to_pandas'):
                    X_val_pd = X_val.to_pandas()
                    # StringDtype을 object dtype으로 변환
                    for col in X_val_pd.columns:
                        if hasattr(X_val_pd[col].dtype, 'name') and 'string' in str(X_val_pd[col].dtype).lower():
                            X_val_pd[col] = X_val_pd[col].astype('object')
                else:
                    X_val_pd = X_val
                    # StringDtype을 object dtype으로 변환
                    for col in X_val_pd.columns:
                        if hasattr(X_val_pd[col].dtype, 'name') and 'string' in str(X_val_pd[col].dtype).lower():
                            X_val_pd[col] = X_val_pd[col].astype('object')
                
                # y_train, y_val도 변환 (cuDF Series인 경우)
                if hasattr(y_train, 'to_pandas'):
                    y_train_pd = y_train.to_pandas()
                else:
                    y_train_pd = y_train
                
                if hasattr(y_val, 'to_pandas'):
                    y_val_pd = y_val.to_pandas()
                else:
                    y_val_pd = y_val
                
                # 디렉토리 생성 (파일 저장 전)
                # fold_cache_path는 문자열이므로 os.path.dirname 사용
                fold_cache_dir = os.path.dirname(fold_cache_path)
                os.makedirs(fold_cache_dir, exist_ok=True)
                
                # 변환된 데이터 저장
                joblib.dump((X_train_pd, y_train_pd, X_val_pd, y_val_pd, train_imputation_map), fold_cache_path)
                log_info(f"   ✅ Fold #{fold_idx+1}/3 데이터 파일 저장 완료.")
            except Exception as e:
                log_warning(f"   ⚠️ Fold #{fold_idx+1} 데이터 파일 저장 실패: {e}. 학습은 계속 진행됩니다.")
                import traceback
                log_warning(f"   상세 오류:\n{traceback.format_exc()}")
    
    log_info(f"   ✅ Fold 데이터 로딩 완료 (총 {len(fold_data_cache)}개 Fold)")
    
    if not fold_data_cache:
        log_critical("Fold 데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # --- Trial 전 언더샘플링 적용 (모든 fold의 train 데이터에 적용) ---
    log_info(f"\n--- 🔄 Trial 전 언더샘플링 적용 중 (급락 패턴 우선 선택) ---")
    step_start = datetime.now()
    
    for fold in range(len(fold_data_cache)):
        fold_data = fold_data_cache[fold]
        if len(fold_data) == 4:
            X_train, y_train, X_val, y_val = fold_data
            train_meta_data = None
        elif len(fold_data) == 5:
            X_train, y_train, X_val, y_val, train_meta_data = fold_data
        else:
            X_train, y_train, X_val, y_val = fold_data[:4]
            train_meta_data = None
        
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
        # target = 1: 상승, target = 0: 급락
        minority_class_name = "상승" if minority_class_label == 1 else "급락"
        majority_class_name = "상승" if majority_class_label == 1 else "급락"
        
        log_info(f"   Fold #{fold+1}/3 클래스 분포:")
        log_info(f"      - 소수 클래스 ({minority_class_label}, {minority_class_name}) 샘플 수: {n_minority:,}개")
        log_info(f"      - 다수 클래스 ({majority_class_label}, {majority_class_name}) 샘플 수: {n_majority:,}개")
        
        # 언더샘플링: 다수 클래스를 소수 클래스 크기만큼 랜덤 선택 (1:1 비율)
        # [중요] 소수 클래스는 전체 사용 (소실 방지)
        if n_majority > n_minority:
            # 위치 인덱스 생성 (0부터 시작)
            all_indices = np.arange(len(y_train))
            y_train_pandas = y_train.to_pandas().values
            
            # 소수 클래스 인덱스 (전체 사용)
            minority_indices = all_indices[y_train_pandas == minority_class_label]
            
            # 다수 클래스 인덱스 (샘플링 대상)
            majority_indices = all_indices[y_train_pandas == majority_class_label]
            
            # Fold별 다양성을 위한 시드 (Fold마다 다른 난수 사용)
            rng = np.random.RandomState(42 + fold * 1000)
            
            log_info(f"      🔀 다수 클래스 랜덤 셔플 및 샘플링 중 (Fold별 다양성 확보)...")
            
            # [다양성 확보] 다수 클래스 셔플
            majority_indices_shuffled = majority_indices.copy()
            rng.shuffle(majority_indices_shuffled)
            
            # 1:1 비율로 샘플링
            target_majority_size = n_minority * 1
            selected_majority_indices = majority_indices_shuffled[:target_majority_size]
            
            # 인덱스 결합
            balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
            
            # [시계열 정합성] 시간 순서 유지 (과거 -> 미래)
            balanced_indices.sort()
            
            # 언더샘플링된 데이터 생성
            X_train_resampled = X_train.iloc[balanced_indices].reset_index(drop=True)
            y_train_resampled = y_train.iloc[balanced_indices].reset_index(drop=True)
            
            # 결과 확인
            y_train_resampled_pandas = y_train_resampled.to_pandas()
            value_counts_resampled = y_train_resampled_pandas.value_counts()
            n_minority_resampled = value_counts_resampled[minority_class_label]
            n_majority_resampled = value_counts_resampled[majority_class_label]
            del y_train_resampled_pandas
            
            # fold_data_cache 업데이트
            fold_data_cache[fold] = (X_train_resampled, y_train_resampled, X_val, y_val)
            
            # 원본 데이터 삭제
            del X_train, y_train, balanced_indices
            enhanced_gpu_memory_cleanup(force_defrag=False)
            gc.collect()
            
            log_info(f"      ✅ 언더샘플링 완료 (1:1 비율):")
            log_info(f"         - 소수 클래스 ({minority_class_label}): {n_minority:,}개 → {n_minority_resampled:,}개 (100% 사용)")
            log_info(f"         - 다수 클래스 ({majority_class_label}): {n_majority:,}개 → {n_majority_resampled:,}개")
        else:
            # 클래스 불균형이 없는 경우
            log_info(f"      ℹ️ 클래스 불균형이 없어 언더샘플링을 건너뜁니다.")
            # 원본 데이터 그대로 사용
            fold_data_cache[fold] = (X_train, y_train, X_val, y_val)
    
    log_info(f"   ✅ Trial 전 언더샘플링 완료: 총 {len(fold_data_cache)}개 Fold | 소요시간: {(datetime.now() - step_start).total_seconds():.1f}초")
    
    # 주의: 샘플링된 데이터는 메모리에서만 사용하고, fold_cache 파일은 원본 데이터로 유지합니다.
    # 다음 실행 시 원본 데이터를 로드하여 일관성 있는 샘플링을 보장합니다.

    log_info(f"\n--- 🤖 Optuna 하이퍼파라미터 최적화 시작 (n_trials={args.n_iter}) ---")

    try:
        study.optimize(
            lambda trial: objective(trial, fold_data_cache, features, args.max_depth, rng),
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
        'target_description': '10거래일 내 5% 이하로 떨어지지 않고 8% 이상 상승'
    }

    # --- 4. 최종 모델 훈련 및 저장 (캐시 데이터 재사용) ---
    try:
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_streams'] = 1  # GPU 병렬 처리 개선 (속도 향상)
        train_final_ensemble_model(fold_data_cache, features, best_params, rng, optimization_results, training_config, data_path)
    except Exception as e:
        log_critical("최종 모델 훈련 또는 저장 중 오류 발생", exception=e)
    finally:
        # 최종 모델 훈련 완료 후 데이터 캐시 정리 (함수 내에서 이미 삭제했지만 안전을 위해 재정리)
        for fold_key in list(fold_data_cache.keys()):
            fold_data = fold_data_cache[fold_key]
            if fold_data:
                try:
                    if len(fold_data) >= 4:
                        X_train, y_train, X_val, y_val = fold_data[:4]
                        del X_train, y_train, X_val, y_val
                    if len(fold_data) >= 5:
                        del fold_data[4]  # meta_data
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
