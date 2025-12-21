"""
머신러닝 모델 훈련 스크립트
========================

이 파일은 주식 상승 예측을 위한 머신러닝 모델을 훈련합니다.
RandomForest 알고리즘을 사용하여 아래 타겟(이진 분류)을 예측합니다:
- 향후 10거래일 동안 -5% 이상 하락 없이, +8% 이상 상승을 한 번이라도 달성 여부

주요 기능:
- 대용량 데이터 처리 및 메모리 최적화
- 하이퍼파라미터 자동 튜닝
- 교차 검증을 통한 모델 성능 평가
- 훈련된 모델 및 전처리기 저장
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from datetime import datetime, timedelta
import os
import sys
import io
import shutil # 폴더 삭제를 위해 shutil 라이브러리 임포트
import gc
import psutil
import locale
import platform
import optuna
from optuna.samplers import TPESampler
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from typing import List, Tuple, Optional, Dict
import ast

# Windows 환경에서 로케일 설정 (FinanceDataReader 내부 오류 방지)
if platform.system() == 'Windows':
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 로케일 설정 실패 시 기본값 유지
        pass

# 크로스 플랫폼 인코딩 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_processor
from path_manager import path_manager
from logger import log_info, log_warning, log_error 

warnings.filterwarnings('ignore', category=FutureWarning)

def get_memory_usage():
    """현재 메모리 사용량을 반환합니다."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB 단위

def log_memory_usage(stage_name):
    """메모리 사용량을 로그로 출력합니다."""
    memory_mb = get_memory_usage()
    log_info(f"   💾 {stage_name} - 메모리 사용량: {memory_mb:.1f} MB")

def safe_memory_cleanup():
    """안전한 메모리 정리를 수행합니다."""
    try:
        gc.collect()
        log_info("   🧹 메모리 정리 완료")
    except Exception as e:
        log_warning(f"   ⚠️ 메모리 정리 중 오류: {e}")

def check_memory_and_cleanup(threshold_mb=8000):
    """메모리 사용량을 확인하고 필요시 정리합니다."""
    memory_mb = get_memory_usage()
    if memory_mb > threshold_mb:
        log_warning(f"   ⚠️ 메모리 사용량이 높습니다: {memory_mb:.1f} MB")
        safe_memory_cleanup()
        new_memory_mb = get_memory_usage()
        log_info(f"   ✅ 메모리 정리 후: {new_memory_mb:.1f} MB")
        return True
    return False

def calculate_shap_importance(model, X_train_scaled, y_train, features, sample_size=1000):
    """
    SHAP 중요도 계산 함수 (계층적 샘플링)
    
    Args:
        model: 학습된 모델
        X_train_scaled: 스케일링된 학습 데이터
        y_train: 학습 타겟
        features: 피처 이름 리스트
        sample_size: 샘플 크기 (기본값: 1000)
    
    Returns:
        list: [(feature, importance), ...] 형태의 리스트
    """
    try:
        import shap
        log_info(f"   📊 SHAP 중요도 계산 중... (샘플 크기: {sample_size}건)")
        log_memory_usage("SHAP 계산 시작")
        
        # 계층적 샘플링
        # numpy array를 pandas DataFrame으로 변환 (resample 호환성)
        if isinstance(X_train_scaled, np.ndarray):
            X_train_df = pd.DataFrame(X_train_scaled, columns=features)
        else:
            X_train_df = X_train_scaled
        
        # y_train을 pandas Series로 변환 (stratify 호환성)
        if isinstance(y_train, np.ndarray):
            y_train_series = pd.Series(y_train)
        else:
            y_train_series = y_train
        
        actual_sample_size = min(sample_size, len(X_train_df))
        if actual_sample_size < len(X_train_df):
            # stratify를 안전하게 처리하기 위해 y_train을 1D 배열로 보장
            if isinstance(y_train_series, pd.Series):
                y_train_1d = y_train_series.values
            elif isinstance(y_train_series, np.ndarray):
                y_train_1d = y_train_series.flatten() if y_train_series.ndim > 1 else y_train_series
            else:
                y_train_1d = np.array(y_train_series).flatten()
            
            # resample 호출 (stratify는 1D 배열이어야 함)
            try:
                X_shap_sample, y_shap_sample = resample(
                    X_train_df, 
                    y_train_1d, 
                    n_samples=actual_sample_size, 
                    stratify=y_train_1d, 
                    random_state=42
                )
            except Exception as e:
                # stratify 실패 시 stratify 없이 재시도
                log_warning(f"   ⚠️ 계층적 샘플링 실패, 일반 샘플링으로 재시도: {e}")
                X_shap_sample, y_shap_sample = resample(
                    X_train_df, 
                    y_train_1d, 
                    n_samples=actual_sample_size, 
                    random_state=42
                )
            # numpy array로 변환 (SHAP 입력 형식)
            if isinstance(X_shap_sample, pd.DataFrame):
                X_shap_sample = X_shap_sample.values
            if isinstance(y_shap_sample, pd.Series):
                y_shap_sample = y_shap_sample.values
            log_info(f"   📊 계층적 샘플링 완료: {len(X_shap_sample):,}건")
        else:
            # numpy array로 변환
            if isinstance(X_train_df, pd.DataFrame):
                X_shap_sample = X_train_df.values
            else:
                X_shap_sample = X_train_scaled
            if isinstance(y_train_series, pd.Series):
                y_shap_sample = y_train_series.values
            else:
                y_shap_sample = y_train_series
            log_info(f"   📊 전체 데이터 사용: {len(X_shap_sample):,}건")
        
        # SHAP TreeExplainer 사용
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap_sample)
        
        # 이진 분류의 경우 shap_values는 리스트 [class_0, class_1] 또는 단일 array
        # 타입 체크를 더 안전하게 처리
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                # 리스트 형태이고 2개 이상의 요소가 있는 경우 (이진 분류)
                shap_values_class1 = shap_values[1]  # 클래스 1(상승)에 대한 SHAP 값
            elif len(shap_values) == 1:
                # 리스트 형태이지만 1개만 있는 경우
                shap_values_class1 = shap_values[0]
            else:
                # 빈 리스트인 경우
                log_warning("   ⚠️ SHAP 값이 빈 리스트입니다.")
                return None
        else:
            # 단일 numpy array인 경우
            shap_values_class1 = shap_values
        
        # numpy array로 변환 (안전성 확보)
        if not isinstance(shap_values_class1, np.ndarray):
            shap_values_class1 = np.array(shap_values_class1)
        
        # 피처별 평균 절댓값으로 중요도 계산
        shap_importance = np.abs(shap_values_class1).mean(axis=0)
        
        # 정규화 (합이 1이 되도록, 0으로 나누기 방지)
        importance_sum = shap_importance.sum()
        if importance_sum > 0:
            shap_importance = shap_importance / importance_sum
        else:
            log_warning("   ⚠️ SHAP 중요도 합이 0입니다. 정규화를 건너뜁니다.")
        
        # 피처 이름과 함께 정렬
        shap_importance_list = list(zip(features, shap_importance))
        shap_importance_list.sort(key=lambda x: x[1], reverse=True)
        
        log_info(f"   ✅ SHAP 중요도 계산 완료")
        log_memory_usage("SHAP 계산 완료")
        
        # 메모리 정리
        del explainer, shap_values, shap_values_class1, X_shap_sample, y_shap_sample
        gc.collect()
        
        return shap_importance_list
        
    except ImportError:
        log_warning("   ⚠️ SHAP 라이브러리가 설치되지 않아 SHAP 중요도를 계산할 수 없습니다.")
        return None
    except Exception as e:
        log_error(f"   ❌ SHAP 중요도 계산 중 오류 발생: {e}")
        return None

def calculate_permutation_importance(model, X_test_scaled, y_test, features, n_repeats=5):
    """
    Permutation Importance 계산 함수
    
    Args:
        model: 학습된 모델
        X_test_scaled: 스케일링된 테스트 데이터
        y_test: 테스트 타겟
        features: 피처 이름 리스트
        n_repeats: 반복 횟수 (기본값: 5)
    
    Returns:
        list: [(feature, importance), ...] 형태의 리스트
    """
    try:
        log_info(f"   📊 Permutation Importance 계산 중... (반복 횟수: {n_repeats})")
        log_memory_usage("Permutation Importance 계산 시작")
        
        # Permutation Importance 계산
        perm_result = permutation_importance(
            model, 
            X_test_scaled, 
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1,  # 메모리 절약
            scoring='roc_auc'
        )
        
        # 평균 중요도 추출
        perm_importance = perm_result.importances_mean
        
        # 피처 이름과 함께 정렬
        perm_importance_list = list(zip(features, perm_importance))
        perm_importance_list.sort(key=lambda x: x[1], reverse=True)
        
        log_info(f"   ✅ Permutation Importance 계산 완료")
        log_memory_usage("Permutation Importance 계산 완료")
        
        # 메모리 정리
        del perm_result
        gc.collect()
        
        return perm_importance_list
        
    except Exception as e:
        log_error(f"   ❌ Permutation Importance 계산 중 오류 발생: {e}")
        return None

def _sanitize_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Inf 값을 NaN으로 치환해 스케일러/중앙값 계산이 깨지지 않도록 합니다."""
    try:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X = X.copy()
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return X
    except Exception:
        return X

def compute_imputation_values_train_only(X_train: pd.DataFrame) -> Dict[str, float]:
    """Train 데이터에서만 중앙값을 계산해 결측치 대체값을 만듭니다(누수 방지)."""
    X_train = _sanitize_numeric_frame(X_train)
    med = X_train.median(numeric_only=True)
    # pandas Series -> dict (joblib 저장/로드 및 fillna 호환)
    return med.to_dict()

def expanding_time_series_folds(
    dates: pd.Series,
    n_splits: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    gpuStock 철학에 맞춘 Expanding Window 방식의 시간 기반 CV split 생성.
    - Train은 누적(expanding)
    - Val은 바로 다음 구간(미래)
    - 같은 날짜는 Train/Val에 동시에 걸리지 않도록 unique date 단위로 나눔
    Returns:
        (train_idx_positions, val_idx_positions) 리스트. 인덱스는 '위치' 기준(0..N-1).
    """
    if dates is None or len(dates) == 0:
        return []
    d = pd.to_datetime(dates).reset_index(drop=True)
    # unique date 단위로 split (동일 날짜가 양쪽에 걸리는 것 방지)
    unique_dates = pd.Series(d.dt.date.unique()).sort_values().tolist()
    if len(unique_dates) < (n_splits + 1):
        return []
    fold_size = len(unique_dates) // (n_splits + 1)
    if fold_size <= 0:
        return []

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = fold_size * (i + 2)
        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        if not train_dates or not val_dates:
            continue
        train_idx = np.where(d.dt.date.isin(train_dates).values)[0]
        val_idx = np.where(d.dt.date.isin(val_dates).values)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds

def expanding_time_series_folds_gpustock(
    dates: pd.Series,
    warmup_days: int = 250,
    val_period_days: int = 365,
    n_folds: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    gpuStock의 LGBM 학습과 동일한 Fold 구성 방식.
    - 웜업(warmup_days) 기간은 학습에서 제외
    - 검증 기간은 고정(val_period_days)이고, 마지막 n_folds개 구간을 1년 단위로 검증
    - Train은 항상 시작점부터 누적(expanding)

    Returns:
        (train_idx_positions, val_idx_positions) 리스트. 인덱스는 '위치' 기준(0..N-1).
    """
    if dates is None or len(dates) == 0:
        return []

    d = pd.to_datetime(dates, errors='coerce').reset_index(drop=True)
    if d.isna().all():
        return []

    # 날짜 비교는 시간 성분을 제거한 기준으로 수행
    d_norm = d.dt.normalize()
    min_date = d_norm.min()
    max_date = d_norm.max()
    if pd.isna(min_date) or pd.isna(max_date):
        return []

    actual_start_date = (min_date + pd.Timedelta(days=warmup_days)).normalize()
    actual_end_date = max_date.normalize()
    if actual_start_date >= actual_end_date:
        return []

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_folds):
        end_offset = (n_folds - 1 - fold_idx) * val_period_days
        val_end = (actual_end_date - pd.Timedelta(days=end_offset)).normalize()
        val_start = (val_end - pd.Timedelta(days=val_period_days)).normalize()

        train_start = actual_start_date
        train_end = val_start

        # 검증 구간이 학습 시작보다 과거로 밀리면 조정
        if val_start < actual_start_date:
            val_start = actual_start_date
            val_end = (val_start + pd.Timedelta(days=val_period_days)).normalize()
            train_end = val_start

        if train_end <= train_start:
            continue

        # Train: [train_start, train_end) / Val: [val_start, val_end]
        # (겹침 방지를 위해 train_end는 제외, val은 범위 내 포함)
        train_mask = (d_norm >= train_start) & (d_norm < train_end)
        val_mask = (d_norm >= val_start) & (d_norm <= val_end)

        train_idx = np.where(train_mask.values)[0]
        val_idx = np.where(val_mask.values)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))

    return folds

def time_series_cv_auc(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    features: List[str],
    n_splits: int = 3
) -> float:
    """
    시간 기반 Expanding CV로 ROC-AUC 평균을 계산합니다.
    - 각 fold마다 Train-only로 imputation/scaler fit
    """
    folds = expanding_time_series_folds(dates, n_splits=n_splits)
    if not folds:
        return 0.0

    scores: List[float] = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    dates = pd.to_datetime(dates).reset_index(drop=True)

    for (tr_idx, va_idx) in folds:
        X_tr = X.iloc[tr_idx][features].copy()
        y_tr = y.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx][features].copy()
        y_va = y.iloc[va_idx].copy()

        # Train-only imputation
        imp = compute_imputation_values_train_only(X_tr)
        X_tr = _sanitize_numeric_frame(X_tr)
        X_va = _sanitize_numeric_frame(X_va)
        X_tr.fillna(imp, inplace=True)
        X_va.fillna(imp, inplace=True)

        # Train-only scaling
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # ✅ 경고 방지: fit/predict 모두 feature name이 있는 DataFrame 형태로 통일
        # (특히 LGBMClassifier는 fit 시 feature names를 기억하므로, predict 입력이 numpy면 sklearn warning 발생 가능)
        X_tr_s = pd.DataFrame(X_tr_s, columns=features)
        X_va_s = pd.DataFrame(X_va_s, columns=features)

        model = model_builder()
        try:
            model.fit(X_tr_s, y_tr)
            y_va_proba = model.predict_proba(X_va_s)[:, 1]
            scores.append(roc_auc_score(y_va, y_va_proba))
        except Exception:
            # fold 실패는 점수 0으로 처리 (Optuna objective 안정성)
            scores.append(0.0)
        finally:
            try:
                del model, X_tr, X_va, X_tr_s, X_va_s, scaler
            except Exception:
                pass
            gc.collect()

    if not scores:
        return 0.0
    return float(np.mean(scores))

def load_training_data_from_file(training_data_path):
    """학습 데이터 파일에서 로드"""
    try:
        log_info(f"📂 학습 데이터 파일에서 로드 중: {training_data_path}")
        final_df = pd.read_parquet(training_data_path)
        log_info(f"✅ 학습 데이터 파일 로드 완료: {len(final_df):,} 행")
        log_memory_usage("학습 데이터 파일 로드 완료")
        return final_df
    except Exception as e:
        log_error(f"학습 데이터 파일 로드 실패: {e}")
        return None

def save_training_data_to_file(final_df, training_data_path):
    """학습 데이터를 파일로 저장"""
    try:
        log_info(f"💾 학습 데이터를 파일로 저장 중: {training_data_path}")
        # 디렉토리가 없으면 생성
        training_data_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(training_data_path, index=False, compression='snappy')
        log_info(f"✅ 학습 데이터 파일 저장 완료: {len(final_df):,} 행")
        log_memory_usage("학습 데이터 파일 저장 완료")
        return True
    except Exception as e:
        log_error(f"학습 데이터 파일 저장 실패: {e}")
        return False

def split_by_date(X, y, dates, test_size=0.3):
    """
    날짜 기반 데이터 분할 함수 (미래 데이터 참조 방지)
    
    Args:
        X: 피처 데이터 (DataFrame)
        y: 타겟 데이터 (Series)
        dates: 날짜 데이터 (Series, datetime 타입)
        test_size: 테스트 데이터 비율 (기본값: 0.3)
    
    Returns:
        X_train, X_test, y_train, y_test, train_dates, test_dates
    """
    # 날짜 기준으로 정렬
    sorted_indices = dates.argsort()
    X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
    y_sorted = y.iloc[sorted_indices].reset_index(drop=True)
    dates_sorted = dates.iloc[sorted_indices].reset_index(drop=True)
    
    # 날짜 기준으로 분할 (과거 데이터 = Train, 미래 데이터 = Test)
    # 같은 날짜의 데이터가 Train과 Test 양쪽에 포함되지 않도록 처리
    split_idx = int(len(X_sorted) * (1 - test_size))
    
    # split_idx 위치의 날짜 확인
    split_date = dates_sorted.iloc[split_idx]
    
    # 같은 날짜의 모든 데이터를 Train에 포함시키기 위해
    # split_date와 같은 날짜의 마지막 인덱스를 찾음
    same_date_mask = dates_sorted.dt.date == split_date.date()
    if same_date_mask.any():
        # 같은 날짜의 마지막 인덱스 찾기 (reset_index 후이므로 인덱스는 0부터 시작)
        same_date_positions = dates_sorted.index[same_date_mask]
        last_same_date_idx = same_date_positions.max()
        # 같은 날짜의 모든 데이터를 Train에 포함
        split_idx = last_same_date_idx + 1
    
    # split_idx가 범위를 벗어나지 않도록 확인
    if split_idx >= len(X_sorted):
        split_idx = len(X_sorted) - 1
    
    X_train = X_sorted.iloc[:split_idx].copy()
    X_test = X_sorted.iloc[split_idx:].copy()
    y_train = y_sorted.iloc[:split_idx].copy()
    y_test = y_sorted.iloc[split_idx:].copy()
    train_dates = dates_sorted.iloc[:split_idx].copy()
    test_dates = dates_sorted.iloc[split_idx:].copy()
    
    # 데이터 정합성 검증: Train의 최대 날짜 < Test의 최소 날짜
    train_max_date = train_dates.max()
    test_min_date = test_dates.min()
    
    # 같은 날짜가 Train과 Test 양쪽에 포함되어 있으면 조정
    if train_max_date >= test_min_date:
        # Train의 최대 날짜와 같은 날짜의 모든 데이터를 Train에 포함
        train_max_date_mask = dates_sorted.dt.date == train_max_date.date()
        if train_max_date_mask.any():
            # 같은 날짜의 마지막 인덱스 찾기
            same_date_positions = dates_sorted.index[train_max_date_mask]
            last_same_date_idx = same_date_positions.max()
            # 같은 날짜의 모든 데이터를 Train에 포함
            split_idx = last_same_date_idx + 1
            
            # split_idx가 범위를 벗어나지 않도록 확인
            if split_idx >= len(X_sorted):
                split_idx = len(X_sorted) - 1
            
            # 재분할
            X_train = X_sorted.iloc[:split_idx].copy()
            X_test = X_sorted.iloc[split_idx:].copy()
            y_train = y_sorted.iloc[:split_idx].copy()
            y_test = y_sorted.iloc[split_idx:].copy()
            train_dates = dates_sorted.iloc[:split_idx].copy()
            test_dates = dates_sorted.iloc[split_idx:].copy()
            
            train_max_date = train_dates.max()
            if len(test_dates) > 0:
                test_min_date = test_dates.min()
            else:
                # Test 데이터가 없으면 오류
                log_error(f"❌ 심각한 오류: Test 데이터가 없습니다!")
                log_error(f"   Train 데이터 수: {len(X_train):,}행")
                raise ValueError("날짜 기반 분할이 실패했습니다. Test 데이터가 없습니다.")
    
    # 최종 검증
    if len(test_dates) > 0 and train_max_date >= test_min_date:
        log_error(f"❌ 심각한 오류: 날짜 기반 분할 실패!")
        log_error(f"   Train 최대 날짜: {train_max_date}")
        log_error(f"   Test 최소 날짜: {test_min_date}")
        log_error(f"   Train 데이터 수: {len(X_train):,}행")
        log_error(f"   Test 데이터 수: {len(X_test):,}행")
        # 디버깅 정보 추가
        log_error(f"   Train 최대 날짜의 데이터 수: {(train_dates.dt.date == train_max_date.date()).sum()}행")
        log_error(f"   Test 최소 날짜의 데이터 수: {(test_dates.dt.date == test_min_date.date()).sum()}행")
        raise ValueError("날짜 기반 분할이 실패했습니다. Train 데이터의 최대 날짜가 Test 데이터의 최소 날짜보다 크거나 같습니다.")
    
    log_info(f"   ✅ 날짜 기반 분할 완료:")
    log_info(f"      Train: {train_dates.min().strftime('%Y-%m-%d')} ~ {train_max_date.strftime('%Y-%m-%d')} ({len(X_train):,}행)")
    log_info(f"      Test:  {test_min_date.strftime('%Y-%m-%d')} ~ {test_dates.max().strftime('%Y-%m-%d')} ({len(X_test):,}행)")
    log_info(f"      ✅ 미래 데이터 참조 방지 확인: Train 최대 날짜 < Test 최소 날짜")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates

def create_training_data(years=None):
    """
    학습 데이터 생성 함수
    
    Args:
        years: 학습에 사용할 최근 N년치 데이터 (None이면 전체 데이터)
    
    Returns:
        X, y, features, imputation_values, dates
    """
    training_data_path = path_manager.get_training_data_path()
    
    # 학습에 사용할 피처 정의
    # - "정답 소스"를 gpuStock의 실제 학습 스크립트에서 자동 추출하여(가능하면) 동기화합니다.
    # - 사용자가 gpuStock을 최신 버전으로 교체하더라도, 루트(CPU) 학습 피처가 자동 반영되도록 합니다.
    def _extract_features_from_gpustock() -> Optional[List[str]]:
        try:
            gpustock_train_lgbm = path_manager.project_root / 'gpuStock' / 'scripts' / 'train_lgbm_gpu_main.py'
            gpustock_train_rf = path_manager.project_root / 'gpuStock' / 'scripts' / 'train_gpu_main.py'
            gpustock_train = gpustock_train_lgbm if gpustock_train_lgbm.exists() else gpustock_train_rf
            if not gpustock_train.exists():
                return None
            src = gpustock_train.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(src)

            feature_lists: List[List[str]] = []

            class FeatureVisitor(ast.NodeVisitor):
                def visit_Assign(self, node: ast.Assign):
                    try:
                        for t in node.targets:
                            if isinstance(t, ast.Name) and t.id == 'features' and isinstance(node.value, ast.List):
                                vals: List[str] = []
                                ok = True
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        vals.append(elt.value)
                                    elif isinstance(elt, ast.Str):
                                        vals.append(elt.s)
                                    else:
                                        ok = False
                                        break
                                if ok and vals:
                                    feature_lists.append(vals)
                    except Exception:
                        pass
                    self.generic_visit(node)

            FeatureVisitor().visit(tree)
            if not feature_lists:
                return None

            # 파일 내에 features가 여러 번 정의될 수 있으므로 "마지막 정의"를 우선 사용
            return feature_lists[-1]
        except Exception:
            return None

    def get_training_features() -> List[str]:
        # fallback (현재 repo의 gpuStock 기준으로 맞춰둔 리스트)
        default = [
            'log_mktcap',
            '52주_신고가_비율',
            'ADX_14',
            'disparity_120',
            'disparity_240',
            'disparity_20',
            'KOSPI_disparity_20',
            'Trend_Pullback_Score',
            'Position_Range_60',
            'MA20_Slope',
            'MA120_Slope',
            'MA240_Slope',
            'KOSPI_MA20_Slope',
            'RVOL',
            '시총 회전율(1W)',
            '시총 회전율(3M)',
            'RSI_Signal_Oscillator',
            'ATRr_5',
            'ATRr_20',
            'ATRr_60',
            'HV_Volatility_5',
            'HV_Volatility_20',
            'HV_Volatility_60',
            'VWAP_Disparity_5',
            'Max_Drawdown_20',
        ]
        extracted = _extract_features_from_gpustock()
        if extracted:
            log_info(f"✅ gpuStock 학습 피처 자동 동기화: {len(extracted)}개")
            return extracted
        log_warning("⚠️ gpuStock 학습 피처 자동 추출 실패(기본 리스트 사용)")
        return default

    features = get_training_features()
    target = 'target'

    # ======================================================================
    # 타겟 정의(버전) 메타데이터
    # - target 컬럼명은 동일해도 "계산식"이 바뀌면 기존 training_data.parquet이 오염될 수 있으므로
    #   메타파일로 변경을 감지해 자동 재생성합니다.
    # ======================================================================
    TARGET_SPEC = {
        "name": "10d_drawdown_floor_-5pct_and_any_hit_+8pct",
        "horizon_trading_days": 10,
        "min_ratio_floor": 0.95,  # future_min / now >= 0.95
        "max_ratio_hit": 1.08,    # future_max / now >= 1.08
        "notes": "향후 10거래일 동안 -5% 이상 하락 없이, +8% 이상 상승 1회라도",
    }
    training_meta_path = training_data_path.with_suffix('.meta.json')
    
    # 학습 데이터 파일이 있는지 확인
    if training_data_path.exists():
        log_info("📂 기존 학습 데이터 파일을 발견했습니다.")
        final_df = load_training_data_from_file(training_data_path)
        
        if final_df is not None and not final_df.empty:
            # 구버전 캐시(피처 세트 변경/타겟 변경)면 자동으로 재생성 (원복: 누락 피처 자동 제외 금지)
            required_cols = set(features + [target, 'date'])
            missing_cols = [c for c in required_cols if c not in final_df.columns]
            if missing_cols:
                log_warning(f"⚠️ 기존 학습 데이터 파일이 현재 피처/타겟과 호환되지 않습니다. 누락 컬럼: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")
                log_info("   🔄 학습 데이터 파일을 새로 수집/생성합니다.")
                final_df = None
            else:
                # 타겟 정의/피처 정의가 바뀌었는지 메타로 추가 검증
                meta_ok = True
                try:
                    if training_meta_path.exists():
                        with open(training_meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        if meta.get('target_spec') != TARGET_SPEC:
                            meta_ok = False
                            log_warning("⚠️ 기존 학습 데이터 메타의 타겟 정의가 현재 설정과 다릅니다. 재생성합니다.")
                        if meta.get('features') != features:
                            meta_ok = False
                            log_warning("⚠️ 기존 학습 데이터 메타의 피처 리스트가 현재 설정과 다릅니다. 재생성합니다.")
                    else:
                        meta_ok = False
                        log_warning("⚠️ 기존 학습 데이터 메타 파일이 없습니다. 안전을 위해 재생성합니다.")
                except Exception as e:
                    meta_ok = False
                    log_warning(f"⚠️ 학습 데이터 메타 검증 실패: {e}. 안전을 위해 재생성합니다.")

                if meta_ok:
                    log_info("✅ 기존 학습 데이터 파일을 사용합니다.")
                else:
                    final_df = None
        else:
            log_warning("⚠️ 기존 학습 데이터 파일이 유효하지 않습니다. 새로 수집합니다.")
            final_df = None
    else:
        log_info("📂 학습 데이터 파일이 없습니다. 새로 수집합니다.")
        final_df = None
    
    # 파일이 없거나 유효하지 않으면 새로 수집
    if final_df is None or final_df.empty:
        if years is None:
            log_info("🚀 전체 기간 데이터 수집을 통해 학습 데이터 생성을 시작합니다...")
            start_date_for_cacher = '2015-01-01'
            # 오늘 기준 2개월 전까지만 수집 (학습 데이터는 최신 데이터 제외)
            end_date_for_cacher = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            log_info(f"   📅 수집 기간: {start_date_for_cacher} ~ {end_date_for_cacher} (오늘 기준 2개월 전까지)")
        else:
            # 최근 N년치 데이터 수집
            # 오늘 기준 2개월 전까지만 수집 (학습 데이터는 최신 데이터 제외)
            end_date = datetime.now() - timedelta(days=60)  # 오늘 기준 2개월 전
            start_date = end_date - timedelta(days=years * 365)
            start_date_for_cacher = start_date.strftime('%Y-%m-%d')
            end_date_for_cacher = end_date.strftime('%Y-%m-%d')
            log_info(f"🚀 최근 {years}년치 데이터 수집을 통해 학습 데이터 생성을 시작합니다...")
            log_info(f"   📅 수집 기간: {start_date_for_cacher} ~ {end_date_for_cacher} (오늘 기준 2개월 전까지)")
        log_memory_usage("학습 데이터 생성 시작")
        
        try:
            # 실시간 데이터 수집 (학습용이므로 팩터 점수 계산 건너뛰기)
            final_df = data_processor.get_preprocessed_data(
                start_date_for_cacher, 
                end_date_for_cacher, 
                calculate_factor_scores=False  # 학습 데이터에는 팩터 점수 불필요
            )
            log_memory_usage("데이터 로딩 완료")
            check_memory_and_cleanup()
        except MemoryError as e:
            log_error(f"메모리 부족으로 데이터 로딩 실패: {e}")
            log_info("   🔄 메모리 정리 후 재시도합니다...")
            safe_memory_cleanup()
            final_df = data_processor.get_preprocessed_data(
                start_date_for_cacher, 
                end_date_for_cacher, 
                calculate_factor_scores=False  # 학습 데이터에는 팩터 점수 불필요
            )
            log_memory_usage("재시도 후 데이터 로딩 완료")
        
        if final_df is None or final_df.empty:
            log_error("데이터를 가져오는 데 실패했습니다.")
            return None, None, None, None
        
        # 저장할 컬럼: features + target + date, 종목코드 (메타데이터)
        save_columns = features + [target]
        if 'date' in final_df.columns:
            save_columns.append('date')
        if '종목코드' in final_df.columns:
            save_columns.append('종목코드')
        
        # 실제 존재하는 컬럼만 선택
        available_columns = [col for col in save_columns if col in final_df.columns]
        final_df_to_save = final_df[available_columns].copy()
        
        # 수집한 데이터를 파일로 저장 (필요한 컬럼만)
        if save_training_data_to_file(final_df_to_save, training_data_path):
            log_info("✅ 학습 데이터 파일이 생성되었습니다. 다음 실행부터는 이 파일을 사용합니다.")
            log_info(f"   📊 저장된 컬럼: {len(available_columns)}개 (학습에 필요한 컬럼만 저장)")
            # 메타 저장 (타겟 정의/피처 정의 변경 감지용)
            try:
                meta = {
                    "created_at": datetime.now().isoformat(),
                    "features": features,
                    "target_column": target,
                    "target_spec": TARGET_SPEC,
                }
                with open(training_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                log_warning(f"⚠️ 학습 데이터 메타 저장 실패(무시하고 진행): {e}")
        else:
            log_warning("⚠️ 학습 데이터 파일 저장에 실패했지만 계속 진행합니다.")

    log_info(f"\n--- 생성된 학습 데이터 요약 ---")
    log_info(f"1. 전체 수집 데이터 (Raw): {len(final_df):,} 행")
    log_memory_usage("원본 데이터 로딩")
    
    # 워밍업 기간 제외 (데이터가 2015년부터 시작하는 경우)
    if 'date' in final_df.columns:
        final_df['date'] = pd.to_datetime(final_df['date'])
        min_date = final_df['date'].min()
        if min_date.year <= 2015:
            training_start_date = '2016-01-01'
            final_df = final_df[final_df['date'] >= pd.to_datetime(training_start_date)]
            log_info(f"2. 워밍업 기간(2015년) 제외 후 실제 학습 데이터: {len(final_df):,} 행")
        else:
            log_info(f"2. 실제 학습 데이터: {len(final_df):,} 행")
    else:
        log_info(f"2. 실제 학습 데이터: {len(final_df):,} 행")
    
    log_memory_usage("데이터 필터링 완료")
    check_memory_and_cleanup()

    # 필요한 컬럼이 모두 있는지 확인 (원복: 누락 시 중단)
    for col in features + [target]:
        if col not in final_df.columns:
            log_error(f"오류: 필요한 컬럼 '{col}'이 데이터프레임에 없습니다.")
            return None, None, None, None, None
    
    # 날짜 컬럼 확인 (필수)
    if 'date' not in final_df.columns:
        log_error("❌ 심각한 오류: 'date' 컬럼이 데이터에 없습니다. 날짜 기반 분할을 수행할 수 없습니다.")
        return None, None, None, None, None
            
    final_df.dropna(subset=[target], inplace=True)
    log_info(f"3. 타겟 변수 결측치 제거 후: {len(final_df):,} 행")
    log_memory_usage("결측치 제거 완료")

    if final_df.empty:
        log_error("오류: 최종 학습 데이터가 비어있습니다.")
        return None, None, None, None, None

    # 메모리 효율적인 데이터 타입 변환
    log_info("   🔄 메모리 효율적인 데이터 타입 변환 중...")
    
    # 날짜 컬럼 보존
    dates = pd.to_datetime(final_df['date'])
    
    # 필요한 컬럼만 선택하여 메모리 절약
    required_columns = features + [target]
    final_df = final_df[required_columns]
    log_info(f"   📊 필요한 컬럼만 선택: {len(required_columns)}개 컬럼")

    # (원복) 메타 갱신은 데이터 파일 생성 시점에만 수행합니다.
    
    # 데이터 타입 최적화
    X = final_df[features].astype(np.float32)  # float64 -> float32로 메모리 절약
    y = final_df[target]
    
    # 원본 데이터프레임 메모리 해제
    del final_df
    gc.collect()
    
    log_memory_usage("데이터 타입 변환 완료")
    
    # ⚠️ 중요: 결측치 대체값(imputation) 계산/적용은 Train 데이터에서만 수행해야 데이터 누수를 방지할 수 있습니다.
    # 여기서는 결측치를 유지한 채로 반환합니다. (gpuStock 방식과 동일 철학)
    imputation_values = None
    
    log_info(f"4. 최종 학습 데이터셋 (X): {X.shape}")
    log_info(f"   - 타겟 분포 (y):\n{y.value_counts(normalize=True).to_string()}")
    log_info(f"   - 날짜 범위: {dates.min().strftime('%Y-%m-%d')} ~ {dates.max().strftime('%Y-%m-%d')}")
    log_info("---------------------------------")
    log_memory_usage("최종 학습 데이터 준비 완료")

    log_info("✅ 학습 데이터 생성 완료!")
    return X, y, features, imputation_values, dates

def train_evaluate_and_save_model(X, y, features, imputation_values, dates, n_jobs, n_iter, max_depth_list, model_path=None):
    if X is None or y is None or X.empty or y.empty:
        log_error("학습 데이터가 없어 모델링을 건너뜁니다.")
        return
    
    if dates is None or dates.empty:
        log_error("❌ 날짜 데이터가 없어 모델링을 건너뜁니다.")
        return

    # 통일된 경로 사용
    if model_path is None:
        model_path = str(path_manager.get_model_path())

    log_info("🤖 모델 학습 및 평가를 시작합니다...")
    log_memory_usage("모델 학습 시작")
    
    # 날짜 기반 데이터 분할 (미래 데이터 참조 방지)
    log_info("   📊 날짜 기반 학습/테스트 데이터 분할 중...")
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_by_date(X, y, dates, test_size=0.3)
    log_memory_usage("데이터 분할 완료")
    check_memory_and_cleanup()

    # Optuna 튜닝은 "raw train" 기준으로 시간 기반 CV를 수행해야 함
    # (스케일러/중앙값을 train 전체에 미리 fit하면 CV 단계에서 누수가 발생할 수 있음)
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()

    # ======================================================================
    # ✅ Trial마다 반복되는 공통 작업 캐싱 (gpuStock 스타일)
    # - Expanding CV fold 인덱스 생성 (1회)
    # - Fold별 Train-only 중앙값(imputation) 계산 (1회)
    # - CV에서 스케일링 제거 (RandomForest는 트리 모델이라 불필요)
    # ======================================================================
    log_info("   🧠 [CACHE] 시간 기반 Expanding CV fold/결측치 대체값을 미리 계산합니다...")
    train_dates_cv = pd.to_datetime(train_dates).reset_index(drop=True)
    y_train_cv = y_train.reset_index(drop=True)
    X_train_cv = X_train_raw.reset_index(drop=True)

    fold_indices = expanding_time_series_folds(train_dates_cv, n_splits=3)
    if not fold_indices:
        log_warning("   ⚠️ Expanding CV fold 생성 실패(데이터 기간 부족). Optuna 튜닝을 중단합니다.")
        return

    cv_cache = []
    for (tr_idx, va_idx) in fold_indices:
        X_tr_base = X_train_cv.iloc[tr_idx][features]
        imp = compute_imputation_values_train_only(X_tr_base)
        cv_cache.append((tr_idx, va_idx, imp))
    log_info(f"   ✅ [CACHE] fold 캐시 준비 완료: {len(cv_cache)}개 fold")

    log_info("\n학습 데이터 타겟 분포:\n" + str(y_train.value_counts(normalize=True)))

    log_info("🔍 Optuna를 사용하여 최적 파라미터 탐색... (시간 기반 Expanding CV)")
    log_info(f"   ⚙️ 탐색할 파라미터 조합: {n_iter}개")
    log_info(f"   🔄 교차 검증: 3-fold")
    log_info(f"   💻 사용할 CPU 코어: {n_jobs}")
    
    # Optuna objective 함수 정의 (클로저로 데이터 접근)
    def objective(trial):
        """Optuna objective 함수: 하이퍼파라미터 튜닝을 위한 목적 함수"""
        # 하이퍼파라미터 제안
        model_n_jobs = -1 if n_jobs == -1 else (1 if n_jobs <= 0 else n_jobs)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_categorical('max_depth', max_depth_list),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_samples': trial.suggest_categorical('max_samples', [0.7, 0.8, 0.9, None]),
            'max_features': trial.suggest_float('max_features', 0.4, 1.0),  # 피처 선택 비율 최적화
            'random_state': 42,
            'class_weight': 'balanced',
            'oob_score': False,  # OOB 점수 비활성화로 메모리 절약
            'n_jobs': model_n_jobs,  # RF 내부 멀티스레드 사용 (trial 병렬 대신 모델 병렬 권장)
            'warm_start': False,  # 메모리 절약
            'bootstrap': True
        }

        # 시간 기반 Expanding CV 수행 (누수 방지 + 캐시 재사용 + 스케일링 제거)
        scores = []
        try:
            for (tr_idx, va_idx, imp) in cv_cache:
                X_tr = X_train_cv.iloc[tr_idx][features].copy()
                y_tr_fold = y_train_cv.iloc[tr_idx].copy()
                X_va = X_train_cv.iloc[va_idx][features].copy()
                y_va_fold = y_train_cv.iloc[va_idx].copy()

                X_tr = _sanitize_numeric_frame(X_tr)
                X_va = _sanitize_numeric_frame(X_va)
                X_tr.fillna(imp, inplace=True)
                X_va.fillna(imp, inplace=True)

                model = RandomForestClassifier(**params)
                model.fit(X_tr, y_tr_fold)
                y_va_proba = model.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va_fold, y_va_proba))

                del model, X_tr, X_va, y_tr_fold, y_va_fold, y_va_proba
                gc.collect()

            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            log_warning(f"   ⚠️ Trial {trial.number} 실패: {e}")
            gc.collect()
            return 0.0

    log_memory_usage("하이퍼파라미터 튜닝 시작")
    
    # Optuna study 생성
    study = None
    best_model = None
    best_score = 0.0
    best_params = None
    
    try:
        # TPE 샘플러 사용 (더 효율적인 탐색)
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='random_forest_optimization'
        )
        
        # ✅ 성능/안정성: trial 병렬(optuna n_jobs) 대신 RF 내부 멀티스레드를 사용
        # (trial 병렬은 메모리/CPU oversubscription으로 오히려 느려질 수 있음)
        optuna_n_jobs = 1
        study.optimize(
            objective, 
            n_trials=n_iter,
            n_jobs=optuna_n_jobs,
            show_progress_bar=False  # 로그와 충돌 방지
        )
        
        log_memory_usage("하이퍼파라미터 튜닝 완료")
        
        # 최적 파라미터 추출
        best_params = study.best_params.copy()
        best_score = study.best_value
        
        # 최종 모델 학습은 "최종 전처리(Train-only)" 이후에 수행합니다.
        
    except MemoryError as e:
        log_error(f"하이퍼파라미터 튜닝 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        
        # 더 작은 파라미터 범위로 재시도
        def objective_small(trial):
            model_n_jobs = -1 if n_jobs == -1 else (1 if n_jobs <= 0 else n_jobs)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_categorical('max_depth', max_depth_list[:2]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_samples': trial.suggest_categorical('max_samples', [0.8, 0.9]),
                'max_features': trial.suggest_float('max_features', 0.4, 1.0),  # 피처 선택 비율 최적화
                'random_state': 42,
                'class_weight': 'balanced',
                'oob_score': False,
                'n_jobs': model_n_jobs,
                'warm_start': False,
                'bootstrap': True
            }
            try:
                scores = []
                for (tr_idx, va_idx, imp) in cv_cache:
                    X_tr = X_train_cv.iloc[tr_idx][features].copy()
                    y_tr_fold = y_train_cv.iloc[tr_idx].copy()
                    X_va = X_train_cv.iloc[va_idx][features].copy()
                    y_va_fold = y_train_cv.iloc[va_idx].copy()

                    X_tr = _sanitize_numeric_frame(X_tr)
                    X_va = _sanitize_numeric_frame(X_va)
                    X_tr.fillna(imp, inplace=True)
                    X_va.fillna(imp, inplace=True)

                    model = RandomForestClassifier(**params)
                    model.fit(X_tr, y_tr_fold)
                    y_va_proba = model.predict_proba(X_va)[:, 1]
                    scores.append(roc_auc_score(y_va_fold, y_va_proba))
                    del model, X_tr, X_va, y_tr_fold, y_va_fold, y_va_proba
                    gc.collect()
                return float(np.mean(scores)) if scores else 0.0
            except Exception as e:
                log_warning(f"   ⚠️ Trial {trial.number} 실패: {e}")
                gc.collect()
                return 0.0
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='random_forest_optimization_small'
        )
        study.optimize(
            objective_small, 
            n_trials=max(5, n_iter//2),
            n_jobs=1,  # 단일 코어로 제한
            show_progress_bar=False
        )
        
        best_params = study.best_params.copy()
        best_score = study.best_value
        
        # 최종 모델 학습은 "최종 전처리(Train-only)" 이후에 수행합니다.
    
    except Exception as e:
        log_error(f"하이퍼파라미터 튜닝 중 오류 발생: {e}")
        raise
    
    # 최적 모델이 생성되었는지 확인
    if best_params is None:
        log_error("최적 파라미터 생성에 실패했습니다.")
        raise RuntimeError("최적 파라미터를 생성할 수 없습니다.")
    
    # ===========================
    # 최종 학습/평가용 전처리 (Train-only)
    # ===========================
    log_info("\n   🔧 최종 학습/평가용 Train-only 결측치 대체 및 스케일링 적용 중...")
    X_train_raw = _sanitize_numeric_frame(X_train_raw)
    X_test_raw = _sanitize_numeric_frame(X_test_raw)
    imputation_values = compute_imputation_values_train_only(X_train_raw)
    X_train_raw.fillna(imputation_values, inplace=True)
    X_test_raw.fillna(imputation_values, inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    del X_train_raw, X_test_raw
    gc.collect()
    log_memory_usage("최종 전처리 완료")

    # 최종 모델 학습 (전처리 완료 후)
    log_info("   🔧 최적 파라미터로 최종 모델 학습 중...")
    model_n_jobs = -1 if n_jobs == -1 else (1 if n_jobs <= 0 else n_jobs)
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_samples=best_params.get('max_samples', None),
        max_features=best_params['max_features'],
        random_state=42,
        class_weight='balanced',
        oob_score=False,
        n_jobs=model_n_jobs,
        warm_start=False,
        bootstrap=True
    )
    best_model.fit(X_train_scaled, y_train)
    log_memory_usage("최적 모델 학습 완료")

    log_info("\n--- 최적 파라미터 탐색 결과 ---")
    log_info(f"최고 점수 (ROC-AUC): {best_score:.4f}")
    log_info("최적 파라미터: " + str(best_params))
    
    # OOB 점수는 비활성화되어 있으므로 로그 제거

    log_info("📊 최적 모델로 테스트 데이터 평가 중...")
    log_memory_usage("모델 평가 시작")
    
    try:
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        log_memory_usage("모델 예측 완료")
    except MemoryError as e:
        log_error(f"모델 예측 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        log_memory_usage("재시도 후 모델 예측 완료")

    log_info("\n--- 최종 모델 평가 결과 ---")
    log_info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    log_info("\n분류 보고서 (Classification Report):")
    log_info(classification_report(y_test, y_pred, target_names=['하락(0)', '상승(1)']))

    # 피처 중요도 계산 (3가지 방식)
    log_info("\n📊 피처 중요도 계산 중...")
    
    # 1. 기본 피처 중요도 (모델 내장)
    default_importance = list(zip(features, best_model.feature_importances_))
    default_importance.sort(key=lambda x: x[1], reverse=True)
    log_info("   ✅ 기본 피처 중요도 계산 완료")
    
    # 2. SHAP 중요도 계산 (1000건 계층적 샘플링)
    # X_train_scaled와 y_train은 함수 내부에서 유지되고 있음
    shap_importance = calculate_shap_importance(
        best_model, 
        X_train_scaled, 
        y_train, 
        features, 
        sample_size=1000
    )
    
    # 3. Permutation Importance 계산 (테스트 데이터 사용)
    perm_importance = calculate_permutation_importance(
        best_model, 
        X_test_scaled, 
        y_test, 
        features, 
        n_repeats=5
    )
    
    log_info("💾 모델 저장 중...")
    log_memory_usage("모델 저장 시작")
    
    # 추가 정보 준비
    training_config = {
        'n_iter': n_iter,
        'n_jobs': n_jobs,
        'max_depth_candidates': max_depth_list,
        'cv_folds': 3,
        'test_size': 0.3,
        'scoring': 'roc_auc',
        'search_method': 'Optuna'
    }
    
    optimization_results = {
        'best_score': best_score,
        'best_params': best_params,
        'total_combinations_tested': len(study.trials) if study else n_iter,
        'n_trials_completed': len(study.trials) if study else n_iter
    }
    
    parameter_explanations = {
        'n_estimators': 'RandomForest가 만들 트리의 개수 (100-500)',
        'max_depth': '각 트리의 최대 깊이 (과적합 방지)',
        'min_samples_split': '노드 분할에 필요한 최소 샘플 수',
        'min_samples_leaf': '리프 노드의 최소 샘플 수',
        'max_samples': '각 트리가 사용할 샘플 비율',
        'max_features': '각 분할에서 고려할 최대 피처 비율 (0.4~1.0)',
        'class_weight': '클래스 불균형 처리 방법'
    }
    
    # 피처 중요도 구조 생성 (3가지 방식)
    feature_importances = {
        'default': default_importance
    }
    
    if shap_importance is not None:
        feature_importances['shap'] = shap_importance
        log_info("   ✅ SHAP 중요도 저장 준비 완료")
    else:
        log_warning("   ⚠️ SHAP 중요도가 없어 기본 중요도만 저장합니다.")
    
    if perm_importance is not None:
        feature_importances['permutation'] = perm_importance
        log_info("   ✅ Permutation Importance 저장 준비 완료")
    else:
        log_warning("   ⚠️ Permutation Importance가 없어 기본 중요도만 저장합니다.")
    
    try:
        joblib.dump({
            'model': best_model, 
            'features': features, 
            'scaler': scaler,
            'imputation_values': imputation_values,
            'training_config': training_config,
            'optimization_results': optimization_results,
            'parameter_explanations': parameter_explanations,
            'feature_importances': feature_importances  # 3가지 중요도 포함
        }, model_path, compress=3)  # 압축 저장으로 메모리 절약
        log_info(f"\n✅ 새로운 데이터로 학습된 최적 모델, 스케일러, 중앙값을 '{model_path}' 경로에 저장했습니다.")
        log_memory_usage("모델 저장 완료")
    except MemoryError as e:
        log_error(f"모델 저장 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        joblib.dump({
            'model': best_model, 
            'features': features, 
            'scaler': scaler,
            'imputation_values': imputation_values,
            'training_config': training_config,
            'optimization_results': optimization_results,
            'parameter_explanations': parameter_explanations,
            'feature_importances': feature_importances  # 3가지 중요도 포함
        }, model_path, compress=3)  # 압축 저장으로 메모리 절약
        log_info(f"\n✅ 재시도 후 모델 저장 완료: '{model_path}'")
        log_memory_usage("재시도 후 모델 저장 완료")

def main():
    parser = argparse.ArgumentParser(description="RandomForest 모델 학습 및 하이퍼파라미터 튜닝")
    parser.add_argument('--n_jobs', type=int, default=-1, help='사용할 CPU 코어 수 (-1은 모든 코어 사용)')
    parser.add_argument('--n_iter', type=int, default=10, help='Optuna 최적화 시도 횟수 (trials)')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[10, 20, 30], help='max_depth 후보 리스트')
    parser.add_argument('--years', type=int, default=None, help='학습에 사용할 최근 N년치 데이터 (None이면 전체 데이터, 파일이 없을 때만 적용)')
    args = parser.parse_args()
    
    # ==============================================================================
    # ✨ 핵심 수정: 임시 폴더 생성 및 자동 삭제 로직 추가 ✨
    # ==============================================================================
    # 1. 통일된 경로로 임시 폴더 경로 설정
    temp_folder_path = str(path_manager.get_temp_dir('joblib_temp'))

    try:
        # 2. 임시 폴더 생성 및 환경 변수 설정
        os.makedirs(temp_folder_path, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder_path
        log_info(f"joblib 임시 폴더가 '{temp_folder_path}'로 설정되었습니다.")
        log_memory_usage("프로그램 시작")

        # 3. 메인 학습 로직 실행
        X, y, features, imputation_values, dates = create_training_data(years=args.years)
        if X is not None and dates is not None:
            log_info("🎯 모델 학습을 시작합니다...")
            train_evaluate_and_save_model(X, y, features, imputation_values, dates, args.n_jobs, args.n_iter, args.max_depth)
            log_info("🎉 모델 학습이 성공적으로 완료되었습니다!")
        else:
            log_error("❌ 학습 데이터 생성에 실패하여 모델 학습을 중단합니다.")

    finally:
        # 4. 학습 성공/실패 여부와 관계없이 항상 임시 폴더 삭제
        log_info(f"\n🧹 학습 완료 후 임시 폴더 삭제 중: {temp_folder_path}")
        try:
            path_manager.cleanup_temp_dir('joblib_temp')
            log_info("✅ 임시 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e:
            log_warning(f"⚠️ 임시 폴더를 삭제하는 중 오류가 발생했습니다: {e}")
        
        # 최종 메모리 사용량 로그
        log_memory_usage("프로그램 종료")
        log_info("🏁 모든 작업이 완료되었습니다.")
    # ==============================================================================

if __name__ == '__main__':
    main()