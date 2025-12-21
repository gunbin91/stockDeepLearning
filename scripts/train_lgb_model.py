"""
LightGBM 모델 훈련 스크립트
===========================

이 파일은 주식 상승 예측을 위한 LightGBM 모델을 훈련합니다.
RandomForest와 동일한 학습 데이터를 사용하여 아래 타겟(이진 분류)을 예측합니다:
- 향후 10거래일 동안 -5% 이상 하락 없이, +8% 이상 상승을 한 번이라도 달성 여부

주요 기능:
- 대용량 데이터 처리 및 메모리 최적화
- 하이퍼파라미터 자동 튜닝 (Optuna)
- 교차 검증을 통한 모델 성능 평가
- 훈련된 모델 및 전처리기 저장
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from datetime import datetime, timedelta
import os
import sys
import io
import shutil
import gc
import psutil
import locale
import platform
import optuna
from optuna.samplers import TPESampler
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import lightgbm as lgb

# Windows 환경에서 로케일 설정 (FinanceDataReader 내부 오류 방지)
if platform.system() == 'Windows':
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

# 크로스 플랫폼 인코딩 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# train_model.py에서 공통 함수들 import
from scripts.train_model import (
    get_memory_usage,
    log_memory_usage,
    safe_memory_cleanup,
    check_memory_and_cleanup,
    calculate_shap_importance,
    calculate_permutation_importance,
    create_training_data,
    split_by_date,
    _sanitize_numeric_frame,
    compute_imputation_values_train_only,
    expanding_time_series_folds
)

import data_processor
from path_manager import path_manager
from logger import log_info, log_warning, log_error 

warnings.filterwarnings('ignore', category=FutureWarning)

def perform_undersampling(X, y, random_state=42):
    """
    1:1 언더샘플링 수행 함수
    다수 클래스(0)를 소수 클래스(1)의 개수만큼 무작위로 줄여서 1:1 비율을 맞춥니다.
    """
    try:
        # y가 Series인지 확인하고 아니면 변환
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # X가 DataFrame인지 확인하고 인덱스 리셋 (안전장치)
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
            
        y = y.reset_index(drop=True)
        
        value_counts = y.value_counts()
        if len(value_counts) < 2:
            return X, y
            
        minority_class = value_counts.idxmin()
        majority_class = value_counts.idxmax()
        n_minority = value_counts[minority_class]
        n_majority = value_counts[majority_class]
        
        # 다수 클래스가 소수 클래스보다 많을 때만 수행
        if n_majority > n_minority:
            # 인덱스 추출
            indices = np.arange(len(y))
            minority_indices = indices[y == minority_class]
            majority_indices = indices[y == majority_class]
            
            # 다수 클래스 셔플 및 샘플링
            rng = np.random.RandomState(random_state)
            rng.shuffle(majority_indices)
            selected_majority_indices = majority_indices[:n_minority]
            
            # 합치기
            balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
            balanced_indices.sort()
            
            # 데이터 교체
            X_resampled = X.iloc[balanced_indices].reset_index(drop=True)
            y_resampled = y.iloc[balanced_indices].reset_index(drop=True)
            
            return X_resampled, y_resampled
        
        return X, y
    except Exception as e:
        log_warning(f"언더샘플링 중 오류 발생 (건너뜀): {e}")
        return X, y

def train_evaluate_and_save_lgb_model(X, y, features, imputation_values, dates, n_jobs, n_iter, model_path=None):
    """
    LightGBM 모델 학습 및 저장 함수
    
    Args:
        X: 학습 데이터 (피처)
        y: 학습 타겟
        features: 피처 이름 리스트
        imputation_values: 결측치 대체값
        dates: 날짜 데이터 (Series, datetime 타입)
        n_jobs: 사용할 CPU 코어 수
        n_iter: Optuna 최적화 시도 횟수
        model_path: 모델 저장 경로 (None이면 기본 경로 사용)
    """
    if X is None or y is None or X.empty or y.empty:
        log_error("학습 데이터가 없어 모델링을 건너뜁니다.")
        return
    
    if dates is None or dates.empty:
        log_error("❌ 날짜 데이터가 없어 모델링을 건너뜁니다.")
        return

    # 통일된 경로 사용
    if model_path is None:
        model_path = str(path_manager.get_lgb_model_path())

    log_info("🤖 LightGBM 모델 학습 및 평가를 시작합니다...")
    log_memory_usage("모델 학습 시작")
    
    # 날짜 기반 데이터 분할 (미래 데이터 참조 방지)
    # log_info("   📊 날짜 기반 학습/테스트 데이터 분할 중...")
    # X_train, X_test, y_train, y_test, train_dates, test_dates = split_by_date(X, y, dates, test_size=0.3)
    
    # gpuStock 방식: 별도의 Test 셋을 떼지 않고 전체를 사용하되, 마지막 구간을 검증용으로 사용
    log_info("   📊 전체 데이터를 학습/검증용으로 사용합니다 (별도 Test 셋 분리 없음 - gpuStock 동일)")
    X_train = X
    y_train = y
    train_dates = dates
    
    # X_test, y_test는 없음 (최종 평가 시 마지막 검증 세트 사용)
    # 호환성을 위해 빈 데이터프레임으로 설정하거나 None 처리
    X_test = None
    y_test = None
    
    log_info(f"      🔸 전체 학습 데이터: {len(X_train):,}행 ({train_dates.min().date()} ~ {train_dates.max().date()})")
    
    log_memory_usage("데이터 준비 완료")
    check_memory_and_cleanup()
    
        # Optuna 튜닝은 raw train 기준으로 시간 기반 CV를 수행해야 함
    # (스케일러/중앙값을 train 전체에 미리 fit하면 CV 단계에서 누수가 발생할 수 있음)
    X_train_raw = X_train.copy()
    # X_test_raw = X_test.copy() if X_test is not None else None 

    log_info("\n학습 데이터 타겟 분포:\n" + str(y_train.value_counts(normalize=True)))

    # ======================================================================
    # ✅ Trial마다 반복되는 공통 작업 캐싱 (gpuStock 스타일)
    # - Expanding CV fold 인덱스 생성 (1회)
    # - Fold별 Train-only 중앙값(imputation) 계산 (1회)
    # - (주의) 본 프로젝트는 학습/평가 정합성을 위해 CV에서도 스케일링을 적용합니다.
    # ======================================================================
    log_info("   🧠 [CACHE] 시간 기반 Expanding CV fold/결측치 대체값을 미리 계산합니다...")
    train_dates_cv = pd.to_datetime(train_dates).reset_index(drop=True)
    y_train_cv = y_train.reset_index(drop=True)
    X_train_cv = X_train_raw.reset_index(drop=True)

    fold_indices = expanding_time_series_folds(train_dates_cv, n_splits=3)
    if not fold_indices:
        log_warning("   ⚠️ Expanding CV fold 생성 실패(데이터 기간 부족). Optuna 튜닝을 중단합니다.")
        return

    # 각 fold마다 (train_idx, val_idx, train_only_imputation_dict) 캐시
    cv_cache = []
    for (tr_idx, va_idx) in fold_indices:
        X_tr_base = X_train_cv.iloc[tr_idx][features]
        imp = compute_imputation_values_train_only(X_tr_base)
        cv_cache.append((tr_idx, va_idx, imp))
    log_info(f"   ✅ [CACHE] fold 캐시 준비 완료: {len(cv_cache)}개 fold")

    log_info("🔍 Optuna를 사용하여 최적 파라미터 탐색... (시간 기반 Expanding CV)")
    log_info(f"   ⚙️ 탐색할 파라미터 조합: {n_iter}개")
    log_info(f"   🔄 교차 검증: 3-fold (시간 기반 Expanding)")
    log_info(f"   💻 사용할 CPU 코어: {n_jobs}")
    
    # Optuna objective 함수 정의 (클로저로 데이터 접근)
    def objective(trial):
        """Optuna objective 함수: 하이퍼파라미터 튜닝을 위한 목적 함수"""
        # ✅ gpuStock 스타일: 파라미터 범위 확장 및 디테일 강화
        # 깊이는 10~30, 리프 노드는 31~150까지 확장하여 표현력 강화
        max_depth = trial.suggest_int('max_depth', 10, 30)
        num_leaves = trial.suggest_int('num_leaves', 31, 150)
        
        # LightGBM 하이퍼파라미터 제안 (주식 예측 최적화)
        model_n_jobs = -1 if n_jobs == -1 else (1 if n_jobs <= 0 else n_jobs)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': 10000,
            # ✅ 디테일 강화: 50~200 (적은 샘플로도 패턴 형성 허용)
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 30.0, log=True),
            # ✅ 언더샘플링을 사용하므로 scale_pos_weight는 1.0으로 고정
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'n_jobs': model_n_jobs,  # LGBM 내부 멀티스레드 사용 (trial 병렬 대신 모델 병렬 권장)
            'verbose': -1  # 로그 출력 억제
        }

        # 시간 기반 Expanding CV 수행 (누수 방지 + 캐시 재사용 + early stopping)
        scores = []
        try:
            for i, (tr_idx, va_idx, imp) in enumerate(cv_cache):
                X_tr = X_train_cv.iloc[tr_idx][features].copy()
                y_tr_fold = y_train_cv.iloc[tr_idx].copy()
                X_va = X_train_cv.iloc[va_idx][features].copy()
                y_va_fold = y_train_cv.iloc[va_idx].copy()

                X_tr = _sanitize_numeric_frame(X_tr)
                X_va = _sanitize_numeric_frame(X_va)
                X_tr.fillna(imp, inplace=True)
                X_va.fillna(imp, inplace=True)
                
                # ✅ gpuStock 스타일: 1:1 언더샘플링 적용
                # 학습 데이터에 대해서만 적용 (검증 데이터는 그대로 유지)
                # 모든 Fold의 데이터 크기를 로그로 출력하여 expanding 확인
                if trial.number == 0: # 첫 번째 Trial에서만 로그 출력 (너무 많아지는 것 방지)
                    log_info(f"   📊 [Trial#0-Fold#{i}] 언더샘플링 전: {y_tr_fold.value_counts().to_dict()}")
                
                X_tr_resampled, y_tr_resampled = perform_undersampling(X_tr, y_tr_fold, random_state=42 + tr_idx[0])
                
                if trial.number == 0:
                    log_info(f"   ✅ [Trial#0-Fold#{i}] 언더샘플링 완료: 총 {len(y_tr_resampled):,}행 {y_tr_resampled.value_counts().to_dict()}")

                # ✅ gpuStock 정합성: CV에서도 스케일링을 적용 (최종 학습/평가와 일치)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_resampled)
                X_va_s = scaler.transform(X_va)
                X_tr_s = pd.DataFrame(X_tr_s, columns=features)
                X_va_s = pd.DataFrame(X_va_s, columns=features)

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr_s, y_tr_resampled,
                    eval_set=[(X_va_s, y_va_fold)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                y_va_proba = model.predict_proba(X_va_s)[:, 1]

                # gpuStock 스타일: 검증 데이터에 단일 클래스만 있으면 fold 평가 skip
                if len(np.unique(y_va_fold)) < 2:
                    continue
                scores.append(roc_auc_score(y_va_fold, y_va_proba))

                del model, X_tr, X_va, X_tr_s, X_va_s, scaler, y_tr_fold, y_va_fold, y_va_proba, X_tr_resampled, y_tr_resampled
                gc.collect()

            mean_score = float(np.mean(scores)) if scores else 0.0
            
            # ✅ Trial 완료 로그 (Fold별 점수 포함)
            if len(scores) > 1:
                log_info(f"   🏆 Trial #{trial.number} 완료: 평균 AUC = {mean_score:.4f} (폴드별: {[f'{s:.4f}' for s in scores]})")
            else:
                log_info(f"   🏆 Trial #{trial.number} 완료: AUC = {mean_score:.4f}")
                
            return mean_score
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
            study_name='lightgbm_optimization'
        )
        
        # 하이퍼파라미터 최적화 실행
        # ✅ 성능/안정성: trial 병렬(optuna n_jobs) 대신 LightGBM 내부 멀티스레드를 사용
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
        
        # ===========================
        # 최적 모델 생성 (최종 학습/평가용 전처리: Train-only)
        # ===========================
        log_info("   🔧 최적 파라미터로 최종 모델 학습 중... (Train-only 전처리)")
        
        # gpuStock 방식: 전체 데이터의 마지막 20%를 검증(Valid)용으로 사용하고, 나머지를 학습(Train)용으로 사용
        # (별도의 Test 셋이 없으므로, 이 Valid 셋이 최종 성능 평가 역할도 겸함)
        
        X_train_raw = _sanitize_numeric_frame(X_train_raw)
        
        # 전체 데이터 기준 분할 (80:20)
        val_size = int(len(X_train_raw) * 0.2)
        X_train_fit_raw = X_train_raw.iloc[:-val_size].copy()
        X_val_raw = X_train_raw.iloc[-val_size:].copy()
        y_train_fit = y_train.iloc[:-val_size].copy()
        y_val = y_train.iloc[-val_size:].copy()
        
        log_info(f"   ✂️ 최종 학습 데이터 분할 (gpuStock 방식):")
        log_info(f"      🔸 학습용 (Train): {len(X_train_fit_raw):,}행 (전체의 80%)")
        log_info(f"      🔸 검증용 (Valid): {len(X_val_raw):,}행 (전체의 20% - Early Stopping 및 최종 평가용)")

        # 결측치 대체값은 학습용 데이터(Train)에서만 계산
        imputation_values = compute_imputation_values_train_only(X_train_fit_raw)
        X_train_fit_raw.fillna(imputation_values, inplace=True)
        X_val_raw.fillna(imputation_values, inplace=True)

        # 날짜 기반 분할 검증
        train_fit_max_date = train_dates.iloc[:-val_size].max()
        val_min_date = train_dates.iloc[-val_size:].min()
        if train_fit_max_date >= val_min_date:
            log_warning(f"   ⚠️ Train/Val 분할 경고: Train 최대 날짜({train_fit_max_date}) >= Val 최소 날짜({val_min_date})")
        else:
            log_info(f"   ✅ Train/Val 날짜 분할 확인: Train 최대 날짜 < Val 최소 날짜")

        # 스케일링은 train_fit에서만 fit (val은 transform)
        scaler = StandardScaler()
        X_train_fit_scaled = scaler.fit_transform(X_train_fit_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # LightGBM 피처 이름 경고 방지를 위해 pandas DataFrame으로 변환
        X_train_fit_scaled = pd.DataFrame(X_train_fit_scaled, columns=features)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=features)
        
        # X_test_scaled는 없음 (X_val_scaled가 그 역할을 대신함)
        X_test_scaled = X_val_scaled 
        y_test = y_val

        # 중간 변수 메모리 해제
        del X_train_fit_raw, X_val_raw, X_train_raw
        gc.collect()
        log_memory_usage("최종 전처리 완료")
        
        # n_estimators를 충분히 크게 설정하고 early stopping 사용
        best_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=best_params['num_leaves'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            n_estimators=10000,  # 넉넉하게 설정 (Early Stopping으로 제어)
            min_child_samples=best_params['min_child_samples'],
            min_split_gain=best_params.get('min_split_gain', 0.0),
            min_child_weight=best_params.get('min_child_weight', 1e-3),
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            subsample_freq=best_params.get('subsample_freq', 1),
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            scale_pos_weight=1.0,  # 언더샘플링 적용 (1.0 고정)
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        
        # ✅ 최종 학습 시에도 1:1 언더샘플링 적용
        log_info("   ⚖️ '순수 학습용 데이터'에 1:1 언더샘플링 적용 중...")
        log_info(f"      - 언더샘플링 전: {y_train_fit.value_counts().to_dict()}")
        X_train_fit_resampled, y_train_fit_resampled = perform_undersampling(X_train_fit_scaled, y_train_fit)
        log_info(f"      - 언더샘플링 후: {y_train_fit_resampled.value_counts().to_dict()}")
        log_info(f"   📊 언더샘플링 완료: {len(X_train_fit_scaled):,} -> {len(X_train_fit_resampled):,} 샘플 (최종 학습 데이터)")

        # Early stopping을 사용하여 학습
        best_model.fit(
            X_train_fit_resampled, y_train_fit_resampled,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        log_memory_usage("최적 모델 학습 완료")
        
    except MemoryError as e:
        log_error(f"하이퍼파라미터 튜닝 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        
        # 더 작은 파라미터 범위로 재시도 (메모리 부족 시)
        def objective_small(trial):
            # 주식 데이터에 최적화된 파라미터 범위 설정 (축소 버전)
            max_depth = trial.suggest_int('max_depth', 10, 24)
            num_leaves = trial.suggest_int('num_leaves', 31, 95)
            
            model_n_jobs = -1 if n_jobs == -1 else (1 if n_jobs <= 0 else n_jobs)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
                'n_estimators': 4000,  # early stopping으로 제어
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 200), # 디테일 강화
                'min_split_gain': trial.suggest_float('min_split_gain', 0.05, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
                'scale_pos_weight': 1.0, # 언더샘플링 고정
                'random_state': 42,
                'n_jobs': model_n_jobs,
                'verbose': -1
            }
            try:
                scores = []
                for i, (tr_idx, va_idx, imp) in enumerate(cv_cache):
                    X_tr = X_train_cv.iloc[tr_idx][features].copy()
                    y_tr_fold = y_train_cv.iloc[tr_idx].copy()
                    X_va = X_train_cv.iloc[va_idx][features].copy()
                    y_va_fold = y_train_cv.iloc[va_idx].copy()

                    X_tr = _sanitize_numeric_frame(X_tr)
                    X_va = _sanitize_numeric_frame(X_va)
                    X_tr.fillna(imp, inplace=True)
                    X_va.fillna(imp, inplace=True)
                    
                    # ✅ 1:1 언더샘플링 적용 (재시도 경로)
                    if trial.number == 0:
                        log_info(f"   📊 [Trial#0-Fold#{i} (Small)] 언더샘플링 전: {y_tr_fold.value_counts().to_dict()}")
                        
                    X_tr_resampled, y_tr_resampled = perform_undersampling(X_tr, y_tr_fold, random_state=42 + tr_idx[0])
                    
                    if trial.number == 0:
                        log_info(f"   ✅ [Trial#0-Fold#{i} (Small)] 언더샘플링 완료: 총 {len(y_tr_resampled):,}행 {y_tr_resampled.value_counts().to_dict()}")

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr_resampled)
                    X_va_s = scaler.transform(X_va)
                    X_tr_s = pd.DataFrame(X_tr_s, columns=features)
                    X_va_s = pd.DataFrame(X_va_s, columns=features)

                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_tr_s, y_tr_resampled,
                        eval_set=[(X_va_s, y_va_fold)],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                    y_va_proba = model.predict_proba(X_va_s)[:, 1]
                    if len(np.unique(y_va_fold)) < 2:
                        continue
                    scores.append(roc_auc_score(y_va_fold, y_va_proba))
                    del model, X_tr, X_va, X_tr_s, X_va_s, scaler, y_tr_fold, y_va_fold, y_va_proba, X_tr_resampled, y_tr_resampled
                    gc.collect()
                
                mean_score = float(np.mean(scores)) if scores else 0.0
                if len(scores) > 1:
                    log_info(f"   🏆 Trial #{trial.number} (Small) 완료: 평균 AUC = {mean_score:.4f} (폴드별: {[f'{s:.4f}' for s in scores]})")
                else:
                    log_info(f"   🏆 Trial #{trial.number} (Small) 완료: AUC = {mean_score:.4f}")
                
                return mean_score
            except Exception as e:
                log_warning(f"   ⚠️ Trial {trial.number} 실패: {e}")
                gc.collect()
                return 0.0
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='lightgbm_optimization_small'
        )
        study.optimize(
            objective_small, 
            n_trials=max(5, n_iter//2),
            n_jobs=1,  # 단일 코어로 제한
            show_progress_bar=False
        )
        
        best_params = study.best_params.copy()
        best_score = study.best_value
        
        # (재시도 경로에서도) 최종 학습/평가용 Train-only 전처리
        log_info("   🔧 재시도 경로: Train-only 전처리 적용 중...")
        
        # gpuStock 방식: 전체 데이터의 마지막 20%를 검증(Valid)용으로 사용
        X_train_raw = _sanitize_numeric_frame(X_train_raw)
        
        val_size = int(len(X_train_raw) * 0.2)
        X_train_fit_raw = X_train_raw.iloc[:-val_size].copy()
        X_val_raw = X_train_raw.iloc[-val_size:].copy()
        y_train_fit = y_train.iloc[:-val_size].copy()
        y_val = y_train.iloc[-val_size:].copy()

        # 결측치 대체값은 학습용 데이터(Train)에서만 계산
        imputation_values = compute_imputation_values_train_only(X_train_fit_raw)
        X_train_fit_raw.fillna(imputation_values, inplace=True)
        X_val_raw.fillna(imputation_values, inplace=True)

        train_fit_max_date = train_dates.iloc[:-val_size].max()
        val_min_date = train_dates.iloc[-val_size:].min()
        if train_fit_max_date >= val_min_date:
            log_warning(f"   ⚠️ Train/Val 분할 경고: Train 최대 날짜({train_fit_max_date}) >= Val 최소 날짜({val_min_date})")
        else:
            log_info(f"   ✅ Train/Val 날짜 분할 확인: Train 최대 날짜 < Val 최소 날짜")

        scaler = StandardScaler()
        X_train_fit_scaled = scaler.fit_transform(X_train_fit_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        X_train_fit_scaled = pd.DataFrame(X_train_fit_scaled, columns=features)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=features)
        
        # X_test_scaled는 없음 (X_val_scaled가 그 역할을 대신함)
        X_test_scaled = X_val_scaled 
        y_test = y_val

        del X_train_fit_raw, X_val_raw, X_train_raw
        gc.collect()
        
        # n_estimators를 충분히 크게 설정하고 early stopping 사용
        best_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=best_params['num_leaves'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            n_estimators=5000,  # Early Stopping으로 제어
            min_child_samples=best_params['min_child_samples'],
            min_split_gain=best_params.get('min_split_gain', 0.0),
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            subsample_freq=best_params.get('subsample_freq', 1),
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            scale_pos_weight=1.0,  # 언더샘플링 적용 (1.0 고정)
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        
        # ✅ 최종 학습 시에도 1:1 언더샘플링 적용
        log_info("   ⚖️ (재시도) 최종 학습 데이터에 1:1 언더샘플링 적용 중...")
        log_info(f"      - 언더샘플링 전: {y_train_fit.value_counts().to_dict()}")
        X_train_fit_resampled, y_train_fit_resampled = perform_undersampling(X_train_fit_scaled, y_train_fit)
        log_info(f"      - 언더샘플링 후: {y_train_fit_resampled.value_counts().to_dict()}")
        log_info(f"   📊 언더샘플링 완료: {len(X_train_fit_scaled):,} -> {len(X_train_fit_resampled):,} 샘플 (정상)")

        # Early stopping을 사용하여 학습
        best_model.fit(
            X_train_fit_resampled, y_train_fit_resampled,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        log_memory_usage("재시도 후 하이퍼파라미터 튜닝 완료")
    
    except Exception as e:
        log_error(f"하이퍼파라미터 튜닝 중 오류 발생: {e}")
        raise
    
    # 최적 모델이 생성되었는지 확인
    if best_model is None or best_params is None:
        log_error("최적 모델 생성에 실패했습니다.")
        raise RuntimeError("최적 모델을 생성할 수 없습니다.")
    
    log_info("\n--- 최적 파라미터 탐색 결과 ---")
    log_info(f"최고 점수 (ROC-AUC): {best_score:.4f}")
    log_info("최적 파라미터: " + str(best_params))

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
    shap_importance = calculate_shap_importance(
        best_model, 
        X_train_fit_scaled, 
        y_train_fit, 
        features, 
        sample_size=1000
    )
    
    # 3. Permutation Importance 계산 (테스트 데이터 사용, 속도를 위해 최대 5000개 샘플링)
    log_info("\n📊 Permutation Importance 계산을 위해 데이터 샘플링 중...")
    try:
        sample_size = min(5000, len(X_test_scaled))
        if len(X_test_scaled) > sample_size:
            # 인덱스를 랜덤으로 선택
            indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
            
            # DataFrame인 경우 iloc 사용, numpy array인 경우 배열 인덱싱
            if isinstance(X_test_scaled, pd.DataFrame):
                X_perm_sample = X_test_scaled.iloc[indices]
            else:
                X_perm_sample = X_test_scaled[indices]
                
            if isinstance(y_test, pd.Series):
                y_perm_sample = y_test.iloc[indices]
            else:
                y_perm_sample = y_test[indices]
            
            log_info(f"   ✅ 샘플링 완료: {len(X_test_scaled)} -> {len(X_perm_sample)} 건")
        else:
            X_perm_sample = X_test_scaled
            y_perm_sample = y_test
            log_info(f"   ℹ️ 데이터가 {sample_size}건 이하이므로 전체 사용: {len(X_perm_sample)} 건")

        perm_importance = calculate_permutation_importance(
            best_model, 
            X_perm_sample, 
            y_perm_sample, 
            features, 
            n_repeats=5
        )
    except Exception as e:
        log_warning(f"   ⚠️ Permutation Importance 샘플링/계산 중 오류: {e}")
        perm_importance = None
    
    log_info("💾 모델 저장 중...")
    log_memory_usage("모델 저장 시작")
    
    # 추가 정보 준비
    training_config = {
        'n_iter': n_iter,
        'n_jobs': n_jobs,
        'cv_folds': 3,
        'test_size': 0.3,
        'scoring': 'roc_auc',
        'search_method': 'Optuna',
        'model_type': 'LightGBM'
    }
    
    optimization_results = {
        'best_score': best_score,
        'best_params': best_params,
        'total_combinations_tested': len(study.trials) if study else n_iter,
        'n_trials_completed': len(study.trials) if study else n_iter
    }
    
    parameter_explanations = {
        'num_leaves': '리프 노드의 최대 개수 (31-150, 표현력 강화)',
        'max_depth': '트리의 최대 깊이 (10-30, 깊은 트리 허용)',
        'learning_rate': '학습 속도 (0.005-0.05, early stopping 필수)',
        'n_estimators': '부스팅 반복 횟수 (5000 이상, Early Stopping으로 제어)',
        'min_child_samples': '리프 노드의 최소 샘플 수 (50-200, 디테일한 패턴 학습)',
        'min_split_gain': '분할 최소 이득 (0.0-1.0)',
        'subsample': '각 트리가 사용할 샘플 비율 (0.6-1.0)',
        'colsample_bytree': '각 트리가 사용할 피처 비율 (0.5-1.0)',
        'subsample_freq': '배깅을 수행하는 주기 (1-7)',
        'reg_alpha': 'L1 정규화 계수 (1e-3-10.0)',
        'reg_lambda': 'L2 정규화 계수 (1e-3-30.0)',
        'scale_pos_weight': '양성 클래스 가중치 (1.0 고정, 언더샘플링으로 비율 맞춤)'
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
        log_info(f"\n✅ 새로운 데이터로 학습된 최적 LightGBM 모델, 스케일러, 중앙값을 '{model_path}' 경로에 저장했습니다.")
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
    parser = argparse.ArgumentParser(description="LightGBM 모델 학습 및 하이퍼파라미터 튜닝")
    parser.add_argument('--n_jobs', type=int, default=-1, help='사용할 CPU 코어 수 (-1은 모든 코어 사용)')
    parser.add_argument('--n_iter', type=int, default=10, help='Optuna 최적화 시도 횟수 (trials)')
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

    # gpuStock과 동일한 피처 리스트 (하드코딩으로 명시적 동기화)
    gpu_stock_features = [
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
    
    # 3. 메인 학습 로직 실행 (공통 create_training_data 함수 사용)
    # create_training_data가 반환하는 features 대신 gpu_stock_features를 사용하도록 덮어쓰기 고려
    X, y, features, imputation_values, dates = create_training_data(years=args.years)
    
    if X is not None:
        # gpuStock 피처만 선택 (존재하는 것만)
        available_gpu_features = [f for f in gpu_stock_features if f in X.columns]
        
        if len(available_gpu_features) < len(gpu_stock_features):
            missing = set(gpu_stock_features) - set(available_gpu_features)
            log_warning(f"⚠️ gpuStock 피처 중 일부가 누락되었습니다: {missing}")
        
        # 피처 리스트 교체 및 데이터 필터링
        features = available_gpu_features
        X = X[features]
        log_info(f"✅ gpuStock 동기화: {len(features)}개 피처로 학습을 진행합니다.")

    if X is not None and dates is not None:
            log_info("🎯 LightGBM 모델 학습을 시작합니다...")
            train_evaluate_and_save_lgb_model(X, y, features, imputation_values, dates, args.n_jobs, args.n_iter)
            log_info("🎉 LightGBM 모델 학습이 성공적으로 완료되었습니다!")
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

