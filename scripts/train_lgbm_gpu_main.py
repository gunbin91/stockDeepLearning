# LightGBM GPU Accelerated Stock Predictor Training Script
#
# Features:
# - Just-in-Time (JIT) data loading to minimize memory usage.
# - Interactive data management (prompts for regeneration).
# - GPU acceleration using LightGBM.
# - Hyperparameter optimization using Optuna.
# - Full dataset usage (no undersampling) with scale_pos_weight for class imbalance.
# - Early Stopping to prevent overfitting.
# - Shared data path with RF/CatBoost: ~/stock_data/processed_feather

import os
import sys
os.environ['PYTHONWARNINGS'] = 'ignore'
import argparse
import shutil
import glob
import subprocess
from datetime import datetime, timedelta
from bisect import bisect_left
import gc
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
_orig_stderr = sys.stderr
if not getattr(_orig_stderr, "_yf_warning_filtered", False):
    class _YFinanceFilteredStderr:
        def __init__(self, original):
            self.original = original
            self._yf_warning_filtered = True
        def write(self, text):
            if text and ("Pandas4Warning" in text or "Timestamp.utcnow" in text or
                         ("deprecated" in text and ("yfinance" in text or "Timestamp" in text)) or
                         ("will be removed" in text and "Timestamp" in text)):
                return
            self.original.write(text)
        def flush(self):
            self.original.flush()
        def __getattr__(self, name):
            return getattr(self.original, name)
    sys.stderr = _YFinanceFilteredStderr(_orig_stderr)

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
    except Exception as e:
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
    except Exception as e:
        # 패치 적용 실패해도 계속 진행
        pass

# 서드파티 라이브러리
import pandas as pd
# 호환성 패치 적용 (cuDF/cuML import 전에 실행)
apply_pandas_compatibility_patch()
apply_sklearn_compatibility_patch()
import numpy as np
import joblib
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import psutil

# LightGBM은 lazy import로 처리 (OpenMP 충돌 방지)

# SHAP 라이브러리 (피처 중요도 계산용)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP 라이브러리가 설치되지 않았습니다. 피처 중요도 계산을 건너뜁니다.")

# 내부 모듈
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import log_info, log_warning, log_error, log_critical
import data_processor
from path_manager import path_manager

# ============================================================
# Horizon / Purge(embargo) 설정
# - 본 프로젝트의 타겟은 "향후 10거래일"을 참조하여 생성됩니다.
# - 시계열 Fold 분할 시 Train/Val 경계 근처 샘플의 라벨이
#   Val 구간의 가격을 참조할 수 있어 평가 누수가 생길 수 있으므로,
#   기본적으로 horizon만큼(purge_trading_days) 갭을 둡니다.
# ============================================================
DEFAULT_HORIZON_TRADING_DAYS = 10

# Optuna TPE Sampler의 seed 고정을 위한 설정
optuna.samplers.TPESampler.init_rng = lambda self: np.random.RandomState()

# --- 메모리 관리 유틸리티 ---

def convert_cudf_to_pandas_for_joblib(obj):
    """
    joblib.dump를 위해 cuDF 객체를 pandas로 변환하는 헬퍼 함수
    - cuDF DataFrame/Series를 pandas로 변환
    - 딕셔너리의 값들도 재귀적으로 변환
    - 리스트/튜플의 요소들도 재귀적으로 변환
    """
    try:
        # 호환성 패치 적용 (import cudf 전에 실행)
        apply_pandas_compatibility_patch()
        import cudf
        if isinstance(obj, cudf.DataFrame):
            return obj.to_pandas()
        elif isinstance(obj, cudf.Series):
            return obj.to_pandas()
        elif isinstance(obj, dict):
            return {k: convert_cudf_to_pandas_for_joblib(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            converted = [convert_cudf_to_pandas_for_joblib(item) for item in obj]
            return tuple(converted) if isinstance(obj, tuple) else converted
        else:
            return obj
    except:
        # cudf가 없거나 변환 실패 시 원본 반환
        return obj

def get_memory_usage():
    """현재 CPU 메모리 사용량을 MB 단위로 반환합니다."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def log_memory_usage(stage_name, additional_info=None):
    """CPU 메모리 사용량을 로그로 출력합니다."""
    cpu_mem = get_memory_usage()
    info_str = f"   💾 {stage_name} - CPU: {cpu_mem:.1f} MB"
    if additional_info:
        info_str += f" | {additional_info}"
    log_info(info_str)

# --- 데이터 준비 함수 ---

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

    # RF와 동일한 피처 목록 (실제 계산되는 피처만 포함)
    features = [
        'log_mktcap',
        '52주_신고가_비율',
        'ADX_14',
        'disparity_120',
        'disparity_240',
        'disparity_20',
        'IXIC_disparity_20',
        'Trend_Pullback_Score',
        'Position_Range_60',
        'MA20_Slope',
        'MA120_Slope',
        'MA240_Slope',
        'IXIC_MA20_Slope',
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
        '등락율(5D)',
    ]

    try:
        # data_processor를 통해 모든 피처가 계산된 거대 데이터프레임 생성
        full_df = data_processor.get_preprocessed_data(actual_start_date, end_date, skip_factor_scores=True)

        log_memory_usage("전체 데이터 로딩 완료")

        if full_df is None or full_df.empty:
            log_error("데이터 전처리 중 오류가 발생하여 데이터를 생성할 수 없습니다.")
            return False

        log_info(f"   📊 전처리된 원본 데이터: {len(full_df):,}행")

        # target 필터링
        log_info("   🔍 target 필터링 중...")
        full_df = full_df[full_df['target'].notna()].copy()
        log_info(f"   📊 target 필터링 후 데이터: {len(full_df):,}행")
        if full_df.empty:
            log_error("target 필터링 후 데이터가 없습니다.")
            return False

        # 날짜 컬럼 확인
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

                # 숫자형 피처만 선택
                numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
                if 'target' in numeric_cols:
                    numeric_cols.remove('target')
                
                if not numeric_cols:
                    log_warning(f"   ⚠️ 종목 {ticker}에 숫자형 피처가 없습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    continue
                
                X_all = ticker_df[numeric_cols].astype(np.float32)
                
                if 'target' not in ticker_df.columns:
                    log_warning(f"   ⚠️ 종목 {ticker}에 target 컬럼이 없습니다. 건너뜁니다.")
                    failed_count += 1
                    failed_tickers.append(ticker)
                    del X_all
                    continue
                
                y = ticker_df['target'].astype(np.int32)
                
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
                if '종목코드' in ticker_df.columns:
                    preprocessed_df['종목코드'] = ticker_df['종목코드'].values
                else:
                    preprocessed_df['종목코드'] = ticker
                if '시가총액' in ticker_df.columns:
                    preprocessed_df['시가총액'] = ticker_df['시가총액'].values
                if '종목명' in ticker_df.columns:
                    preprocessed_df['종목명'] = ticker_df['종목명'].values

                file_path = os.path.join(data_path, f"{ticker}.feather")

                try:
                    preprocessed_df.to_feather(file_path)
                    saved_count += 1
                except Exception as save_error:
                    log_error(f"   ❌ 종목 {ticker} feather 파일 저장 실패: {save_error}")
                    failed_count += 1
                    failed_tickers.append(ticker)
                
                del ticker_df, X_all, y, preprocessed_df

                if (i + 1) % 200 == 0:
                    log_info(f"      ... {i+1}/{len(tickers)} 개 종목 저장 완료 (성공: {saved_count}, 실패: {failed_count})")
                    
            except Exception as e:
                log_error(f"   ❌ 종목 {ticker} 처리 중 오류 발생: {e}")
                failed_count += 1
                failed_tickers.append(ticker)
                continue

        log_info(f"   ✅ 총 {saved_count}개 종목 저장 완료 (실패: {failed_count}개)")
        if failed_tickers:
            log_warning(f"   ⚠️ 실패한 종목 목록 (최대 10개): {failed_tickers[:10]}")

        log_memory_usage("데이터 전처리 완료")
        return True

    except Exception as e:
        log_error(f"❌ 데이터 전처리 중 치명적 오류 발생: {e}")
        import traceback
        log_error(f"상세 오류:\n{traceback.format_exc()}")
        return False

# --- 데이터 로딩 함수 ---

def load_data_period(file_paths, features, start_date, end_date, imputation_map=None, batch_size=50, max_workers=4):
    """
    지정된 기간의 데이터를 로드하고 결측치를 처리합니다.
    """
    log_info(f"   📂 데이터 로딩 중... (기간: {start_date} ~ {end_date})")
    
    all_data = []
    failed_tickers = []
    
    for i, file_path in enumerate(file_paths):
        try:
            ticker_df = pd.read_feather(file_path)
            
            # 날짜 필터링
            if 'date' in ticker_df.columns:
                ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                # ✅ 시계열 정합성: end_date는 "exclusive"로 통일 (Train/Val 경계 날짜 겹침 방지)
                mask = (ticker_df['date'] >= start_date) & (ticker_df['date'] < end_date)
                ticker_df = ticker_df[mask].copy()
            
            if ticker_df.empty:
                continue
            
            # 피처 필터링
            available_features = [f for f in features if f in ticker_df.columns]
            if 'target' in ticker_df.columns:
                available_features.append('target')
            if 'date' in ticker_df.columns:
                available_features.append('date')
            
            ticker_df = ticker_df[available_features]
            
            # 결측치 처리 (imputation_map 사용)
            if imputation_map is not None:
                for col in ticker_df.select_dtypes(include=[np.number]).columns:
                    if col in imputation_map and col != 'target':
                        ticker_df[col] = ticker_df[col].fillna(imputation_map[col])
            
            all_data.append(ticker_df)
            
        except Exception as e:
            log_warning(f"   ⚠️ {os.path.basename(file_path)} 로드 실패: {e}")
            failed_tickers.append(os.path.basename(file_path))
            continue
    
    if not all_data:
        log_error("로드된 데이터가 없습니다.")
        return None, None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    log_info(f"   ✅ 데이터 로딩 완료: {len(combined_df):,}행")
    
    return combined_df, None, None

def _build_trading_calendar_from_files(file_paths, max_files=30, min_unique_dates=2000):
    """
    파일들에서 'date' 컬럼만 읽어 거래일 캘린더(정렬된 unique normalize 날짜)를 구성합니다.
    - 최소 변경/성능을 위해 일부 파일만 샘플링합니다.
    - 주식 시장 캘린더가 종목 간 동일하다는 가정 하에, 충분한 개수의 날짜가 확보되면 조기 종료합니다.
    """
    dates_set = set()
    if not file_paths:
        return []
    for fp in file_paths[:max_files]:
        try:
            df = pd.read_feather(fp, columns=['date'])
            if 'date' not in df.columns:
                continue
            d = pd.to_datetime(df['date'], errors='coerce').dropna().dt.normalize()
            dates_set.update(d.unique().tolist())
            if len(dates_set) >= min_unique_dates:
                break
        except Exception:
            continue
    if not dates_set:
        return []
    # Timestamp 정렬
    return sorted(pd.to_datetime(list(dates_set)).to_list())

def _apply_purge_to_fold_ranges(fold_ranges, trading_dates, purge_trading_days):
    """
    fold_ranges(list[dict])의 train_end를 Val 시작 직전 purge_trading_days 거래일만큼 앞당겨
    Train/Val 경계 누수를 줄입니다.
    - 날짜 필터가 [start, end) (end exclusive)라는 전제에서 동작합니다.
    """
    if not fold_ranges or not trading_dates or purge_trading_days <= 0:
        return fold_ranges

    td = [pd.Timestamp(x).normalize() for x in trading_dates]
    for fr in fold_ranges:
        try:
            val_start = pd.Timestamp(fr['val_start']).normalize()
            # val_start가 캘린더에 없으면 삽입 위치를 사용 (가장 가까운 이후 거래일로 정렬)
            pos = bisect_left(td, val_start)
            if pos <= 0:
                continue
            cutoff_pos = pos - int(purge_trading_days)
            if cutoff_pos <= 0:
                continue
            cutoff_date = td[cutoff_pos]
            # Train은 [train_start, cutoff_date) 로 축소
            if pd.Timestamp(fr['train_start']).normalize() < cutoff_date < val_start:
                fr['train_end'] = cutoff_date
        except Exception:
            continue
    return fold_ranges

def get_date_range_from_files(file_paths):
    """
    파일 목록에서 전체 날짜 범위를 확인합니다.
    """
    min_date = None
    max_date = None
    
    # 샘플링하여 확인 (처음, 중간, 끝)
    sample_paths = file_paths[:5]
    if len(file_paths) > 10:
        sample_paths.extend(file_paths[len(file_paths)//2 : len(file_paths)//2 + 5])
        sample_paths.extend(file_paths[-5:])
    
    for file_path in sample_paths:
        try:
            df = pd.read_feather(file_path, columns=['date'])
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                file_min = dates.min()
                file_max = dates.max()
                
                if min_date is None or file_min < min_date:
                    min_date = file_min
                if max_date is None or file_max > max_date:
                    max_date = file_max
        except Exception:
            continue
            
    if min_date is None or max_date is None:
        # 샘플링 실패 시 전체 검색 (느리지만 정확함)
        log_info("   ⚠️ 샘플링으로 날짜 범위를 확인할 수 없어 전체 파일을 검색합니다...")
        for file_path in file_paths:
            try:
                df = pd.read_feather(file_path, columns=['date'])
                if 'date' in df.columns:
                    dates = pd.to_datetime(df['date'])
                    file_min = dates.min()
                    file_max = dates.max()
                    
                    if min_date is None or file_min < min_date:
                        min_date = file_min
                    if max_date is None or file_max > max_date:
                        max_date = file_max
            except:
                continue
    
    return min_date, max_date

def calculate_expanding_fold_ranges(file_paths, warmup_days=250, val_period_days=365, n_folds=3):
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
    
    if min_date is None or max_date is None:
        log_error("날짜 범위를 확인할 수 없습니다.")
        return []

    # 실제 학습 시작일: 웜업 기간 제외
    actual_start_date = min_date + timedelta(days=warmup_days)
    actual_end_date = max_date
    
    log_info(f"   📅 전체 데이터 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    log_info(f"   📅 실제 학습 기간 (웜업 제외): {actual_start_date.strftime('%Y-%m-%d')} ~ {actual_end_date.strftime('%Y-%m-%d')}")
    
    # Expanding Window 방식으로 Fold 범위 계산
    # 검증 기간을 고정하고, Train은 누적되는 방식
    fold_ranges = []
    
    # 마지막 n_folds개 검증 기간을 역순으로 계산
    for fold_idx in range(n_folds):
        # 검증 기간 계산 (역순)
        # Fold 0: 가장 과거의 검증 기간
        # Fold 2: 가장 최근의 검증 기간 (미래와 가까움)
        # RF 코드와 동일하게: 마지막 3년치를 1년씩 검증
        
        # 마지막 날짜로부터 (n_folds - fold_idx) * val_period_days 만큼 뒤로 간 날짜가 val_end가 됨
        # 예: n_folds=3, fold_idx=0 (첫번째) -> 뒤에서 3번째 기간
        # 예: n_folds=3, fold_idx=2 (마지막) -> 뒤에서 1번째 기간 (가장 최근)
        
        # RF 로직:
        # Fold 0: val_end = max - 2*365
        # Fold 1: val_end = max - 1*365
        # Fold 2: val_end = max
        
        # 이렇게 하려면:
        end_offset = (n_folds - 1 - fold_idx) * val_period_days
        val_end = actual_end_date - timedelta(days=end_offset)
        val_start = val_end - timedelta(days=val_period_days)
        
        # Train 기간: actual_start_date ~ val_start (항상 처음부터 시작 = Expanding)
        train_start = actual_start_date
        train_end = val_start
        
        # 검증 기간이 실제 학습 기간을 벗어나지 않도록 조정
        if val_start < actual_start_date:
            log_warning(f"   ⚠️ Fold #{fold_idx+1} 검증 기간이 학습 시작일보다 이전입니다. 조정합니다.")
            val_start = actual_start_date
            val_end = val_start + timedelta(days=val_period_days)
            train_end = val_start
        
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
        log_info(f"      Val:   {val_start.strftime('%Y-%m-%d')} ~ {val_end.strftime('%Y-%m-%d')}")
    
    return fold_ranges

# --- GPU 사용 가능 여부 확인 함수 ---

def check_gpu_availability():
    """
    LightGBM GPU 사용 가능 여부를 확인합니다.
    GPU가 사용 불가능하면 CPU로 자동 전환합니다.
    """
    import lightgbm as lgb
    
    # 전역 변수로 캐싱 (한 번만 확인)
    if hasattr(check_gpu_availability, '_gpu_available'):
        return check_gpu_availability._gpu_available, check_gpu_availability._device
    
    try:
        # 작은 더미 데이터로 GPU 사용 가능 여부 확인
        dummy_X = np.random.rand(100, 10).astype(np.float32)
        dummy_y = np.random.randint(0, 2, 100).astype(np.int32)
        dummy_data = lgb.Dataset(dummy_X, label=dummy_y)
        
        test_params = {
            'objective': 'binary',
            'metric': 'auc',
            'device': 'gpu',
            'num_threads': 0,
            'verbosity': -1,
        }
        
        # GPU 학습 시도
        test_model = lgb.train(
            test_params,
            dummy_data,
            num_boost_round=1,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # 성공하면 GPU 사용 가능
        check_gpu_availability._gpu_available = True
        check_gpu_availability._device = 'gpu'
        log_info("   ✅ GPU 사용 가능: LightGBM이 GPU로 학습합니다.")
        return True, 'gpu'
        
    except Exception as e:
        # GPU 실패 시 CPU로 전환
        check_gpu_availability._gpu_available = False
        check_gpu_availability._device = 'cpu'
        log_warning(f"   ⚠️ GPU 사용 불가능: {str(e)}")
        log_info("   🔄 CPU로 자동 전환합니다.")
        return False, 'cpu'

# --- Optuna Objective 함수 ---

def objective(trial, fold_data_cache, features):
    """
    Optuna를 위한 objective 함수 (LightGBM).
    캐시된 데이터를 사용하여 I/O 병목을 제거합니다.
    """
    # Lazy import to avoid OpenMP conflicts
    import lightgbm as lgb
    
    trial_start_time = datetime.now()
    log_info(f"\n{'='*60}")
    log_info(f"🚀 Optuna Trial #{trial.number} 시작")
    
    # GPU 사용 가능 여부 확인
    gpu_available, device = check_gpu_availability()
    
    # 하이퍼파라미터 제안
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': device,  # GPU 또는 CPU (자동 선택)
        'num_threads': 0 if device == 'gpu' else -1,  # GPU: 0, CPU: -1 (모든 스레드 사용)
        'verbosity': -1,
        'seed': 42,
        'deterministic': True,
        # 핵심 파라미터
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 300),
        # 정규화
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 30.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        # 샘플링
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # 불균형 처리 (언더샘플링 적용으로 scale_pos_weight 제거 또는 1.0 고정)
        # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0, log=True),
        'scale_pos_weight': 1.0,  # 1:1 샘플링이므로 가중치 불필요
        # 기타
        'max_bin': 255,
        'n_estimators': 10000,  # 고정 (Early Stopping으로 제어)
    }
    
    log_info(f"   📋 파라미터: learning_rate={params['learning_rate']:.6f}, num_leaves={params['num_leaves']}, "
             f"max_depth={params['max_depth']}, min_child_samples={params['min_child_samples']}")
    
    fold_scores = []
    
    # 각 Fold 처리 (캐시 사용)
    for fold_idx, (X_train, y_train, X_val, y_val) in fold_data_cache.items():
        try:
            # log_info(f"   📊 Fold #{fold_idx+1} 학습 시작...")
            
            # --- 1:1 언더샘플링 (RF와 동일하게 적용) ---
            # 다수 클래스(0)를 소수 클래스(1) 개수에 맞춰 무작위로 줄임
            y_train_series = pd.Series(y_train)
            value_counts = y_train_series.value_counts()
            
            if len(value_counts) >= 2:
                minority_class = value_counts.idxmin()
                majority_class = value_counts.idxmax()
                n_minority = value_counts[minority_class]
                n_majority = value_counts[majority_class]
                
                # [로그 추가] 첫 번째 Trial의 첫 번째 Fold에서만 상세 로그 출력
                if trial.number == 0 and fold_idx == 0:
                     log_info(f"   📊 [Trial#0-Fold#0] 언더샘플링 전: 소수({minority_class})={n_minority:,}, 다수({majority_class})={n_majority:,}")

                if n_majority > n_minority:
                    # 인덱스 추출
                    indices = np.arange(len(y_train))
                    minority_indices = indices[y_train == minority_class]
                    majority_indices = indices[y_train == majority_class]
                    
                    # 다수 클래스 셔플 및 샘플링 (1:1 비율)
                    rng_sampler = np.random.RandomState(42 + fold_idx) 
                    rng_sampler.shuffle(majority_indices)
                    selected_majority_indices = majority_indices[:n_minority]
                    
                    # 합치기
                    balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
                    balanced_indices.sort()
                    
                    # 데이터 교체 (DataFrame이므로 iloc 사용, y는 numpy array)
                    X_train_resampled = X_train.iloc[balanced_indices].reset_index(drop=True)
                    y_train_resampled = y_train[balanced_indices]

                    if trial.number == 0 and fold_idx == 0:
                        log_info(f"   ✅ [Trial#0-Fold#0] 1:1 언더샘플링 완료: 총 {len(y_train_resampled):,}행 (소수: {n_minority:,}, 다수: {len(selected_majority_indices):,})")
                else:
                    X_train_resampled = X_train
                    y_train_resampled = y_train
                    if trial.number == 0 and fold_idx == 0:
                        log_info(f"   ℹ️ [Trial#0-Fold#0] 클래스 불균형이 심하지 않아 샘플링 건너뜀.")
            else:
                X_train_resampled = X_train
                y_train_resampled = y_train

            # 트리 기반 모델이므로 StandardScaler 생략 (메모리 및 속도 최적화)
            X_train_scaled = X_train_resampled
            X_val_scaled = X_val
            
            # LightGBM Dataset 생성
            train_data = lgb.Dataset(X_train_scaled, label=y_train_resampled)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            # 모델 학습 (Early Stopping 포함)
            # GPU 실패 시 자동으로 CPU로 전환
            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params['n_estimators'],
                    valid_sets=[val_data],
                    valid_names=['val'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)  # 로그 비활성화
                    ]
                )
            except Exception as gpu_error:
                # GPU 실패 시 CPU로 전환
                if 'gpu' in str(gpu_error).lower() or 'opencl' in str(gpu_error).lower() or 'device' in str(gpu_error).lower():
                    # log_warning(f"   ⚠️ GPU 학습 실패: {str(gpu_error)}")
                    # log_info("   🔄 CPU로 자동 전환하여 재시도합니다...")
                    # CPU 파라미터로 변경
                    params_cpu = params.copy()
                    params_cpu['device'] = 'cpu'
                    params_cpu['num_threads'] = -1  # 모든 CPU 스레드 사용
                    # GPU 사용 불가능 플래그 설정
                    check_gpu_availability._gpu_available = False
                    check_gpu_availability._device = 'cpu'
                    # CPU로 재시도
                    model = lgb.train(
                        params_cpu,
                        train_data,
                        num_boost_round=params_cpu['n_estimators'],
                        valid_sets=[val_data],
                        valid_names=['val'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                else:
                    # GPU 관련이 아닌 다른 에러는 그대로 raise
                    raise
            
            # 예측 및 평가
            y_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            
            # 검증 데이터에 두 클래스가 모두 있는지 확인
            unique_classes = np.unique(y_val)
            if len(unique_classes) < 2:
                # log_warning(f"   ⚠️ Fold #{fold_idx+1}: 검증 데이터에 한 클래스만 존재합니다. 건너뜁니다.")
                continue  # 이 폴드는 건너뜀
            
            auc_score = roc_auc_score(y_val, y_pred)
            fold_scores.append(auc_score)
            
            # 메모리 정리 (스케일링된 데이터 등)
            del X_train_scaled, X_val_scaled, train_data, val_data, model
            # gc.collect() # 너무 잦은 GC는 성능 저하
            
        except Exception as e:
            log_error(f"   ❌ Fold #{fold_idx+1} 처리 중 오류: {e}")
            continue
    
    if not fold_scores:
        log_warning("   ⚠️ 모든 Fold에서 실패하거나 평가할 수 없었습니다. 0.0을 반환합니다.")
        return 0.0
    
    mean_score = np.mean(fold_scores)
    # 폴드별 점수 요약 로깅
    if len(fold_scores) > 1:
        log_info(f"   🏆 Trial #{trial.number} 완료: 평균 AUC = {mean_score:.4f} (폴드별: {[f'{s:.4f}' for s in fold_scores]})")
    else:
        log_info(f"   🏆 Trial #{trial.number} 완료: AUC = {mean_score:.4f}")
    
    return mean_score

# --- 최종 모델 학습 함수 ---

def train_final_model(fold_ranges, file_paths, features, best_params, best_score=None, optimization_results=None, training_config=None):
    """
    최적 파라미터로 최종 모델을 학습합니다.
    """
    # Lazy import to avoid OpenMP conflicts
    import lightgbm as lgb
    
    log_info("\n--- 🚂 [LGBM] 최종 모델 학습 시작 ---")
    log_info("   [DATA] 데이터 로드 중...")
    
    # 전체 학습 데이터 로드 (마지막 폴드의 학습 데이터만 사용 - Expanding Window이므로 전체 포함)
    all_train_data = []
    val_data_df = None
    
    # 마지막 폴드의 검증 데이터를 검증 데이터셋으로 사용
    if fold_ranges:
        last_fold = fold_ranges[-1]
        train_start, train_end, val_start, val_end = last_fold
        
        # 검증 데이터 로드
        val_data_df, _, _ = load_data_period(
            file_paths, features, val_start, val_end,
            imputation_map=None, batch_size=50, max_workers=4
        )
        log_info(f"   📊 검증 데이터셋 로드 완료: {len(val_data_df):,}행 (마지막 폴드 검증 데이터)")
        
        # 학습 데이터 로드 (마지막 폴드의 Train 데이터만 사용)
        log_info(f"   📊 학습 데이터셋 로드 중 (마지막 폴드 Train: {train_start} ~ {train_end})...")
        train_df, _, _ = load_data_period(
            file_paths, features, train_start, train_end,
            imputation_map=None, batch_size=50, max_workers=4
        )
        if train_df is not None and not train_df.empty:
            all_train_data.append(train_df)
    
    if not all_train_data:
        log_error("학습 데이터를 로드할 수 없습니다.")
        return None, None
    
    combined_train_df = pd.concat(all_train_data, ignore_index=True)
    log_info(f"   ✅ 전체 학습 데이터: {len(combined_train_df):,}행")
    
    # 피처와 타겟 분리 (존재하는 피처만 선택)
    available_features = [f for f in features if f in combined_train_df.columns]
    if len(available_features) != len(features):
        missing_features = [f for f in features if f not in combined_train_df.columns]
        log_warning(f"   ⚠️ 누락된 피처: {missing_features}")
        log_info(f"   ✅ 사용 가능한 피처 수: {len(available_features)}/{len(features)}")
    
    X_all = combined_train_df[available_features].copy()
    y_all = combined_train_df['target'].values
    
    # --- 클래스 불균형 확인 및 언더샘플링 (RF와 동일하게 1:1) ---
    sampling_start = datetime.now()
    log_info("   📊 [DATA] 클래스 분포 확인 및 언더샘플링 준비...")
    
    # NumPy 배열을 DataFrame으로 변환하여 value_counts() 사용
    y_all_series = pd.Series(y_all)
    value_counts = y_all_series.value_counts()
    
    if len(value_counts) >= 2:
        minority_class_label = value_counts.idxmin()
        majority_class_label = value_counts.idxmax()
        n_minority = value_counts[minority_class_label]
        n_majority = value_counts[majority_class_label]
        
        log_info(f"      - 소수 클래스 ({minority_class_label}): {n_minority:,}개")
        log_info(f"      - 다수 클래스 ({majority_class_label}): {n_majority:,}개")
        
        # 언더샘플링: 다수 클래스를 소수 클래스 크기만큼 랜덤 선택 (1:1 비율)
        if n_majority > n_minority:
            all_indices = np.arange(len(y_all))
            
            minority_indices = all_indices[y_all == minority_class_label]
            majority_indices = all_indices[y_all == majority_class_label]
            
            log_info(f"      🔀 다수 클래스 랜덤 셔플 및 샘플링 중 (1:1 비율)...")
            
            # 다수 클래스 셔플
            rng = np.random.RandomState(42)
            majority_indices_shuffled = majority_indices.copy()
            rng.shuffle(majority_indices_shuffled)
            
            # 1:1 비율로 샘플링
            selected_majority_indices = majority_indices_shuffled[:n_minority]
            
            # 인덱스 결합
            balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
            balanced_indices.sort()  # 시간 순서 유지
            
            # 언더샘플링된 데이터 생성
            X_all = X_all.iloc[balanced_indices].reset_index(drop=True)
            y_all = y_all[balanced_indices]
            
            log_info(f"   ✅ 언더샘플링 완료: 총 {len(X_all):,}행 ({(datetime.now() - sampling_start).total_seconds():.1f}초)")
        else:
            log_info("   ℹ️ 클래스 불균형이 심하지 않아 샘플링을 건너뜁니다.")
    else:
        log_warning("   ⚠️ 클래스가 1개만 존재합니다.")

    # 결측치 및 무한대 값 처리
    for col in X_all.columns:
        if X_all[col].isna().any() or np.isinf(X_all[col]).any():
            median_val = X_all[col].replace([np.inf, -np.inf], np.nan).median()
            X_all[col] = X_all[col].replace([np.inf, -np.inf], np.nan).fillna(median_val)
    
    # 스케일링 생략
    X_all_scaled = X_all
    
    # GPU 사용 가능 여부 확인
    gpu_available, device = check_gpu_availability()
    
    # 최종 파라미터 설정
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': device,  # GPU 또는 CPU (자동 선택)
        'num_threads': 0 if device == 'gpu' else -1,  # GPU: 0, CPU: -1 (모든 스레드 사용)
        'verbosity': 1,
        'seed': 42,
        'deterministic': True,
        'max_bin': 255,
        'n_estimators': 10000,
    })
    
    # LightGBM Dataset 생성
    train_data = lgb.Dataset(X_all_scaled, label=y_all)
    
    # 검증 데이터셋 준비 (마지막 폴드의 검증 데이터 사용)
    val_data = None
    if val_data_df is not None and not val_data_df.empty:
        # 검증 데이터 전처리
        available_features_val = [f for f in available_features if f in val_data_df.columns]
        X_val = val_data_df[available_features_val].copy()
        y_val = val_data_df['target'].values
        
        # 결측치 및 무한대 값 처리 (학습 데이터의 중앙값 사용)
        for col in X_val.columns:
            if X_val[col].isna().any() or np.isinf(X_val[col]).any():
                # 학습 데이터의 중앙값 사용
                median_val = X_all[col].replace([np.inf, -np.inf], np.nan).median()
                X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan).fillna(median_val)
        
        # 스케일링 생략: 원본 데이터 사용
        X_val_scaled = X_val
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        log_info(f"   ✅ 검증 데이터셋 준비 완료: {len(X_val_scaled):,}행")
    
    # 최종 모델 학습
    device_name = "GPU" if device == 'gpu' else "CPU"
    log_info(f"   🔥 학습 시작 ({device_name})...")
    
    # GPU 실패 시 자동으로 CPU로 전환
    try:
        if val_data is not None:
            # 검증 데이터셋이 있으면 Early Stopping 사용
            final_model = lgb.train(
                final_params,
                train_data,
                num_boost_round=final_params['n_estimators'],
                valid_sets=[val_data],
                valid_names=['val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=True),
                    lgb.log_evaluation(period=100)
                ]
            )
        else:
            # 검증 데이터셋이 없으면 Early Stopping 없이 학습
            log_warning("   ⚠️ 검증 데이터셋이 없어 Early Stopping을 사용하지 않습니다.")
            final_model = lgb.train(
                final_params,
                train_data,
                num_boost_round=final_params['n_estimators'],
                callbacks=[
                    lgb.log_evaluation(period=100)
                ]
            )
    except Exception as gpu_error:
        # GPU 실패 시 CPU로 전환
        if 'gpu' in str(gpu_error).lower() or 'opencl' in str(gpu_error).lower() or 'device' in str(gpu_error).lower():
            log_warning(f"   ⚠️ GPU 학습 실패: {str(gpu_error)}")
            log_info("   🔄 CPU로 자동 전환하여 재시도합니다...")
            # CPU 파라미터로 변경
            final_params_cpu = final_params.copy()
            final_params_cpu['device'] = 'cpu'
            final_params_cpu['num_threads'] = -1  # 모든 CPU 스레드 사용
            # GPU 사용 불가능 플래그 설정
            check_gpu_availability._gpu_available = False
            check_gpu_availability._device = 'cpu'
            # CPU로 재시도
            if val_data is not None:
                # 검증 데이터셋이 있으면 Early Stopping 사용
                final_model = lgb.train(
                    final_params_cpu,
                    train_data,
                    num_boost_round=final_params_cpu['n_estimators'],
                    valid_sets=[val_data],
                    valid_names=['val'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=True),
                        lgb.log_evaluation(period=100)
                    ]
                )
            else:
                # 검증 데이터셋이 없으면 Early Stopping 없이 학습
                final_model = lgb.train(
                    final_params_cpu,
                    train_data,
                    num_boost_round=final_params_cpu['n_estimators'],
                    callbacks=[
                        lgb.log_evaluation(period=100)
                    ]
                )
        else:
            # GPU 관련이 아닌 다른 에러는 그대로 raise
            raise
    
    # best_iteration은 Early Stopping이 사용된 경우에만 존재
    if hasattr(final_model, 'best_iteration') and final_model.best_iteration is not None:
        log_info(f"   ✅ 최종 모델 학습 완료 (best_iteration: {final_model.best_iteration})")
        best_iteration = final_model.best_iteration
    else:
        log_info(f"   ✅ 최종 모델 학습 완료 (총 {final_params['n_estimators']} iterations)")
        best_iteration = final_params['n_estimators']
    
    # 모델 저장
    model_path = path_manager.data_dir / 'lgbm_model.txt'
    final_model.save_model(str(model_path))
    log_info(f"   [OK] 모델 저장 완료: {model_path}")
    
    # --- 중요도 계산 (SHAP & Permutation) ---
    feature_importances = pd.DataFrame({'feature': available_features})
    
    # 1. 기본 중요도 (Gain)
    try:
        gain_importance = final_model.feature_importance(importance_type='gain')
        feature_importances['gain_importance'] = gain_importance
    except Exception as e:
        log_warning(f"   ⚠️ 기본 중요도 계산 실패: {e}")
        feature_importances['gain_importance'] = 0
        
    # 2. SHAP 중요도 (샘플링)
    if SHAP_AVAILABLE:
        try:
            log_info("   [SHAP] 피처 중요도 계산 중...")
            shap_start = datetime.now()
            
            # 데이터 샘플링 (너무 많으면 오래 걸림)
            sample_size = min(1000, len(X_all_scaled))
            # numpy array인 경우
            if isinstance(X_all_scaled, np.ndarray):
                X_sample = X_all_scaled[np.random.choice(X_all_scaled.shape[0], sample_size, replace=False)]
            else:
                X_sample = X_all_scaled.sample(n=sample_size, random_state=42)
            
            # TreeExplainer 사용
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_sample)
            
            # 이진 분류의 경우 shap_values는 리스트일 수 있음 (class 0, class 1)
            # LightGBM의 경우 [0]번 인덱스가 0클래스, [1]번 인덱스가 1클래스일 수 있음
            # 또는 그냥 하나의 배열일 수도 있음
            if isinstance(shap_values, list):
                # 양성 클래스(1)에 대한 SHAP 값 사용
                shap_values_target = shap_values[1]
            else:
                shap_values_target = shap_values
                
            # 절대값 평균 계산
            mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
            feature_importances['shap_importance'] = mean_abs_shap
            
            log_info(f"   [OK] SHAP 계산 완료 ({(datetime.now() - shap_start).total_seconds():.1f}초)")
            
        except Exception as e:
            log_warning(f"   ⚠️ SHAP 계산 실패: {e}")
            feature_importances['shap_importance'] = 0
    else:
        feature_importances['shap_importance'] = 0
        
    # 3. Permutation Importance (검증 데이터 사용)
    # 검증 데이터가 없으면 학습 데이터 일부 사용
    try:
        from sklearn.inspection import permutation_importance
        
        log_info("   [PERM] 순열 중요도 계산 중...")
        perm_start = datetime.now()
        
        # 평가용 데이터 준비
        if val_data is not None:
            X_eval = X_val_scaled
            y_eval = y_val
        else:
            # 학습 데이터 일부 샘플링
            sample_size = min(5000, len(X_all_scaled))
            indices = np.random.choice(len(X_all_scaled), sample_size, replace=False)
            X_eval = X_all_scaled[indices]
            y_eval = y_all[indices]
            
        # LightGBM 모델 래퍼 (sklearn 호환성 위함)
        class LGBMWrapper:
            def __init__(self, model):
                self.model = model
                self._estimator_type = "classifier"  # 분류기임을 명시
                self.classes_ = np.array([0, 1])     # 클래스 정보 추가
            
            def fit(self, X, y):
                # 이미 학습된 모델이므로 아무것도 하지 않음 (sklearn 인터페이스 만족용)
                return self
                
            def predict(self, X):
                # 클래스 예측 (0 또는 1)
                proba = self.model.predict(X)
                return (proba > 0.5).astype(int)
            
            def predict_proba(self, X):
                # LightGBM predict returns class 1 probability
                prob = self.model.predict(X)
                # Sklearn expects [prob_class_0, prob_class_1]
                return np.vstack([1 - prob, prob]).T
                
        wrapper = LGBMWrapper(final_model)
        
        # roc_auc 점수 기준으로 중요도 계산
        # n_repeats=5 정도로 빠르게
        results = permutation_importance(
            wrapper, X_eval, y_eval, 
            scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=1
        )
        
        feature_importances['permutation_importance'] = results.importances_mean
        log_info(f"   [OK] 순열 중요도 계산 완료 ({(datetime.now() - perm_start).total_seconds():.1f}초)")
        
    except Exception as e:
        log_warning(f"   ⚠️ 순열 중요도 계산 실패: {e}")
        feature_importances['permutation_importance'] = 0

    # 중요도 파일 저장
    importance_path = path_manager.data_dir / 'lgbm_feature_importance.csv'
    feature_importances.to_csv(importance_path, index=False)
    log_info(f"   [OK] 피처 중요도 저장 완료: {importance_path}")

    # 메타데이터 저장
    metadata = {
        'features': available_features,  # 실제 사용된 피처만 저장
        'best_params': best_params,
        'best_iteration': best_iteration,
        'best_score': best_score,  # Optuna 최적 점수 저장
        'scaler': None,
    }

    # Optuna 탐색(Trial) 결과/학습 설정은 있을 때만 저장 (하위 호환)
    if isinstance(optimization_results, dict):
        metadata['optimization_results'] = optimization_results
    if isinstance(training_config, dict):
        metadata['training_config'] = training_config
    
    metadata_path = path_manager.data_dir / 'lgbm_model_metadata.joblib'
    # cuDF 객체를 pandas로 변환하여 저장 (joblib 직렬화 호환성)
    metadata_converted = convert_cudf_to_pandas_for_joblib(metadata)
    joblib.dump(metadata_converted, metadata_path)
    log_info(f"   [OK] 메타데이터 저장 완료: {metadata_path}")
    
    return final_model, features

# --- 메인 실행 로직 ---

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LightGBM GPU 가속 모델 훈련 스크립트")
    parser.add_argument('--n_iter', type=int, default=50, help='Optuna 탐색 횟수')
    args = parser.parse_args()
    
    # 사용자 입력 (Trial 횟수)
    try:
        trials_input = input("\n   [LGBM] Optuna Trial 횟수를 입력하세요 (기본값: 50, Enter): ").strip()
        n_trials = int(trials_input) if trials_input else args.n_iter
    except ValueError:
        log_warning("   ⚠️ 올바른 숫자를 입력해주세요. 기본값 50을 사용합니다.")
        n_trials = args.n_iter
    
    # 별도 데이터 경로 설정 (RF와 분리)
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
        # 최근 3개월 전일까지만 수집
        end_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        log_info(f"   데이터 수집 기간: {start_date} ~ {end_date} ({years}년, 최근 3개월 제외)")
        
        if not prepare_data_and_save(data_path, start_date, end_date):
            log_critical("데이터 준비에 실패하여 프로그램을 종료합니다.")
            sys.exit(1)

    # --- 2. 학습 설정 단계 ---
    log_info("\n--- ⚙️ 학습 설정 시작 ---")
    file_paths = glob.glob(os.path.join(data_path, "*.feather"))
    if not file_paths:
        log_critical("학습할 데이터 파일이 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    file_paths = sorted(file_paths)
    log_info(f"   총 {len(file_paths)}개 종목의 데이터를 학습에 사용합니다.")

    # RF와 동일한 피처 목록 (실제 계산되는 피처만 포함)
    features = [
        'log_mktcap',
        '52주_신고가_비율',
        'ADX_14',
        'disparity_120',
        'disparity_240',
        'disparity_20',
        'IXIC_disparity_20',
        'Trend_Pullback_Score',
        'Position_Range_60',
        'MA20_Slope',
        'MA120_Slope',
        'MA240_Slope',
        'IXIC_MA20_Slope',
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
        '등락율(5D)',
    ]

    # --- 3. Fold 분할 (날짜 기반) ---
    log_info("\n--- 📁 Fold 데이터 구성 중 ---")
    
    # 3. Expanding Window 방식의 Fold 범위 계산
    log_info(f"\n--- 📁 Fold 데이터 로딩 및 구성 중 (날짜 기반 Expanding Window) ---")
    
    fold_ranges = calculate_expanding_fold_ranges(file_paths, warmup_days=250, val_period_days=365, n_folds=3)

    # ✅ 10거래일 purge(embargo) 적용: Val 시작 직전 10거래일은 Train에서 제외
    trading_dates = _build_trading_calendar_from_files(file_paths)
    if trading_dates:
        fold_ranges = _apply_purge_to_fold_ranges(fold_ranges, trading_dates, DEFAULT_HORIZON_TRADING_DAYS)
        log_info(f"   🧹 purge 적용: Val 시작 직전 {DEFAULT_HORIZON_TRADING_DAYS}거래일은 Train에서 제외합니다.")
    else:
        log_warning("   ⚠️ 거래일 캘린더 생성 실패: purge(10거래일) 적용을 건너뜁니다.")
    
    if not fold_ranges:
        log_critical("Fold 범위 계산에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # ✅ 정합성 검증 로그: Train/Val 날짜가 겹치지 않는지 확인
    for fr in fold_ranges:
        try:
            if pd.Timestamp(fr['train_end']).normalize() >= pd.Timestamp(fr['val_start']).normalize():
                log_warning(
                    f"   ⚠️ Fold#{fr.get('fold', '?')+1 if isinstance(fr.get('fold', None), int) else fr.get('fold', '?')}: "
                    f"Train_end({pd.Timestamp(fr['train_end']).date()}) >= Val_start({pd.Timestamp(fr['val_start']).date()})"
                )
        except Exception:
            pass
        
    # 4. 데이터 캐싱 (메모리 로드)
    # I/O 병목 제거를 위해 미리 데이터를 로드하여 메모리에 저장합니다.
    fold_data_cache = {}  # {fold_idx: (X_train, y_train, X_val, y_val)}
    
    for fold_info in fold_ranges:
        fold_idx = fold_info['fold']
        train_start = fold_info['train_start'].strftime('%Y-%m-%d')
        train_end = fold_info['train_end'].strftime('%Y-%m-%d')
        val_start = fold_info['val_start'].strftime('%Y-%m-%d')
        val_end = fold_info['val_end'].strftime('%Y-%m-%d')
        
        log_info(f"   📂 Fold #{fold_idx+1} 데이터 캐싱 중...")
        log_info(f"      Train: {train_start} ~ {train_end}")
        log_info(f"      Val:   {val_start} ~ {val_end}")
        
        try:
            # Train 로드
            X_train_df, _, _ = load_data_period(
                file_paths, features, train_start, train_end,
                imputation_map=None, batch_size=100, max_workers=8
            )
            # Val 로드
            X_val_df, _, _ = load_data_period(
                file_paths, features, val_start, val_end,
                imputation_map=None, batch_size=100, max_workers=8
            )
            
            if X_train_df is None or X_val_df is None or X_train_df.empty or X_val_df.empty:
                log_warning(f"   ⚠️ Fold #{fold_idx+1} 데이터가 비어있어 캐싱에 실패했습니다.")
                continue
            
            # 피처와 타겟 분리
            available_features = [f for f in features if f in X_train_df.columns]
            
            X_train = X_train_df[available_features].copy()
            y_train = X_train_df['target'].values
            X_val = X_val_df[available_features].copy()
            y_val = X_val_df['target'].values
            
            # 결측치/무한대 미리 처리 (Objective에서 반복하지 않도록)
            for col in X_train.columns:
                if X_train[col].isna().any() or np.isinf(X_train[col]).any():
                    median_val = X_train[col].replace([np.inf, -np.inf], np.nan).median()
                    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(median_val)
                    X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan).fillna(median_val)
            
            # 메모리에 저장
            fold_data_cache[fold_idx] = (X_train, y_train, X_val, y_val)
            
            # 원본 DataFrame 삭제
            del X_train_df, X_val_df
            gc.collect()
            
            log_info(f"   ✅ Fold #{fold_idx+1} 캐싱 완료 (Train: {len(X_train):,}행, Val: {len(X_val):,}행)")
            
        except Exception as e:
            log_error(f"   ❌ Fold #{fold_idx+1} 데이터 캐싱 중 오류: {e}")
            import traceback
            log_error(traceback.format_exc())
            continue
            
    if not fold_data_cache:
        log_critical("데이터 캐싱에 실패했습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # --- 5. Optuna 최적화 ---
    log_info(f"\n--- 🤖 Optuna 최적화 시작 (n_trials={n_trials}) ---")
    log_info("   🚀 캐시된 데이터를 사용하여 고속 학습을 진행합니다.")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, fold_data_cache, features),
            n_trials=n_trials
        )
    except KeyboardInterrupt:
        log_warning("사용자가 최적화를 중단했습니다.")
    
    log_info(f"\n--- 🏆 최적화 결과 | 최고 점수: {study.best_value:.4f} | 최적 파라미터: {study.best_params} ---")
    
    # --- 6. 최종 모델 학습 ---
    best_params = study.best_params
    best_score = study.best_value

    # Optuna 최적화 결과 요약 (RF와 동일한 키 구조 유지)
    try:
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        optimization_results = {
            'best_score': best_score,
            'best_params': best_params,
            'total_combinations_tested': len(study.trials),
            'n_complete_trials': len(complete_trials),
            'n_pruned_trials': len(pruned_trials),
            'n_trials_requested': int(n_trials),
        }
    except Exception:
        optimization_results = {
            'best_score': best_score,
            'best_params': best_params,
            'total_combinations_tested': None,
            'n_complete_trials': None,
            'n_pruned_trials': None,
            'n_trials_requested': int(n_trials),
        }

    training_config = {
        'search_method': 'optuna_tpe',
        'n_trials': int(n_trials),
    }
    
    # 최종 모델 학습을 위해 fold_ranges 재구성 (문자열 날짜 형식)
    fold_ranges_str = []
    for fold_info in fold_ranges:
        fold_ranges_str.append((
            fold_info['train_start'].strftime('%Y-%m-%d'),
            fold_info['train_end'].strftime('%Y-%m-%d'),
            fold_info['val_start'].strftime('%Y-%m-%d'),
            fold_info['val_end'].strftime('%Y-%m-%d')
        ))
        
    final_model, final_features = train_final_model(
        fold_ranges_str,
        file_paths,
        features,
        best_params,
        best_score,
        optimization_results=optimization_results,
        training_config=training_config
    )
    
    if final_model is None:
        log_critical("최종 모델 학습에 실패했습니다.")
        sys.exit(1)
    
    log_info("\n🎉 LightGBM 모델 학습이 완료되었습니다!")

if __name__ == "__main__":
    main()
