# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import warnings
import argparse
from datetime import datetime
import os
import sys
import io
import shutil # 폴더 삭제를 위해 shutil 라이브러리 임포트
import gc
import psutil

# 크로스 플랫폼 인코딩 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encoding_utils  # 인코딩 유틸리티 임포트

import data_cacher
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

def create_training_data():
    log_info("🚀 캐시 관리 모듈을 통해 학습 데이터 생성을 시작합니다...")
    log_memory_usage("학습 데이터 생성 시작")
    
    start_date_for_cacher = '2015-01-01'
    end_date_for_cacher = datetime.now().strftime('%Y-%m-%d')
    
    try:
        final_df = data_cacher.get_preprocessed_data(start_date_for_cacher, end_date_for_cacher)
        log_memory_usage("데이터 로딩 완료")
        check_memory_and_cleanup()
    except MemoryError as e:
        log_error(f"메모리 부족으로 데이터 로딩 실패: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        final_df = data_cacher.get_preprocessed_data(start_date_for_cacher, end_date_for_cacher)
        log_memory_usage("재시도 후 데이터 로딩 완료")
    
    if final_df is None or final_df.empty:
        log_error("데이터를 가져오는 데 실패했습니다.")
        return None, None, None, None

    log_info(f"\n--- 생성된 학습 데이터 요약 ---")
    log_info(f"1. 전체 수집 데이터 (Raw): {len(final_df):,} 행")
    log_memory_usage("원본 데이터 로딩")
    
    training_start_date = '2016-01-01'
    final_df = final_df[final_df['date'] >= pd.to_datetime(training_start_date)]
    log_info(f"2. 워밍업 기간(2015년) 제외 후 실제 학습 데이터: {len(final_df):,} 행")
    log_memory_usage("데이터 필터링 완료")
    check_memory_and_cleanup()

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
    target = 'target'
    
    for col in features + [target]:
        if col not in final_df.columns:
            log_error(f"오류: 필요한 컬럼 '{col}'이 데이터프레임에 없습니다.")
            return None, None, None, None
            
    final_df.dropna(subset=[target], inplace=True)
    log_info(f"3. 타겟 변수 결측치 제거 후: {len(final_df):,} 행")
    log_memory_usage("결측치 제거 완료")

    if final_df.empty:
        log_error("오류: 최종 학습 데이터가 비어있습니다.")
        return None, None, None, None

    # 메모리 효율적인 데이터 타입 변환
    log_info("   🔄 메모리 효율적인 데이터 타입 변환 중...")
    
    # 필요한 컬럼만 선택하여 메모리 절약
    required_columns = features + [target]
    final_df = final_df[required_columns]
    log_info(f"   📊 필요한 컬럼만 선택: {len(required_columns)}개 컬럼")
    
    # 데이터 타입 최적화
    X = final_df[features].astype(np.float32)  # float64 -> float32로 메모리 절약
    y = final_df[target]
    
    # 원본 데이터프레임 메모리 해제
    del final_df
    gc.collect()
    
    log_memory_usage("데이터 타입 변환 완료")
    
    # 중앙값 계산 (메모리 효율적)
    log_info("   📊 피처별 대표 중앙값 계산 중...")
    imputation_values = X.median()
    log_info("\n--- 피처별 대표 중앙값 (결측치 대체용) ---")
    log_info(str(imputation_values))
    log_info("-----------------------------------------")
    
    # 결측치 대체
    log_info("   🔧 결측치 대체 중...")
    X.fillna(imputation_values, inplace=True)
    log_memory_usage("결측치 대체 완료")
    check_memory_and_cleanup()
    
    log_info(f"4. 최종 학습 데이터셋 (X): {X.shape}")
    log_info(f"   - 타겟 분포 (y):\n{y.value_counts(normalize=True).to_string()}")
    log_info("---------------------------------")
    log_memory_usage("최종 학습 데이터 준비 완료")

    log_info("✅ 학습 데이터 생성 완료!")
    return X, y, features, imputation_values

def train_evaluate_and_save_model(X, y, features, imputation_values, n_jobs, n_iter, max_depth_list, model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_prediction_model_rf_upgraded.joblib')):
    if X is None or y is None or X.empty or y.empty:
        log_error("학습 데이터가 없어 모델링을 건너뜁니다.")
        return

    log_info("🤖 모델 학습 및 평가를 시작합니다...")
    log_memory_usage("모델 학습 시작")
    
    # 메모리 효율적인 train_test_split
    log_info("   📊 학습/테스트 데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    log_memory_usage("데이터 분할 완료")
    check_memory_and_cleanup()

    log_info("\n   📏 피처 스케일링 (StandardScaler) 적용 중...")
    scaler = StandardScaler()
    
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        log_memory_usage("훈련 데이터 스케일링 완료")
        check_memory_and_cleanup()
        
        X_test_scaled = scaler.transform(X_test)
        log_memory_usage("테스트 데이터 스케일링 완료")
        
        # 중간 변수 즉시 메모리 해제
        del X_train, X_test
        gc.collect()
        log_info("   🧹 중간 변수 메모리 해제 완료")
        
    except MemoryError as e:
        log_error(f"스케일링 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 중간 변수 즉시 메모리 해제
        del X_train, X_test
        gc.collect()
        
        log_memory_usage("재시도 후 스케일링 완료")

    log_info("\n학습 데이터 타겟 분포:\n" + str(y_train.value_counts(normalize=True)))

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': max_depth_list,
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_samples': [0.7, 0.8, 0.9, None]
    }

    log_info("🔍 RandomizedSearchCV를 사용하여 최적 파라미터 탐색...")
    log_info(f"   ⚙️ 탐색할 파라미터 조합: {n_iter}개")
    log_info(f"   🔄 교차 검증: 3-fold")
    log_info(f"   💻 사용할 CPU 코어: {n_jobs}")
    
    # 메모리 효율적인 모델 설정
    model = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced', 
        oob_score=False,  # OOB 점수 비활성화로 메모리 절약
        n_jobs=1,  # 각 모델은 단일 코어 사용 (메모리 절약)
        warm_start=False,  # 메모리 절약
        bootstrap=True  # 기본값 유지
    )
    
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist,
        n_iter=n_iter, 
        cv=3, 
        n_jobs=n_jobs, 
        verbose=2, 
        random_state=42, 
        scoring='roc_auc',
        pre_dispatch='2*n_jobs',  # 메모리 효율적 디스패치
        return_train_score=False  # 훈련 점수 저장 안함 (메모리 절약)
    )

    log_memory_usage("하이퍼파라미터 튜닝 시작")
    try:
        random_search.fit(X_train_scaled, y_train)
        log_memory_usage("하이퍼파라미터 튜닝 완료")
    except MemoryError as e:
        log_error(f"하이퍼파라미터 튜닝 중 메모리 부족: {e}")
        log_info("   🔄 메모리 정리 후 재시도합니다...")
        safe_memory_cleanup()
        # 더 작은 파라미터로 재시도
        param_dist_small = {
            'n_estimators': randint(50, 200),
            'max_depth': max_depth_list[:2],  # 처음 2개만 사용
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'max_samples': [0.8, 0.9]
        }
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist_small,
            n_iter=max(5, n_iter//2),  # 반으로 줄임
            cv=3, 
            n_jobs=1,  # 단일 코어로 제한
            verbose=2, 
            random_state=42, 
            scoring='roc_auc'
        )
        random_search.fit(X_train_scaled, y_train)
        log_memory_usage("재시도 후 하이퍼파라미터 튜닝 완료")

    log_info("\n--- 최적 파라미터 탐색 결과 ---")
    log_info(f"최고 점수 (ROC-AUC): {random_search.best_score_:.4f}")
    log_info("최적 파라미터: " + str(random_search.best_params_))

    best_model = random_search.best_estimator_
    
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

    log_info("💾 모델 저장 중...")
    log_memory_usage("모델 저장 시작")
    
    try:
        joblib.dump({
            'model': best_model, 
            'features': features, 
            'scaler': scaler,
            'imputation_values': imputation_values 
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
            'imputation_values': imputation_values 
        }, model_path, compress=3)  # 압축 저장으로 메모리 절약
        log_info(f"\n✅ 재시도 후 모델 저장 완료: '{model_path}'")
        log_memory_usage("재시도 후 모델 저장 완료")

def main():
    parser = argparse.ArgumentParser(description="RandomForest 모델 학습 및 하이퍼파라미터 튜닝")
    parser.add_argument('--n_jobs', type=int, default=-1, help='사용할 CPU 코어 수 (-1은 모든 코어 사용)')
    parser.add_argument('--n_iter', type=int, default=10, help='RandomizedSearchCV 반복 횟수')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[10, 20, 30], help='max_depth 후보 리스트')
    args = parser.parse_args()
    
    # ==============================================================================
    # ✨ 핵심 수정: 임시 폴더 생성 및 자동 삭제 로직 추가 ✨
    # ==============================================================================
    # 1. 프로젝트 루트 경로를 기준으로 임시 폴더 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_folder_path = os.path.join(project_root, '.joblib_temp')

    try:
        # 2. 임시 폴더 생성 및 환경 변수 설정
        os.makedirs(temp_folder_path, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder_path
        log_info(f"joblib 임시 폴더가 '{temp_folder_path}'로 설정되었습니다.")
        log_memory_usage("프로그램 시작")

        # 3. 메인 학습 로직 실행
        X, y, features, imputation_values = create_training_data()
        if X is not None:
            log_info("🎯 모델 학습을 시작합니다...")
            train_evaluate_and_save_model(X, y, features, imputation_values, args.n_jobs, args.n_iter, args.max_depth)
            log_info("🎉 모델 학습이 성공적으로 완료되었습니다!")
        else:
            log_error("❌ 학습 데이터 생성에 실패하여 모델 학습을 중단합니다.")

    finally:
        # 4. 학습 성공/실패 여부와 관계없이 항상 임시 폴더 삭제
        if os.path.exists(temp_folder_path):
            log_info(f"\n🧹 학습 완료 후 임시 폴더 삭제 중: {temp_folder_path}")
            try:
                shutil.rmtree(temp_folder_path)
                log_info("✅ 임시 폴더가 성공적으로 삭제되었습니다.")
            except Exception as e:
                log_warning(f"⚠️ 임시 폴더를 삭제하는 중 오류가 발생했습니다: {e}")
        
        # 최종 메모리 사용량 로그
        log_memory_usage("프로그램 종료")
        log_info("🏁 모든 작업이 완료되었습니다.")
    # ==============================================================================

if __name__ == '__main__':
    main()