import pandas as pd
import numpy as np
import json
import os
# smart_cache 사용 안함 - 실시간 데이터 수집으로 전환
from logger import log_info, log_warning, log_error, log_step, log_success, log_start, log_complete

def calculate_final_score(df):
    if df.empty:
        log_warning("[WARN] 입력 데이터가 비어있어 앙상블 점수 계산을 건너뜁니다.")
        return df.copy()

    log_step("앙상블 점수 계산", "START", {"종목수": len(df)})
    final_df = df.copy()

    # 기본 가중치 설정 (optimal_weights.json 파일이 없을 경우 사용)
    factor_weights = {
        'volatility_score': 0.10,
        'ml_pred_proba': 0.90, # ML 예측 확률
    }

    # 최적화된 가중치 파일이 있으면 불러오기
    script_dir = os.path.dirname(__file__)
    optimal_weights_path = os.path.join(script_dir, 'data', 'optimal_weights.json')

    if os.path.exists(optimal_weights_path):
        with open(optimal_weights_path, 'r') as f:
            loaded_weights = json.load(f)
            factor_weights.update(loaded_weights)
        log_info("[OK] 최적화된 가중치 적용")
    else:
        log_info("[INFO] 기본 가중치 사용")

    active_factors = {k: v for k, v in factor_weights.items() if v > 0 and k in final_df.columns}
    log_info(f"[SEARCH] 활성 팩터: {len(active_factors)}개")
    
    if not active_factors:
        log_warning("[WARN] 활성화된 팩터가 없어 기본 점수를 적용합니다.")
        final_df['final_score'] = 50
        final_df['최종순위'] = 1
        return final_df

    # 실시간 정규화 값 계산 (캐시 사용 안함)
    log_info("   📊 정규화 값 계산 중...")
    cached_norms = {}
    for factor in active_factors.keys():
        if factor == 'ml_pred_proba':
            source_series = final_df[factor] * 100
        else:
            source_series = final_df[factor]
            
        min_val = source_series.min()
        max_val = source_series.max()
        
        if (max_val - min_val) > 0:
            cached_norms[factor] = {
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val
            }
        else:
            cached_norms[factor] = {
                'min': 0,
                'max': 1,
                'range': 1
            }
    
    # 벡터화된 정규화 계산
    for factor in active_factors.keys():
        if factor == 'ml_pred_proba':
            source_series = final_df[factor] * 100
        else:
            source_series = final_df[factor]
        
        norm_info = cached_norms[factor]
        if norm_info['range'] > 0:
            final_df[factor + '_norm'] = 100 * (source_series - norm_info['min']) / norm_info['range']
        else:
            final_df[factor + '_norm'] = 50

    total_weight = sum(active_factors.values())
    normalized_weights = {k: v / total_weight for k, v in active_factors.items()}

    log_info("   🎯 최종 점수 계산 중...")
    final_df['final_score'] = 0
    for factor, weight in normalized_weights.items():
        # '_norm' 컬럼을 사용하여 최종 점수 계산
        if factor + '_norm' in final_df.columns:
             final_df['final_score'] += final_df[factor + '_norm'].fillna(50) * weight

    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
    final_df['final_score'] = final_df['final_score'].round(2)

    # <<< 핵심 수정: 계산에 사용했던 임시 '_norm' 컬럼들 삭제 >>>
    cols_to_drop = [col for col in final_df.columns if '_norm' in col]
    final_df.drop(columns=cols_to_drop, inplace=True)

    # 결과 통계
    avg_score = final_df['final_score'].mean()
    top_score = final_df['final_score'].max()
    log_info(f"🎉 앙상블 점수 계산 완료! (평균: {avg_score:.1f}, 최고: {top_score:.1f})")

    return final_df.sort_values(by='최종순위')
