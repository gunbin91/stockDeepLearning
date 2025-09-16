import pandas as pd
import numpy as np
import json
import os
from smart_cache import get_cache
from logger import log_info, log_warning, log_error

def calculate_final_score(df):
    if df.empty:
        log_warning("입력 데이터가 비어있어 앙상블 점수 계산을 건너뜁니다.")
        return df.copy()

    log_info("🎯 앙상블 점수 계산을 시작합니다...")
    log_info(f"   📊 입력 데이터: {len(df):,}개 종목")
    final_df = df.copy()

    # 기본 가중치 설정 (optimal_weights.json 파일이 없을 경우 사용)
    factor_weights = {
        'volatility_score': 0.10,
        'ml_pred_proba': 0.90, # ML 예측 확률
        'sentiment_score': 0.00,
        'dl_trend_score(더미)': 0.00,
    }

    # 최적화된 가중치 파일이 있으면 불러오기
    # ensemble.py 파일의 절대 경로를 기준으로 optimal_weights.json 파일 경로 설정
    script_dir = os.path.dirname(__file__)
    optimal_weights_path = os.path.join(script_dir, 'data', 'optimal_weights.json')

    if os.path.exists(optimal_weights_path):
        log_info(f"   📁 최적화된 가중치 파일 발견: {optimal_weights_path}")
        with open(optimal_weights_path, 'r') as f:
            loaded_weights = json.load(f)
            factor_weights.update(loaded_weights)
        log_info("   ✅ 최적화된 가중치를 적용합니다.")
    else:
        log_info("   📋 기본 가중치를 사용합니다.")

    active_factors = {k: v for k, v in factor_weights.items() if v > 0 and k in final_df.columns}
    
    log_info(f"   🔍 활성화된 팩터: {len(active_factors)}개")
    for factor, weight in active_factors.items():
        log_info(f"      - {factor}: {weight:.3f}")
    
    if not active_factors:
        log_warning("활성화된 팩터가 없어 기본 점수를 적용합니다.")
        final_df['final_score'] = 50
        final_df['최종순위'] = 1
        return final_df

    # 캐시된 정규화 값 사용
    log_info("   🔄 정규화 값 계산/캐시 확인 중...")
    cache = get_cache()
    cache_key = f"normalization_{hash(str(active_factors.keys()))}"
    
    # 정규화 값 캐시 확인
    cached_norms = cache.get('normalization', {'key': cache_key}, ttl_seconds=1800)
    
    if cached_norms is None:
        log_info("   📊 새로운 정규화 값을 계산합니다...")
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
        
        # 캐시 저장
        cache.set('normalization', {'key': cache_key}, cached_norms, ttl_seconds=1800)
        log_info("   💾 정규화 값을 캐시에 저장했습니다.")
    else:
        log_info("   ✅ 캐시된 정규화 값을 사용합니다.")
    
    # 벡터화된 정규화 계산
    log_info("   📏 팩터별 정규화 계산 중...")
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

    log_info("   🎯 가중 평균으로 최종 점수 계산 중...")
    final_df['final_score'] = 0
    for factor, weight in normalized_weights.items():
        # '_norm' 컬럼을 사용하여 최종 점수 계산
        if factor + '_norm' in final_df.columns:
             final_df['final_score'] += final_df[factor + '_norm'].fillna(50) * weight

    log_info("   📊 최종 순위 계산 중...")
    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
    final_df['final_score'] = final_df['final_score'].round(2)

    # <<< 핵심 수정: 계산에 사용했던 임시 '_norm' 컬럼들 삭제 >>>
    cols_to_drop = [col for col in final_df.columns if '_norm' in col]
    final_df.drop(columns=cols_to_drop, inplace=True)

    # 결과 통계
    avg_score = final_df['final_score'].mean()
    top_score = final_df['final_score'].max()
    log_info(f"   📈 최종 점수 통계: 평균 {avg_score:.2f}, 최고점 {top_score:.2f}")
    log_info("🎉 앙상블 점수 계산이 완료되었습니다!")

    return final_df.sort_values(by='최종순위')
