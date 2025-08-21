import pandas as pd
import numpy as np
import json
import os

def calculate_final_score(df):
    if df.empty:
        return df.copy()

    final_df = df.copy()

    # 기본 가중치 설정 (optimal_weights.json 파일이 없을 경우 사용)
    factor_weights = {
        'value_score': 0.15,
        'quality_score': 0.15,
        'momentum_score': 0.20,
        'supply_score': 0.10,
        'volatility_score': 0.10,
        'ml_pred_proba': 0.30, # ML 예측 확률
        'sentiment_score': 0.00,
        'dl_trend_score(더미)': 0.00,
    }

    # 최적화된 가중치 파일이 있으면 불러오기
    if os.path.exists('optimal_weights.json'):
        print("INFO: `optimal_weights.json` 파일을 발견하여 가중치를 적용합니다.")
        with open('optimal_weights.json', 'r') as f:
            loaded_weights = json.load(f)
            factor_weights.update(loaded_weights)

    active_factors = {k: v for k, v in factor_weights.items() if v > 0 and k in final_df.columns}
    
    if not active_factors:
        final_df['final_score'] = 50
        final_df['최종순위'] = 1
        return final_df

    # <<< 핵심 수정: 원본 컬럼을 덮어쓰지 않고, '_norm' 임시 컬럼에 정규화된 점수 저장 >>>
    for factor in active_factors.keys():
        # ml_pred_proba는 0~1 사이 값이므로 100을 곱해 다른 0~100점짜리 팩터와 스케일을 맞춤
        if factor == 'ml_pred_proba':
             # 원본은 그대로 두고, 100을 곱한 값을 사용
            source_series = final_df[factor] * 100
        else:
            source_series = final_df[factor]
            
        min_val = source_series.min()
        max_val = source_series.max()
        
        if (max_val - min_val) > 0:
            final_df[factor + '_norm'] = 100 * (source_series - min_val) / (max_val - min_val)
        else:
            final_df[factor + '_norm'] = 50 # 모든 값이 같으면 중간값인 50점 부여

    total_weight = sum(active_factors.values())
    normalized_weights = {k: v / total_weight for k, v in active_factors.items()}

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

    return final_df.sort_values(by='최종순위')
