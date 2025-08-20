import pandas as pd
import numpy as np
import json
import os

def calculate_final_score(df):
    if df.empty:
        return df.copy()

    final_df = df.copy()

    # <<< 개선됨: ml_score -> ml_pred_proba로 기본 가중치 키 이름 수정
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
            # <<< 개선됨: 파일에 없는 키가 있어도 오류나지 않도록 기본값과 업데이트
            loaded_weights = json.load(f)
            factor_weights.update(loaded_weights)

    # 가중치가 0보다 크고, 실제 데이터프레임에 존재하는 팩터만 사용
    active_factors = {k: v for k, v in factor_weights.items() if v > 0 and k in final_df.columns}
    
    if not active_factors:
        print("경고: 최종 점수 계산에 사용될 유효한 팩터가 없습니다.")
        final_df['final_score'] = 50
        final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
        return final_df.sort_values(by='최종순위')

    # 각 팩터 점수를 0-100점으로 정규화 (Min-Max Scaling)
    for factor in active_factors.keys():
        # ml_pred_proba는 이미 0~1 사이 값이므로 100을 곱해 스케일 맞춤
        if factor == 'ml_pred_proba':
            final_df[factor] = final_df[factor] * 100
        
        min_val = final_df[factor].min()
        max_val = final_df[factor].max()
        
        # 모든 값이 동일하여 분모가 0이 되는 경우 방지
        if (max_val - min_val) > 0:
            final_df[factor + '_norm'] = 100 * (final_df[factor] - min_val) / (max_val - min_val)
        else:
            final_df[factor + '_norm'] = 50

    # 가중치 정규화 (총합이 1이 되도록)
    total_weight = sum(active_factors.values())
    normalized_weights = {k: v / total_weight for k, v in active_factors.items()}

    # 최종 점수 계산
    final_df['final_score'] = 0
    for factor, weight in normalized_weights.items():
        # 정규화된 점수 사용
        final_df['final_score'] += final_df[factor + '_norm'].fillna(50) * weight

    # 순위 계산
    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
    final_df['final_score'] = final_df['final_score'].round(2)
    
    # 임시로 생성된 정규화 컬럼 제거
    cols_to_drop = [col for col in final_df.columns if '_norm' in col]
    final_df.drop(columns=cols_to_drop, inplace=True)

    return final_df.sort_values(by='최종순위')