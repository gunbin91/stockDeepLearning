import pandas as pd
import numpy as np
import json
import os

def calculate_final_score(df):
    if df.empty:
        return df.copy()

    final_df = df.copy()

    # 기본 가중치 설정
    factor_weights = {
        'value_score': 0.20,
        'quality_score': 0.20,
        'momentum_score': 0.30,
        'supply_score': 0.00,
        'volatility_score': 0.10,
        'ml_pred_proba': 0.20,
        'sentiment_score': 0.00,
        'dl_trend_score(더미)': 0.00,
    }

    # 최적화된 가중치 파일이 있으면 불러오기
    if os.path.exists('optimal_weights.json'):
        print("INFO: `optimal_weights.json` 파일을 발견하여 가중치를 적용합니다.")
        with open('optimal_weights.json', 'r') as f:
            factor_weights = json.load(f)


    active_factors = {k: v for k, v in factor_weights.items() if v > 0 and k in final_df.columns}
    total_weight = sum(active_factors.values())
    if total_weight > 0:
        for k in active_factors:
            factor_weights[k] /= total_weight

    final_df['final_score'] = 0
    for factor, weight in active_factors.items():
        if factor in final_df.columns:
             final_df['final_score'] += final_df[factor].fillna(0) * weight

    min_score = final_df['final_score'].min()
    max_score = final_df['final_score'].max()
    if max_score > min_score:
        final_df['final_score'] = 100 * (final_df['final_score'] - min_score) / (max_score - min_score)
    else:
        final_df['final_score'] = 50

    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
    final_df['final_score'] = final_df['final_score'].round(2)

    return final_df.sort_values(by='최종순위')
