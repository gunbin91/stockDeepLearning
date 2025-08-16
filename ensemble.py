import pandas as pd
import numpy as np

def calculate_final_score(df):
    if df.empty:
        return df.copy()

    final_df = df.copy()

    factor_weights = {
        'value_score': 0.20,      # 가치 (0.15 -> 0.20)
        'quality_score': 0.20,    # 퀄리티 (0.15 -> 0.20)
        'momentum_score': 0.30,   # 모멘텀 (0.25 -> 0.30)
        'supply_score': 0.00,     # 수급 (0.15 -> 0.00)
        'volatility_score': 0.10,
        'ml_pred_proba': 0.20,
        'sentiment_score': 0.00,
        'dl_trend_score(더미)': 0.00,
    }

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
