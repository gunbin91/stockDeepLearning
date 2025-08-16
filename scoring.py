import pandas as pd
import numpy as np

def calculate_factor_scores(df):
    if df.empty:
        print("경고: calculate_factor_scores에 빈 데이터프레임이 전달되었습니다.")
        return df.copy()

    essential_cols = ['종목코드', '종목명']
    if not all(col in df.columns for col in essential_cols):
        return pd.DataFrame()

    cols_to_keep = essential_cols + [col for col in ['현재가', '시가총액'] if col in df.columns]
    scored_df = df[cols_to_keep].copy()

    per_rank = df['PER'].apply(lambda x: 1/x if x > 0 else -np.inf).rank(method='min', pct=True)
    pbr_rank = df['PBR'].apply(lambda x: 1/x if x > 0 else -np.inf).rank(method='min', pct=True)
    scored_df['value_score'] = (per_rank + pbr_rank).rank(method='min', pct=True) * 100
    scored_df['quality_score'] = df['ROE'].rank(method='min', pct=True, na_option='bottom') * 100
    
    momentum_rank = (df['수익률(1M)'].rank(method='min', pct=True, na_option='bottom') +
                     df['수익률(3M)'].rank(method='min', pct=True, na_option='bottom'))
    scored_df['momentum_score'] = momentum_rank.rank(method='min', pct=True) * 100

    scored_df['supply_score'] = 0
    scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100

    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
        
    return scored_df
