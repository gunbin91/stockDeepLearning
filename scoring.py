import pandas as pd
import numpy as np

def calculate_factor_scores(df):
    if df.empty:
        return df.copy()

    if '종목코드' not in df.columns:
        print("오류: '종목코드' 컬럼이 없어 점수 계산을 진행할 수 없습니다.")
        return pd.DataFrame()

    scored_df = df.copy()

    if '이익수익률' in df.columns and 'PBR' in df.columns:
        ep_ratio_rank = df['이익수익률'].rank(method='min', pct=True)
        pbr_rank = df['PBR'].apply(lambda x: 1/x if x > 0 else -np.inf).rank(method='min', pct=True)
        scored_df['value_score'] = (ep_ratio_rank + pbr_rank).rank(method='min', pct=True) * 100
    
    if 'ROE' in df.columns:
        scored_df['quality_score'] = df['ROE'].rank(method='min', pct=True, na_option='bottom') * 100
    
    if '수익률(1M)' in df.columns and '수익률(3M)' in df.columns:
        momentum_rank = (df['수익률(1M)'].rank(method='min', pct=True, na_option='bottom') +
                         df['수익률(3M)'].rank(method='min', pct=True, na_option='bottom'))
        scored_df['momentum_score'] = momentum_rank.rank(method='min', pct=True) * 100

    if '거래대금_MA20' in df.columns:
        scored_df['supply_score'] = df['거래대금_MA20'].rank(method='min', pct=True, na_option='bottom') * 100
    
    if '변동성(1M)' in df.columns:
        scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100

    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
        
    return scored_df