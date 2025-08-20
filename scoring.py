import pandas as pd
import numpy as np

def calculate_factor_scores(df):
    if df.empty:
        print("경고: calculate_factor_scores에 빈 데이터프레임이 전달되었습니다.")
        return df.copy()

    # <<< 핵심 수정: '종목명'을 필수에서 선택 사항으로 변경 >>>
    # 필수로 필요한 컬럼은 '종목코드' 하나면 충분합니다.
    if '종목코드' not in df.columns:
        print("오류: '종목코드' 컬럼이 없어 점수 계산을 진행할 수 없습니다.")
        return pd.DataFrame()

    # '종목명'과 같이 화면 표시에 필요한 컬럼들은 있을 때만 포함합니다.
    cols_to_keep = ['종목코드']
    optional_cols = ['종목명', '현재가', '시가총액']
    for col in optional_cols:
        if col in df.columns:
            cols_to_keep.append(col)
    
    scored_df = df[cols_to_keep].copy()

    # 각 팩터 점수 계산 (팩터 계산에 필요한 컬럼이 없는 경우를 대비하여 방어 코드 추가)
    if 'PER' in df.columns and 'PBR' in df.columns:
        per_rank = df['PER'].apply(lambda x: 1/x if x > 0 else -np.inf).rank(method='min', pct=True)
        pbr_rank = df['PBR'].apply(lambda x: 1/x if x > 0 else -np.inf).rank(method='min', pct=True)
        scored_df['value_score'] = (per_rank + pbr_rank).rank(method='min', pct=True) * 100
    
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

    # 계산된 점수 컬럼만 반올림
    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
        
    return scored_df