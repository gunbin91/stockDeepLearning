
import pandas as pd
import numpy as np

def calculate_factor_scores(df):
    """
    데이터프레임을 받아 팩터 기반 스코어를 계산합니다.
    각 팩터는 0~100점 척도로 정규화됩니다.

    [책임]
    - 가치(Value) 팩터 스코어 계산
    - 퀄리티(Quality) 팩터 스코어 계산 (ROE 사용)
    - 모멘텀(Momentum) 팩터 스코어 계산 (실제 데이터 기반)
    - 수급(Supply) 팩터 스코어 계산
    """
    scored_df = df.copy()

    # 1. 가치(Value) 팩터: PER, PBR (낮을수록 좋음)
    scored_df['value_score'] = (scored_df['PER'].rank(ascending=True) + 
                                scored_df['PBR'].rank(ascending=True)).rank(pct=True) * 100

    # 2. 퀄리티(Quality) 팩터: ROE (높을수록 좋음)
    scored_df['quality_score'] = scored_df['ROE'].rank(pct=True) * 100
    
    # 3. 모멘텀(Momentum) 팩터: 3, 6, 12개월 수익률 (높을수록 좋음)
    # 각 기간의 수익률 순위를 합산하여 종합 모멘텀 순위를 계산
    momentum_rank = (scored_df['수익률(3M)'].rank(pct=True) +
                     scored_df['수익률(6M)'].rank(pct=True) +
                     scored_df['수익률(12M)'].rank(pct=True))
    scored_df['momentum_score'] = momentum_rank.rank(pct=True) * 100

    # 4. 수급(Supply/Demand) 팩터: 기관/외국인 순매수 (높을수록 좋음)
    scored_df['supply_score'] = (scored_df['기관순매수(억)'].rank(ascending=True) +
                                 scored_df['외국인순매수(억)'].rank(ascending=True)).rank(pct=True) * 100

    # 각 점수는 소수점 둘째 자리까지 반올림
    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score']
    for col in score_cols:
        scored_df[col] = scored_df[col].round(2)
        
    return scored_df
