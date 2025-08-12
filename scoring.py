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
    if df.empty:
        print("경고: calculate_factor_scores에 빈 데이터프레임이 전달되었습니다. 빈 데이터프레임을 반환합니다.")
        return df.copy()

    scored_df = df.copy()

    # 1. 가치(Value) 팩터: PER, PBR (낮을수록 좋음)
    scored_df['value_score'] = (scored_df['PER'].rank(ascending=True) + 
                                scored_df['PBR'].rank(ascending=True)).rank(pct=True) * 100

    # 2. 퀄리티(Quality) 팩터: ROE (높을수록 좋음)
    scored_df['quality_score'] = scored_df['ROE'].rank(pct=True) * 100
    
    # 3. 모멘텀(Momentum) 팩터: 1, 3개월 수익률 (높을수록 좋음)
    momentum_rank = (scored_df['수익률(1M)'].rank(pct=True) +
                     scored_df['수익률(3M)'].rank(pct=True))
    scored_df['momentum_score'] = momentum_rank.rank(pct=True) * 100

    # 4. 수급(Supply/Demand) 팩터: 기관/외국인 순매수 (높을수록 좋음)
    # 데이터가 모두 NaN이거나 동일한 값일 경우 랭크 계산 오류 방지
    if scored_df['기관순매수(억)'].nunique() <= 1 and scored_df['외국인순매수(억)'].nunique() <= 1:
        scored_df['supply_score'] = 50.0 # 기본값 할당
    else:
        scored_df['supply_score'] = (scored_df['기관순매수(억)'].rank(ascending=True) +
                                     scored_df['외국인순매수(억)'].rank(ascending=True)).rank(pct=True) * 100

    # 5. 변동성(Volatility) 팩터: 1개월 변동성 (낮을수록 좋음)
    scored_df['volatility_score'] = scored_df['변동성(1M)'].rank(ascending=True, pct=True) * 100

    # 각 점수는 소수점 둘째 자리까지 반올림
    print("Scored DataFrame columns before rounding:", scored_df.columns)
    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    for col in score_cols:
        scored_df[col] = scored_df[col].round(2)
        
    return scored_df