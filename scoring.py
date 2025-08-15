import pandas as pd
import numpy as np

def calculate_factor_scores(df):
    """
    데이터프레임을 받아 팩터 기반 스코어를 계산하고,
    원본의 주요 정보(종목코드, 종목명 등)와 스코어 컬럼을 반환합니다.
    """
    if df.empty:
        print("경고: calculate_factor_scores에 빈 데이터프레임이 전달되었습니다. 빈 데이터프레임을 반환합니다.")
        return df.copy()

    # 필수 컬럼 확인
    essential_cols = ['종목코드', '종목명']
    if not all(col in df.columns for col in essential_cols):
        print(f"오류: 입력 데이터에 필수 컬럼({essential_cols})이 부족합니다.")
        # 빈 결과라도 반환하여 이후 로직 중단을 최소화
        return pd.DataFrame(columns=essential_cols + ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score'])

    # 점수 계산의 기반이 될 주요 컬럼들을 먼저 복사
    # app.py의 최종 표시에 필요한 컬럼들도 여기서 미리 챙겨갑니다.
    display_cols_to_keep = ['현재가', '시가총액']
    cols_to_keep = essential_cols + [col for col in display_cols_to_keep if col in df.columns]
    scored_df = df[cols_to_keep].copy()

    # 1. 가치(Value) 팩터: PER, PBR (낮을수록 좋음)
    scored_df['value_score'] = (df['PER'].rank(method='min', ascending=True, na_option='bottom') + 
                                df['PBR'].rank(method='min', ascending=True, na_option='bottom')).rank(method='min', pct=True) * 100

    # 2. 퀄리티(Quality) 팩터: ROE (높을수록 좋음)
    scored_df['quality_score'] = df['ROE'].rank(method='min', pct=True, na_option='bottom') * 100
    
    # 3. 모멘텀(Momentum) 팩터: 1, 3개월 수익률 (높을수록 좋음)
    momentum_rank = (df['수익률(1M)'].rank(method='min', pct=True, na_option='bottom') +
                     df['수익률(3M)'].rank(method='min', pct=True, na_option='bottom'))
    scored_df['momentum_score'] = momentum_rank.rank(method='min', pct=True) * 100

    # 4. 수급(Supply/Demand) 팩터: 기관/외국인 순매수 (높을수록 좋음)
    supply_rank = (df['기관순매수(억)'].rank(method='min', pct=True, na_option='bottom') +
                   df['외국인순매수(억)'].rank(method='min', pct=True, na_option='bottom'))
    scored_df['supply_score'] = supply_rank.rank(method='min', pct=True) * 100

    # 5. 변동성(Volatility) 팩터: 1개월 변동성 (낮을수록 좋음)
    scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100

    # 각 점수는 소수점 둘째 자리까지 반올림
    score_cols = ['value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
        
    return scored_df