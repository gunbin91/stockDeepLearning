import pandas as pd
import numpy as np
# smart_cache 사용 안함 - 실시간 데이터 수집으로 전환
from logger import log_info, log_warning, log_error, log_step, log_success, log_start, log_complete

def calculate_factor_scores(df):
    if df.empty:
        log_warning("[WARN] 입력 데이터가 비어있어 팩터 점수 계산을 건너뜁니다.")
        return df.copy()

    if '종목코드' not in df.columns:
        log_error("[ERROR] '종목코드' 컬럼이 없어 점수 계산을 진행할 수 없습니다.")
        return pd.DataFrame()

    log_step("팩터 점수 계산", "START", {"종목수": len(df)})
    scored_df = df.copy()

    # 변동성 점수 계산
    if '변동성(1M)' in df.columns:
        scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100
        log_info(f"[OK] 변동성 점수 완료 (평균: {scored_df['volatility_score'].mean():.1f})")
    else:
        log_warning("[WARN] 변동성 데이터 부족")

    # 점수 반올림
    score_cols = ['volatility_score']
    calculated_scores = []
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
            calculated_scores.append(col)
    
    log_step("팩터 점수 계산", "COMPLETE", {"팩터수": len(calculated_scores)})
        
    return scored_df