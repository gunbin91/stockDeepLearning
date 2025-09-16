import pandas as pd
import numpy as np
from smart_cache import get_cache
from logger import log_info, log_warning, log_error

def calculate_factor_scores(df):
    if df.empty:
        log_warning("입력 데이터가 비어있어 팩터 점수 계산을 건너뜁니다.")
        return df.copy()

    if '종목코드' not in df.columns:
        log_error("'종목코드' 컬럼이 없어 점수 계산을 진행할 수 없습니다.")
        return pd.DataFrame()

    log_info("📊 팩터별 점수 계산을 시작합니다...")
    log_info(f"   📈 입력 데이터: {len(df):,}개 종목")
    scored_df = df.copy()

    
    # 변동성 점수 계산
    if '변동성(1M)' in df.columns:
        log_info("   📊 변동성 점수 계산 중... (1개월 변동성, 낮을수록 좋음)")
        scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100
        log_info(f"      ✅ 변동성 점수 완료: 평균 {scored_df['volatility_score'].mean():.2f}")
    else:
        log_warning("   ⚠️ 변동성 점수 계산을 위한 데이터가 부족합니다.")

    # 점수 반올림
    log_info("   🔢 점수 반올림 처리 중...")
    score_cols = ['volatility_score']
    calculated_scores = []
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
            calculated_scores.append(col)
    
    log_info(f"   📊 계산된 팩터 점수: {len(calculated_scores)}개 ({', '.join(calculated_scores)})")
    log_info("🎉 팩터별 점수 계산이 완료되었습니다!")
        
    return scored_df