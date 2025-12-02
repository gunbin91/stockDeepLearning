"""
팩터 점수 계산 모듈
==================

이 파일은 주식의 투자 매력도를 평가하기 위한 팩터 점수를 계산합니다.
각 팩터는 0-100점 사이의 점수로 정규화되어 종목 간 비교가 가능합니다.

주요 팩터:
- 변동성 점수: 낮을수록 좋음 (안정적인 종목 선호)
- 향후 추가 팩터 확장 가능 (가치, 퀄리티, 모멘텀 등)
"""

import pandas as pd
import numpy as np
from logger import log_info, log_warning, log_error, log_step, log_success, log_start, log_complete

def calculate_factor_scores(df):
    """
    팩터 점수 계산 함수
    
    주식의 투자 매력도를 평가하기 위한 팩터 점수를 계산합니다.
    현재는 변동성 점수만 계산하지만, 향후 가치, 퀄리티, 모멘텀 등
    다양한 팩터를 추가할 수 있습니다.
    
    Args:
        df: 종목 데이터가 포함된 데이터프레임
        
    Returns:
        pandas.DataFrame: 팩터 점수가 추가된 데이터프레임
    """
    if df.empty:
        log_warning("[WARN] 입력 데이터가 비어있어 팩터 점수 계산을 건너뜁니다.")
        return df.copy()

    if '종목코드' not in df.columns:
        log_error("[ERROR] '종목코드' 컬럼이 없어 점수 계산을 진행할 수 없습니다.")
        return pd.DataFrame()

    log_step("팩터 점수 계산", "START", {"종목수": len(df)})
    scored_df = df.copy()

    # 변동성 점수 계산 (낮을수록 좋음)
    # ATRr_20을 사용하여 변동성 점수 계산
    if 'ATRr_20' in df.columns:
        # 변동성이 낮은 종목일수록 높은 점수를 받도록 순위 계산
        scored_df['volatility_score'] = df['ATRr_20'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100
        log_info(f"[OK] 변동성 점수 완료 (평균: {scored_df['volatility_score'].mean():.1f})")
    elif '변동성(1M)' in df.columns:
        # 기존 변동성(1M)이 있는 경우 사용 (하위 호환성)
        scored_df['volatility_score'] = df['변동성(1M)'].rank(method='min', ascending=True, pct=True, na_option='bottom') * 100
        log_info(f"[OK] 변동성 점수 완료 (평균: {scored_df['volatility_score'].mean():.1f})")
    else:
        log_warning("[WARN] 변동성 데이터 부족 - 기본값 50 설정")
        scored_df['volatility_score'] = 50.0  # 기본값 설정

    # 점수 반올림 (소수점 2자리까지)
    score_cols = ['volatility_score']
    calculated_scores = []
    for col in score_cols:
        if col in scored_df.columns:
            scored_df[col] = scored_df[col].round(2)
            calculated_scores.append(col)
    
    log_step("팩터 점수 계산", "COMPLETE", {"팩터수": len(calculated_scores)})
        
    return scored_df