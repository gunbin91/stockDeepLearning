"""
팩터 점수 계산 모듈
==================

이 파일은 주식의 투자 매력도를 평가하기 위한 팩터 점수를 계산합니다.
각 팩터는 0-100점 사이의 점수로 정규화되어 종목 간 비교가 가능합니다.

주요 팩터:
- (2025-12) 프로젝트 정책 변경: "변동성 팩터"는 더 이상 사용하지 않음
- 향후 필요 시 가치, 퀄리티, 모멘텀 등 팩터를 추가할 수 있음
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

    # 변동성 팩터 제거 정책에 따라, 현재는 별도 팩터 점수를 계산하지 않습니다.
    # (기존 호환성을 위해 함수는 유지하며 입력 df를 그대로 반환합니다.)
    log_step("팩터 점수 계산", "SKIP", {"종목수": len(df), "사유": "변동성 팩터 미사용"})
    return df.copy()