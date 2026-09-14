"""
일자별 횡단면(cross-sectional) 정규화 유틸.

같은 날 전 종목 대비 백분위 랭크(0~1)를 추가한다.
학습(data_processor)과 실시간(data_fetcher)에서 동일 함수를 사용한다.
docs/PLAN_model_improvement.md Phase 2-1
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# 절대 수준이라 레짐에 취약한 피처 → _cs 컬럼 생성 대상
CROSS_SECTIONAL_FEATURE_COLS: tuple[str, ...] = (
    "log_mktcap",
    "disparity_20",
    "disparity_120",
    "disparity_240",
    "ATRr_5",
    "ATRr_20",
    "ATRr_60",
    "HV_Volatility_5",
    "HV_Volatility_20",
    "HV_Volatility_60",
    "RVOL",
    "시총 회전율(1W)",
    "시총 회전율(3M)",
    "Max_Drawdown_20",
    "등락율(5D)",
    "VWAP_Disparity_5",
    "ADX_14",
    "MA20_Slope",
    "MA120_Slope",
    "MA240_Slope",
    # Phase C (PLAN_model_improvement_v2): 비대칭/방향성 피처
    "Downside_Vol_20",
    "Upside_Downside_Vol_Ratio_20",
    "Return_Skew_60",
    "Down_Day_Ratio_20",
    "Down_Volume_Ratio_20",
    "Gap_Mean_20",
    "Intraday_Strength_20",
    "Amihud_Illiq_20",
)

# 원본 유지 (이미 상대척도이거나 시장 전체 값 — KOSPI는 Phase 2-2)
CROSS_SECTIONAL_SKIP_COLS: tuple[str, ...] = (
    "52주_신고가_비율",
    "Position_Range_60",
    "CLV",
    "RSI_Signal_Oscillator",
    "Trend_Pullback_Score",
    "KOSPI_disparity_20",
    "KOSPI_MA20_Slope",
)

# 당일 종목 수가 이보다 적으면 랭크 불안정 → _cs = NaN
DEFAULT_MIN_COUNT = 100


def add_cross_sectional_ranks(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    date_col: str = "date",
    suffix: str = "_cs",
    min_count: int = DEFAULT_MIN_COUNT,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    일자별 백분위 랭크 컬럼(`{col}_cs`)을 추가한다.

    - 변환식: groupby(date)[col].rank(method='average', pct=True)
    - 당일 행 수 < min_count 이면 해당 일자 _cs = NaN
    - 원본 컬럼은 덮어쓰지 않음
    """
    if df is None or df.empty:
        return df

    out = df if inplace else df.copy()
    cols = list(columns) if columns is not None else list(CROSS_SECTIONAL_FEATURE_COLS)
    existing = [c for c in cols if c in out.columns]
    if not existing:
        return out

    if date_col not in out.columns:
        raise ValueError(f"횡단면 정규화에 '{date_col}' 컬럼이 필요합니다.")

    # 일자별 종목 수 (행 수). 피처 결측과 무관하게 유니버스 크기로 판정.
    day_count = out.groupby(date_col, sort=False)[date_col].transform("size")
    valid_day = day_count >= int(min_count)

    for col in existing:
        ranked = out.groupby(date_col, sort=False)[col].rank(method="average", pct=True)
        out[f"{col}{suffix}"] = ranked.where(valid_day, np.nan).astype(np.float32)

    return out


def mean_daily_auc(dates, y_true, y_score, *, min_pos: int = 5) -> float:
    """일자별 ROC-AUC 평균. 전략(당일 랭킹)과 정합되는 Optuna 지표.

    양/음 클래스가 모두 없거나 양성 수가 min_pos 미만인 날짜는 건너뛴다.
    유효 일자가 없으면 0.0 반환.
    """
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "d": pd.to_datetime(dates),
            "y": np.asarray(y_true),
            "s": np.asarray(y_score),
        }
    )
    scores = []
    for _, g in df.groupby("d", sort=False):
        if g["y"].nunique() < 2:
            continue
        if int(g["y"].sum()) < int(min_pos):
            continue
        try:
            scores.append(float(roc_auc_score(g["y"], g["s"])))
        except ValueError:
            continue
    if not scores:
        return 0.0
    return float(np.mean(scores))


def mean_daily_precision_at_k(dates, y_true, y_score, *, k: int = 15, min_count: int = 50) -> float:
    """일자별 상위 K종목의 양성 비율 평균. 전략(buy_universe_rank)과 직접 정합.

    당일 종목 수가 min_count 미만이면 건너뛴다. 유효 일자가 없으면 0.0.
    docs/PLAN_model_improvement_v2.md Task B-3
    """
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(dates),
            "y": np.asarray(y_true),
            "s": np.asarray(y_score),
        }
    )
    scores = []
    for _, g in df.groupby("d", sort=False):
        if len(g) < int(min_count):
            continue
        top = g.nlargest(int(k), "s")
        if top.empty:
            continue
        scores.append(float(top["y"].mean()))
    if not scores:
        return 0.0
    return float(np.mean(scores))
