"""
Sample weight / stratified flat undersampling helpers.
See docs/PLAN_sample_weight_stratified.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed weights (plan §1.1)
SAMPLE_WEIGHT_MAP = {
    "pos": 1.0,
    "weak": 1.2,
    "crash": 1.7,
    "flat": 0.7,
}

FLAT_KEEP_K = 1.0  # keep at most k * n_pos flat rows

_MISSING_COLS_MSG = (
    "processed_feather에 sample_type/sample_weight 컬럼이 없습니다. "
    "data_processor로 feather를 재생성한 뒤 다시 학습하세요. "
    "(docs/PLAN_sample_weight_stratified.md)"
)


def require_sample_meta(sample_type, sample_weight=None, *, need_weight: bool = True, context: str = ""):
    """Raise if stratified training metadata is missing (hard-fail; no silent skip)."""
    prefix = f"{context}: " if context else ""
    if sample_type is None:
        raise ValueError(f"{prefix}sample_type 없음. {_MISSING_COLS_MSG}")
    if need_weight and sample_weight is None:
        raise ValueError(f"{prefix}sample_weight 없음. {_MISSING_COLS_MSG}")


def stratified_flat_indices(
    sample_type,
    *,
    k: float = FLAT_KEEP_K,
    random_state: int = 42,
) -> np.ndarray:
    """Return row indices keeping all pos/weak/crash and undersampling flat.

    Val sets should NOT call this. Train only.
    """
    types = np.asarray(sample_type).astype(str)
    n = len(types)
    all_idx = np.arange(n)

    is_pos = types == "pos"
    is_weak = types == "weak"
    is_crash = types == "crash"
    is_flat = types == "flat"

    # unknown/NaN type → treat as flat (undersample pool)
    is_known = is_pos | is_weak | is_crash | is_flat
    is_flat = is_flat | (~is_known)

    keep = is_pos | is_weak | is_crash
    n_pos = int(is_pos.sum())
    flat_idx = all_idx[is_flat]
    n_flat = len(flat_idx)

    if n_flat == 0:
        return all_idx[keep] if keep.any() else all_idx

    if n_pos <= 0:
        # no positive baseline → do not drop flats
        return all_idx

    n_keep_flat = min(n_flat, max(0, int(k * n_pos)))
    rng = np.random.RandomState(random_state)
    if n_keep_flat < n_flat:
        chosen_flat = rng.choice(flat_idx, size=n_keep_flat, replace=False)
    else:
        chosen_flat = flat_idx

    selected = np.concatenate([all_idx[keep], chosen_flat])
    selected.sort()
    return selected


def apply_stratified_flat_undersample(
    X,
    y,
    sample_type,
    sample_weight=None,
    *,
    k: float = FLAT_KEEP_K,
    random_state: int = 42,
):
    """Undersample flat rows; return aligned X, y, weight, type."""
    idx = stratified_flat_indices(sample_type, k=k, random_state=random_state)

    if isinstance(X, pd.DataFrame):
        X_out = X.iloc[idx].reset_index(drop=True)
    else:
        X_out = X[idx]

    y_arr = np.asarray(y)
    y_out = y_arr[idx]

    type_arr = np.asarray(sample_type)
    type_out = type_arr[idx]

    if sample_weight is None:
        w_out = None
    else:
        w_arr = np.asarray(sample_weight, dtype=np.float64)
        w_out = w_arr[idx]

    return X_out, y_out, w_out, type_out
