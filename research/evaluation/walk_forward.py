"""
Rolling Walk-Forward fold 생성 및 실행.

각 fold에서 in-sample 구간으로 optimizer를 실행하고,
strict OOS 구간에서 성과를 평가한다.
"""
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from research.config import ExperimentConfig


@dataclass
class Fold:
    """단일 walk-forward fold."""
    fold_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def generate_folds(df: pd.DataFrame, config: ExperimentConfig) -> List[Fold]:
    """Rolling window 방식으로 fold 목록을 생성한다."""
    wf = config.walk_forward
    ts = df["timestamp"]
    data_start = ts.iloc[0]
    data_end = ts.iloc[-1]

    folds: List[Fold] = []
    fold_id = 0
    is_start = data_start

    while True:
        is_end = is_start + pd.DateOffset(months=wf.is_window_months)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=wf.oos_window_months)

        if oos_end > data_end:
            break

        folds.append(Fold(
            fold_id=fold_id,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
        ))

        fold_id += 1
        is_start = is_start + pd.DateOffset(months=wf.step_months)

    return folds


def split_fold_data(
    df: pd.DataFrame, fold: Fold
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """fold 기준으로 in-sample과 out-of-sample DataFrame을 분리한다."""
    ts = df["timestamp"]
    is_df = df[(ts >= fold.is_start) & (ts < fold.is_end)].copy()
    oos_df = df[(ts >= fold.oos_start) & (ts < fold.oos_end)].copy()
    return is_df, oos_df
