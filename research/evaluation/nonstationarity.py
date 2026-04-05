"""
비정상성 정량화 및 시장 국면 분류.

Rolling realized volatility를 기준으로 각 OOS fold에 regime label을 부여한다.
"""
from typing import Any, Dict, List, Literal, Tuple, cast

import numpy as np
import pandas as pd

from research.evaluation.walk_forward import Fold

RegimeLabel = Literal["high", "medium", "low"]


def _coerce_score_mapping(scores_obj: object) -> Dict[int, float]:
    """fold_id -> score 형태로 입력을 정규화한다."""
    score_by_fold: Dict[int, float] = {}

    if isinstance(scores_obj, dict):
        raw_mapping = cast(Dict[Any, Any], scores_obj)
        for fold_id, score in raw_mapping.items():
            score_by_fold[int(fold_id)] = float(score)
        return score_by_fold

    if isinstance(scores_obj, list):
        raw_list = cast(List[Any], scores_obj)
        if raw_list and isinstance(raw_list[0], dict):
            for raw_item in raw_list:
                if not isinstance(raw_item, dict):
                    continue
                if "fold_id" not in raw_item or "oos_sharpe" not in raw_item:
                    continue
                item = cast(Dict[str, Any], raw_item)
                score_by_fold[int(item["fold_id"])] = float(item["oos_sharpe"])
            return score_by_fold

        for fold_id, score in enumerate(raw_list):
            score_by_fold[fold_id] = float(score)

    return score_by_fold


def rolling_realized_volatility(
    df: pd.DataFrame,
    lookback_days: int = 90,
    timeframe: str = "30m",
) -> pd.Series:
    """
    Rolling realized volatility (log return의 표준편차, 연율화)를 계산한다.

    Parameters
    ----------
    df : DataFrame
        OHLCV 데이터. 'close'와 'timestamp' 컬럼 필수.
    lookback_days : int
        Rolling window 일수.
    timeframe : str
        데이터 타임프레임 (annualization factor 결정용).

    Returns
    -------
    pd.Series
        timestamp 인덱스, 연율화 실현 변동성 값.
    """
    bars_per_day = {
        "1m": 1440, "5m": 288, "15m": 96,
        "30m": 48, "1h": 24, "4h": 6, "1d": 1,
    }
    bpd = bars_per_day.get(timeframe, 48)
    window = lookback_days * bpd

    close = df["close"]
    log_ret: pd.Series = close.div(close.shift(1)).apply(np.log)
    rolling_std: pd.Series = log_ret.rolling(window=window, min_periods=max(window // 2, 1)).std()
    annualized: pd.Series = rolling_std * np.sqrt(bpd * 365.25)

    result = annualized.copy()
    result.index = df["timestamp"]
    return result


def compute_regime_quantiles(
    vol_series: pd.Series,
    quantiles: Tuple[float, float] = (0.33, 0.67),
) -> Tuple[float, float]:
    """전체 기간 변동성에서 regime 경계 quantile 값을 계산한다."""
    clean = vol_series.dropna()
    q_low = float(clean.quantile(quantiles[0]))
    q_high = float(clean.quantile(quantiles[1]))
    return q_low, q_high


def assign_fold_regime(
    df: pd.DataFrame,
    fold: Fold,
    vol_series: pd.Series,
    q_low: float,
    q_high: float,
) -> Dict[str, object]:
    """
    OOS fold의 평균 실현 변동성을 기준으로 regime label을 부여한다.

    Returns
    -------
    dict
        regime_label, mean_vol, q_low, q_high
    """
    mask = (vol_series.index >= fold.oos_start) & (vol_series.index < fold.oos_end)
    oos_vol = vol_series[mask]

    if oos_vol.empty:
        mean_vol = float("nan")
        label: RegimeLabel = "medium"
    else:
        mean_vol = float(oos_vol.mean())
        if mean_vol > q_high:
            label = "high"
        elif mean_vol < q_low:
            label = "low"
        else:
            label = "medium"

    return {
        "regime_label": label,
        "mean_oos_vol": mean_vol,
        "q_low": q_low,
        "q_high": q_high,
    }


def label_all_folds(
    df: pd.DataFrame,
    folds: List[Fold],
    lookback_days: int = 90,
    timeframe: str = "30m",
    quantiles: Tuple[float, float] = (0.33, 0.67),
) -> List[Dict[str, object]]:
    """
    모든 fold에 regime label을 부여한다.

    Returns
    -------
    list[dict]
        fold별 {fold_id, regime_label, mean_oos_vol, q_low, q_high}
    """
    vol_series = rolling_realized_volatility(df, lookback_days, timeframe)
    q_low, q_high = compute_regime_quantiles(vol_series, quantiles)

    results: List[Dict[str, object]] = []
    for fold in folds:
        info = assign_fold_regime(df, fold, vol_series, q_low, q_high)
        info["fold_id"] = fold.fold_id
        results.append(info)
    return results


def regime_conditional_summary(
    fold_regimes: List[Dict[str, object]],
    fold_oos_scores: Dict[str, object],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Regime별 optimizer OOS Sharpe 요약 통계를 계산한다.

    Parameters
    ----------
    fold_regimes : list[dict]
        label_all_folds()의 출력.
    fold_oos_scores : dict
        optimizer_name →
        1) list of OOS Sharpe (legacy: fold_id와 순서 동일)
        2) list of fold result dicts (권장)
        3) dict[fold_id, score]

    Returns
    -------
    dict
        regime → optimizer → {median, mean, std, n_folds}
    """
    regime_indices: Dict[str, List[int]] = {"high": [], "medium": [], "low": []}
    for info in fold_regimes:
        label = str(info["regime_label"])
        idx = int(cast(Any, info["fold_id"]))
        if label in regime_indices:
            regime_indices[label].append(idx)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for regime, indices in regime_indices.items():
        if not indices:
            continue
        summary[regime] = {}
        for opt_name, scores_obj in fold_oos_scores.items():
            score_by_fold = _coerce_score_mapping(scores_obj)
            if not score_by_fold:
                continue
            regime_scores = [score_by_fold[i] for i in indices if i in score_by_fold]
            if not regime_scores:
                continue
            arr = np.array(regime_scores)
            summary[regime][opt_name] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n_folds": len(arr),
            }
    return summary
