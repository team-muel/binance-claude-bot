"""
Optimization landscape shift 분석.

대표 파라미터 축에 대한 IS objective heatmap을 생성하여
fold 간 최적점 이동을 시각화한다.
"""
import math
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np


def compute_2d_landscape(
    objective_fn: Callable[[Dict[str, Any]], float],
    param_space: Dict[str, list[Any]],
    axis_x: str,
    axis_y: str,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    2개 파라미터 축에 대한 objective landscape를 계산한다.

    나머지 파라미터는 fixed_params로 고정한다.
    fixed_params가 None이면 각 파라미터의 중간값을 사용한다.

    Parameters
    ----------
    objective_fn : callable
        params dict → float
    param_space : dict
        전체 파라미터 공간.
    axis_x, axis_y : str
        landscape를 그릴 2개 파라미터 이름.
    fixed_params : dict, optional
        나머지 파라미터 고정값.

    Returns
    -------
    dict
        x_values, y_values, scores (2D array), best_x, best_y, best_score
    """
    x_vals = param_space[axis_x]
    y_vals = param_space[axis_y]

    if fixed_params is None:
        fixed_params = {}
        for key, vals in param_space.items():
            if key not in (axis_x, axis_y):
                fixed_params[key] = vals[len(vals) // 2]

    scores = np.full((len(y_vals), len(x_vals)), np.nan)
    best_score = float("-inf")
    best_x = x_vals[0]
    best_y = y_vals[0]

    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            params = fixed_params.copy()
            params[axis_x] = x_val
            params[axis_y] = y_val

            # EMA 제약 체크
            if params.get("ema_fast", 0) >= params.get("ema_slow", float("inf")):
                scores[i, j] = np.nan
                continue

            score = objective_fn(params)
            scores[i, j] = score

            if score > best_score:
                best_score = score
                best_x = x_val
                best_y = y_val

    return {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "x_values": x_vals,
        "y_values": y_vals,
        "scores": scores,
        "best_x": best_x,
        "best_y": best_y,
        "best_score": best_score,
    }


def compute_landscape_shift(
    fold_landscapes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    여러 fold의 landscape에서 최적점 이동 통계를 계산한다.

    Parameters
    ----------
    fold_landscapes : list[dict]
        compute_2d_landscape()의 출력 리스트.

    Returns
    -------
    dict
        best_positions, centroid, max_displacement, mean_displacement
    """
    if not fold_landscapes:
        return {
            "best_positions": [],
            "centroid": [0.0, 0.0],
            "max_displacement": 0.0,
            "mean_displacement": 0.0,
        }

    positions: List[tuple[float, float]] = []
    for ls in fold_landscapes:
        x_vals = ls["x_values"]
        y_vals = ls["y_values"]
        # 정규화 (0-1 범위)
        x_norm = x_vals.index(ls["best_x"]) / max(len(x_vals) - 1, 1)
        y_norm = y_vals.index(ls["best_y"]) / max(len(y_vals) - 1, 1)
        positions.append((x_norm, y_norm))

    centroid_x = sum(pos[0] for pos in positions) / len(positions)
    centroid_y = sum(pos[1] for pos in positions) / len(positions)
    centroid = [centroid_x, centroid_y]
    displacements = [
        math.dist(position, (centroid_x, centroid_y))
        for position in positions
    ]

    return {
        "best_positions": positions,
        "centroid": centroid,
        "max_displacement": float(max(displacements)) if displacements else 0.0,
        "mean_displacement": float(sum(displacements) / len(displacements)) if displacements else 0.0,
    }


def rank_correlation_across_landscapes(
    fold_landscapes: List[Dict[str, Any]],
) -> float:
    """
    Fold 간 landscape 순위 상관을 계산한다.

    모든 fold 쌍에 대해 score grid의 Spearman rank correlation을 구하고
    평균을 반환한다. 낮은 값 = fold 간 landscape가 크게 변함.

    Returns
    -------
    float
        평균 pairwise Spearman rank correlation.
    """
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    n = len(fold_landscapes)
    if n < 2:
        return 1.0

    correlations: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            scores_i = fold_landscapes[i]["scores"].ravel()
            scores_j = fold_landscapes[j]["scores"].ravel()
            # NaN 제거 (EMA constraint 등)
            valid = np.isfinite(scores_i) & np.isfinite(scores_j)
            if valid.sum() < 3:
                continue
            corr_result = spearmanr(scores_i[valid], scores_j[valid])
            corr_value = cast(Any, getattr(corr_result, "correlation", None))
            if corr_value is None:
                corr_value = cast(Any, corr_result[0])
            corr = float(corr_value)
            if np.isfinite(corr):
                correlations.append(float(corr))

    return float(np.mean(correlations)) if correlations else 0.0
