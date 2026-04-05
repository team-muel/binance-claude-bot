"""
Parameter stability 측정.

1. fold 간 parameter dispersion: 선택된 파라미터 벡터들의 정규화 거리
2. local robustness: 최적점 주변 one-step perturbation에 따른 OOS 성과 하락
"""
from typing import Any, Callable, Dict, List

import numpy as np

from research.config import PARAM_SPACE


def normalize_params(params: Dict[str, Any]) -> np.ndarray:
    """파라미터를 0–1 범위로 정규화한 벡터로 변환한다."""
    vec: list[float] = []
    for key in sorted(PARAM_SPACE.keys()):
        values = PARAM_SPACE[key]
        val = params[key]
        if len(values) <= 1:
            vec.append(0.0)
        else:
            idx = values.index(val)
            vec.append(idx / (len(values) - 1))
    return np.array(vec)


def parameter_dispersion(fold_params: List[Dict[str, Any]]) -> float:
    """fold별 선택 파라미터 벡터들의 평균 쌍별 Euclidean 거리를 반환한다."""
    if len(fold_params) < 2:
        return 0.0
    vecs = [normalize_params(p) for p in fold_params]
    dists: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.mean(dists))


def local_robustness(
    best_params: Dict[str, Any],
    oos_objective_fn: Callable[[Dict[str, Any]], float],
    param_space: Dict[str, list[Any]] | None = None,
) -> Dict[str, float]:
    """
    최적 파라미터 주변에서 한 칸씩 흔들어 OOS 성과 하락을 측정한다.

    Returns
    -------
    dict
        mean_signed_delta, mean_degradation, worst_degradation,
        mean_improvement, n_neighbors
    """
    if param_space is None:
        param_space = PARAM_SPACE

    base_score = oos_objective_fn(best_params)
    signed_deltas: list[float] = []

    for key, values in param_space.items():
        current_val = best_params[key]
        if current_val not in values:
            continue
        idx = values.index(current_val)
        for neighbor_idx in [idx - 1, idx + 1]:
            if 0 <= neighbor_idx < len(values):
                neighbor = best_params.copy()
                neighbor[key] = values[neighbor_idx]
                # EMA 제약 체크
                if neighbor.get("ema_fast", 0) >= neighbor.get("ema_slow", float("inf")):
                    continue
                neighbor_score = oos_objective_fn(neighbor)
                signed_deltas.append(base_score - neighbor_score)

    if not signed_deltas:
        return {
            "mean_signed_delta": 0.0,
            "mean_degradation": 0.0,
            "worst_degradation": 0.0,
            "mean_improvement": 0.0,
            "n_neighbors": 0,
        }

    degradations = [max(delta, 0.0) for delta in signed_deltas]
    improvements = [max(-delta, 0.0) for delta in signed_deltas]

    return {
        "mean_signed_delta": float(np.mean(signed_deltas)),
        "mean_degradation": float(np.mean(degradations)),
        "worst_degradation": float(np.max(degradations)),
        "mean_improvement": float(np.mean(improvements)),
        "n_neighbors": len(signed_deltas),
    }
