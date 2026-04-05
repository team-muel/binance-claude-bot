"""
Convergence trace 분석.

각 optimizer의 eval_count vs best_so_far 기록을 분석하여
탐색 효율과 수렴 행동을 정량화한다.
"""
from typing import Any, Dict, List

import numpy as np


def extract_convergence_curve(
    history: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Optimizer history에서 cumulative best score curve를 추출한다.

    Parameters
    ----------
    history : list[dict]
        각 원소에 'score' 키가 있어야 한다.

    Returns
    -------
    np.ndarray
        shape (n_evals,) — 각 eval 시점까지의 best score.
    """
    if not history:
        return np.array([])

    scores = [h["score"] for h in history]
    best_so_far = np.maximum.accumulate(scores)
    return best_so_far


def convergence_statistics(
    curve: np.ndarray,
) -> Dict[str, float]:
    """
    단일 convergence curve의 요약 통계를 계산한다.

    Returns
    -------
    dict
        final_score, half_score_eval, last100_improvement, area_under_curve
    """
    if len(curve) == 0:
        return {
            "final_score": float("nan"),
            "half_score_eval": float("nan"),
            "last100_improvement": float("nan"),
            "area_under_curve": float("nan"),
        }

    initial = float(curve[0])
    final = float(curve[-1])

    # 절대값이 아니라 "초기값에서 최종값으로 이동한 개선폭의 50%"에 도달하는 시점.
    # 이렇게 해야 음수 Sharpe 구간에서도 해석 가능하다.
    threshold = initial + 0.5 * (final - initial)
    if final >= initial:
        reached = np.where(curve >= threshold)[0]
    else:
        reached = np.where(curve <= threshold)[0]
    half_eval = int(reached[0]) + 1 if len(reached) > 0 else len(curve)

    # 마지막 100 evals에서의 improvement
    tail = min(100, len(curve))
    last100_impr = float(curve[-1] - curve[-tail])

    # Area under curve (정규화)
    auc = float(np.sum(curve) / max(len(curve), 1))

    return {
        "final_score": final,
        "half_score_eval": float(half_eval),
        "last100_improvement": last100_impr,
        "area_under_curve": auc,
    }


def aggregate_convergence_across_folds(
    fold_curves: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    여러 fold의 convergence curve를 집계한다.

    Returns
    -------
    dict
        mean, stderr — shape (max_len,)
        각 fold의 curve 길이가 다를 수 있으므로 최소 길이로 잘라서 집계.
    """
    if not fold_curves or all(len(c) == 0 for c in fold_curves):
        return {"mean": np.array([]), "stderr": np.array([])}

    non_empty = [c for c in fold_curves if len(c) > 0]
    min_len = min(len(c) for c in non_empty)
    trimmed = np.array([c[:min_len] for c in non_empty])

    mean = trimmed.mean(axis=0)
    stderr = trimmed.std(axis=0, ddof=1) / np.sqrt(len(non_empty)) if len(non_empty) > 1 else np.zeros(min_len)

    return {"mean": mean, "stderr": stderr}
