"""
CSCV 기반 PBO (Probability of Backtest Overfitting).

Bailey, Borwein, Lopez de Prado, Zhu (2014):
"Pseudo-Mathematics and Financial Charlatanism"

Combinatorially Symmetric Cross-Validation (CSCV) 방법으로
엄밀한 PBO를 계산한다.
"""
from itertools import combinations
from typing import Any, Callable, Dict, List, cast

import numpy as np


def _split_matrix(
    returns_matrix: np.ndarray,
    n_partitions: int = 10,
) -> List[np.ndarray]:
    """
    수익률 행렬의 시간 축을 n_partitions개 서브셋으로 분할한다.

    Parameters
    ----------
    returns_matrix : np.ndarray
        shape (T, N) — T개 시점, N개 strategy (파라미터 조합).
    n_partitions : int
        분할 수 (짝수여야 함).

    Returns
    -------
    list[np.ndarray]
        각 서브셋의 시간 인덱스 배열.
    """
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even for CSCV.")
    T = returns_matrix.shape[0]
    indices = np.arange(T, dtype=np.int64)
    return cast(List[np.ndarray], list(np.array_split(indices, n_partitions)))


def compute_pbo_cscv(
    returns_matrix: np.ndarray,
    n_partitions: int = 10,
    metric_fn: Callable[[np.ndarray], float] | None = None,
) -> Dict[str, Any]:
    """
    CSCV 방법으로 PBO를 계산한다.

    Parameters
    ----------
    returns_matrix : np.ndarray
        shape (T, N) — T개 시점, N개 strategy 수익률.
        각 strategy는 서로 다른 파라미터 조합.
    n_partitions : int
        시간 축 분할 수 (짝수). C(S, S/2) 조합을 생성한다.
    metric_fn : callable, optional
        수익률 시계열 → 스칼라 성과지표. 기본값은 Sharpe ratio.

    Returns
    -------
    dict
        pbo: PBO 값 (0~1), n_combinations, logit_distribution
    """
    if metric_fn is None:
        def default_metric_fn(r: np.ndarray) -> float:
            if len(r) == 0 or np.std(r) < 1e-12:
                return 0.0
            return float(np.mean(r) / np.std(r))
        score_fn = default_metric_fn
    else:
        score_fn = metric_fn

    partitions = _split_matrix(returns_matrix, n_partitions)
    half = n_partitions // 2
    N = returns_matrix.shape[1]

    combos = list(combinations(range(n_partitions), half))
    n_overfit = 0
    logit_values: List[float] = []

    for combo_is in combos:
        combo_oos = tuple(i for i in range(n_partitions) if i not in combo_is)

        # IS / OOS 인덱스 합치기
        is_idx = np.concatenate([partitions[i] for i in combo_is])
        oos_idx = np.concatenate([partitions[i] for i in combo_oos])

        # 각 strategy의 IS/OOS 성과 계산
        is_perf = np.array([score_fn(returns_matrix[is_idx, j]) for j in range(N)])
        oos_perf = np.array([score_fn(returns_matrix[oos_idx, j]) for j in range(N)])

        # IS에서 best인 strategy의 OOS 순위 확인
        best_is_idx = int(np.argmax(is_perf))
        best_oos_score = oos_perf[best_is_idx]

        # OOS에서 중위수 이하면 overfitting
        oos_median = float(np.median(oos_perf))
        if best_oos_score < oos_median:
            n_overfit += 1

        # Logit: best-IS의 OOS 순위 비율
        rank = float(np.sum(oos_perf <= best_oos_score)) / N
        # Clamp to avoid log(0)
        rank = max(min(rank, 1 - 1e-10), 1e-10)
        logit = float(np.log(rank / (1 - rank)))
        logit_values.append(logit)

    pbo = n_overfit / len(combos) if combos else 0.0

    return {
        "pbo": pbo,
        "n_combinations": len(combos),
        "n_overfit": n_overfit,
        "logit_mean": float(np.mean(logit_values)) if logit_values else 0.0,
        "logit_std": float(np.std(logit_values)) if logit_values else 0.0,
    }
