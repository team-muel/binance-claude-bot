"""
과최적화 통제 지표: IS→OOS decay, PBO, Deflated Sharpe.
"""
from typing import Dict, List

import numpy as np


def is_oos_decay(is_sharpe: float, oos_sharpe: float) -> float:
    """In-sample 대비 OOS Sharpe 저하율을 반환한다. 0이면 저하 없음.

    is_sharpe == 0인 경우, OOS가 음수면 양수 (+overfitting 신호),
    OOS가 양수면 0.0을 반환 (야간 IS=0 시작점에서 개선).
    """
    if is_sharpe == 0:
        return float(-oos_sharpe) if oos_sharpe < 0 else 0.0
    return (is_sharpe - oos_sharpe) / abs(is_sharpe)


def probability_of_backtest_overfitting(
    is_scores: List[float],
    oos_scores: List[float],
) -> float:
    """
    Bailey et al. (2014) 방식의 단순 PBO proxy.

    IS에서 가장 좋았던 선택점이 OOS에서 중위수 아래로 떨어지면
    overfitting 징후로 본다. 반환값은 0.0 또는 1.0이다.
    """
    if not is_scores or not oos_scores:
        return 0.0

    if len(is_scores) != len(oos_scores):
        raise ValueError("is_scores와 oos_scores의 길이는 같아야 합니다.")

    oos_median = float(np.median(oos_scores))
    best_is_idx = int(np.argmax(is_scores))
    best_oos = float(oos_scores[best_is_idx])
    return float(best_oos < oos_median)


def deflated_sharpe_proxy(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    mean_sharpe: float = 0.0,
    std_sharpe: float = 1.0,
) -> float:
    """
    Harvey & Liu (2015) 형태를 단순화한 Deflated Sharpe proxy.

    artifact 단계에서 과장된 해석을 피하기 위해 proxy로만 사용한다.
    """
    if n_trials <= 1 or n_obs <= 1:
        return observed_sharpe
    # Expected max Sharpe under null (Euler-Mascheroni corrected)
    z = np.sqrt(2 * np.log(n_trials))
    expected_max_z = z - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * z)
    expected_max = mean_sharpe + std_sharpe * expected_max_z * (1 / np.sqrt(n_obs))
    # 보정된 Sharpe
    deflated = observed_sharpe - expected_max
    return float(deflated)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    mean_sharpe: float = 0.0,
    std_sharpe: float = 1.0,
) -> float:
    """Backward-compatible alias for the current proxy implementation."""
    return deflated_sharpe_proxy(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_obs=n_obs,
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
    )


def is_oos_correlation(
    is_scores: List[float],
    oos_scores: List[float],
) -> Dict[str, float]:
    """
    IS score와 OOS score의 Spearman rank 상관계수를 계산한다.

    높은 IS-OOS 상관 = optimizer가 일반화 가능한 영역을 찾음.
    낮은 IS-OOS 상관 = IS 과최적화 경향.

    Parameters
    ----------
    is_scores, oos_scores : list[float]
        Fold별 IS/OOS 점수 (동일 길이).

    Returns
    -------
    dict
        spearman_r, spearman_p, pearson_r, n_obs
    """
    from scipy.stats import spearmanr, pearsonr  # type: ignore[import-untyped]

    if len(is_scores) != len(oos_scores):
        raise ValueError("is_scores와 oos_scores의 길이가 같아야 합니다.")
    if len(is_scores) < 3:
        return {
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "n_obs": float(len(is_scores)),
        }

    is_arr = np.array(is_scores)
    oos_arr = np.array(oos_scores)

    sp_r, sp_p = spearmanr(is_arr, oos_arr)
    pe_r, _ = pearsonr(is_arr, oos_arr)

    return {
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "pearson_r": float(pe_r),
        "n_obs": float(len(is_scores)),
    }
