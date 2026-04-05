"""
통계 검정: paired bootstrap, Wilcoxon signed-rank, budget sensitivity.
"""
from typing import Any, Callable, Dict, List

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
    statistic: str = "mean",
) -> Dict[str, float | str]:
    """
    Paired bootstrap test로 두 optimizer의 OOS score 차이를 검정한다.

    Returns
    -------
    dict
        statistic, observed_diff, ci_lower, ci_upper, p_value
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the same length.")
    if len(a) == 0:
        raise ValueError("paired_bootstrap_test requires at least one paired observation.")

    statistic_key = statistic.lower()
    if statistic_key not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic: {statistic}")

    boot_diffs: list[float] = []

    if statistic_key == "median":
        # primary endpoint 전용 방식:
        # fold index를 재표본추출한 뒤 median(a[idx]) - median(b[idx]) 를 계산한다.
        # 이는 논문의 "fold-전체 median OOS Sharpe 차이" 를 직접 bootstrap하는 올바른 방법이다.
        # (median(a-b) 와 median(a)-median(b) 는 일반적으로 다르며 부호가 반전될 수 있다.)
        observed_diff = float(np.median(a) - np.median(b))
        for _ in range(n_bootstrap):
            idx = rng.choice(len(a), size=len(a), replace=True)
            boot_diffs.append(float(np.median(a[idx]) - np.median(b[idx])))
    else:
        # mean의 경우 mean(a-b) == mean(a)-mean(b) 이므로 기존 방식과 동일하다.
        diffs = a - b
        observed_diff = float(np.mean(diffs))
        for _ in range(n_bootstrap):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_diffs.append(float(np.mean(diffs[idx])))

    boot_arr = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    # 양측 p-value: observed_diff 방향 기준으로 반대쪽 꼬리 비율 × 2
    p_value = float(np.mean(boot_arr <= 0)) * 2 if observed_diff > 0 else float(np.mean(boot_arr >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "statistic": statistic_key,
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """웈콕슨 signed-rank test를 수행한다."""
    if len(scores_a) < 6:
        return {"statistic": float("nan"), "p_value": float("nan"), "note": "n < 6"}
    try:
        result = stats.wilcoxon(scores_a, scores_b)  # type: ignore[no-untyped-call]
    except ValueError as exc:
        return {"statistic": float("nan"), "p_value": float("nan"), "note": str(exc)}
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}  # type: ignore[union-attr]


def budget_sensitivity(
    run_fn: Callable[[int], Dict[str, Any]],
    budgets: List[int],
) -> List[Dict[str, Any]]:
    """
    여러 탐색 예산 수준에서 실험을 반복하고 결과를 비교한다.

    Parameters
    ----------
    run_fn : callable
        budget (int) → dict (optimizer별 OOS scores)
    budgets : list[int]
        테스트할 budget 수준들

    Returns
    -------
    list[dict]
        각 budget 수준별 결과
    """
    results: List[Dict[str, Any]] = []
    for b in budgets:
        result = run_fn(b)
        result["budget"] = b
        results.append(result)
    return results


def cliffs_delta(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """
    Cliff's delta (비모수 effect size)를 계산한다.

    Parameters
    ----------
    scores_a, scores_b : list[float]
        비교할 두 그룹의 점수.

    Returns
    -------
    dict
        delta, interpretation (negligible/small/medium/large)
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        return {"delta": 0.0, "interpretation": "negligible"}

    # 모든 쌍에 대해 비교
    dominance = 0.0
    for ai in a:
        for bi in b:
            if ai > bi:
                dominance += 1.0
            elif ai < bi:
                dominance -= 1.0

    delta = dominance / (n_a * n_b)

    # 해석 (Vargha & Delaney 기준)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {"delta": float(delta), "interpretation": interpretation}


def holm_bonferroni_correction(
    p_values: List[float],
) -> List[float]:
    """
    Holm-Bonferroni stepdown correction을 적용한다.

    Parameters
    ----------
    p_values : list[float]
        원래 p-value들.

    Returns
    -------
    list[float]
        보정된 p-value들 (원래 순서 유지).
    """
    n = len(p_values)
    if n == 0:
        return []

    # (원래 인덱스, p-value) 쌍을 p-value 오름차순 정렬
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    corrected = [0.0] * n
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)

    return corrected
