"""
파라미터 공간 정의 및 이산화 유틸리티.

QUBO 인코딩과 classical optimizer 모두 동일한 이산 후보 집합을 사용하도록 보장한다.
"""
from itertools import product
from typing import Dict, List

from research.config import PARAM_SPACE


def get_param_space() -> Dict[str, list[int] | list[float]]:
    """현재 설정된 이산 파라미터 공간을 반환한다."""
    return PARAM_SPACE.copy()


def total_combinations() -> int:
    """유효한 총 조합 수를 반환한다 (EMA_fast < EMA_slow 제약 적용)."""
    count = 0
    for combo in product(*PARAM_SPACE.values()):
        p = dict(zip(PARAM_SPACE.keys(), combo))
        if p["ema_fast"] < p["ema_slow"]:
            count += 1
    return count


def sample_params(idx: int) -> Dict[str, int | float]:
    """인덱스로 파라미터 조합을 결정론적으로 선택한다."""
    keys = list(PARAM_SPACE.keys())
    values = list(PARAM_SPACE.values())
    all_valid = [
        dict(zip(keys, combo))
        for combo in product(*values)
        if combo[0] < combo[1]  # ema_fast < ema_slow
    ]
    return all_valid[idx % len(all_valid)]


def get_all_valid_params() -> List[Dict[str, int | float]]:
    """EMA_fast < EMA_slow 제약을 만족하는 모든 유효 파라미터 조합을 반환한다."""
    keys = list(PARAM_SPACE.keys())
    values = list(PARAM_SPACE.values())
    return [
        dict(zip(keys, combo))
        for combo in product(*values)
        if combo[0] < combo[1]
    ]
