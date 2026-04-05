"""
거래비용 모델: 수수료와 슬리피지.

첫 논문 범위에서는 funding 비용을 제외한다.
"""
from typing import Any, Dict


def apply_costs(
    raw_pnl: float,
    entry_price: float,
    exit_price: float,
    cost_config: Dict[str, Any],
    quantity: float = 1.0,
) -> float:
    """원시 PnL에서 수수료와 슬리피지를 차감한 순 PnL을 반환한다.

    engine.py의 인라인 비용 계산과 동일한 공식을 사용한다:
        cost = notional * per_side_rate  (진입) + notional * per_side_rate (청산)

    Parameters
    ----------
    raw_pnl : float
        gross PnL (fees 미반영).
    entry_price : float
        진입 가격.
    exit_price : float
        청산 가격.
    cost_config : dict
        commission_rate, slippage_rate.
    quantity : float
        포지션 수량. 기본값 1.0 (unit 기준).
    """
    commission = cost_config.get("commission_rate", 0.0004)
    slippage = cost_config.get("slippage_rate", 0.0001)
    per_side_rate = commission + slippage
    entry_cost = entry_price * quantity * per_side_rate
    exit_cost = exit_price * quantity * per_side_rate
    return raw_pnl - entry_cost - exit_cost
