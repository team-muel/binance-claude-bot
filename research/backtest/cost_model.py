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
) -> float:
    """원시 PnL에서 수수료와 슬리피지를 차감한 순 PnL을 반환한다."""
    commission = cost_config.get("commission_rate", 0.0004)
    slippage = cost_config.get("slippage_rate", 0.0001)

    # 1-unit 포지션 기준 진입/청산 각각의 비용을 더한다.
    per_side_rate = commission + slippage
    cost = entry_price * per_side_rate + exit_price * per_side_rate

    return raw_pnl - cost
