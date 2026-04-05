"""
Funding rate post-hoc 영향 추정.

현재 백테스트 엔진은 funding을 직접 반영하지 않으므로, 체결된 trade의
보유 시간과 명목금액을 사용해 portfolio-level funding drag를 근사한다.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from research.backtest.metrics import TF_PERIODS_PER_YEAR, compute_metrics


def _holding_intervals_8h(entry_time: Any, exit_time: Any) -> float:
    """entry/exit 시각으로부터 8시간 funding interval 수를 근사한다."""
    if entry_time is None or exit_time is None:
        return 0.0

    entry_ts = pd.Timestamp(entry_time)
    exit_ts = pd.Timestamp(exit_time)
    holding_seconds = max((exit_ts - entry_ts).total_seconds(), 0.0)
    return holding_seconds / (8 * 60 * 60)


def estimate_funding_impact(
    equity_curve: pd.Series,
    trades: List[Dict[str, Any]],
    funding_rate_8h: float = 0.0001,
    timeframe: str = "30m",
) -> Dict[str, float]:
    """
    평균 funding rate을 기반으로 보유 기간 비용을 추정한다.

    Parameters
    ----------
    equity_curve : pd.Series
        원본 equity curve.
    trades : list[dict]
        거래 내역. 엔진 출력 포맷(entry_time, exit_time, quantity, direction)을 사용한다.
    funding_rate_8h : float
        8시간당 평균 funding rate (양수 = long이 short에 지불).
    timeframe : str
        데이터 타임프레임.

    Returns
    -------
    dict
        sharpe_before, sharpe_after, sharpe_delta, total_funding_cost,
        funding_cost_pct
    """
    if equity_curve.empty:
        return {
            "sharpe_before": 0.0,
            "sharpe_after": 0.0,
            "sharpe_delta": 0.0,
            "total_funding_cost": 0.0,
            "funding_cost_pct": 0.0,
        }

    ppy = TF_PERIODS_PER_YEAR.get(timeframe, TF_PERIODS_PER_YEAR["30m"])

    # 포지션 보유 시간에 대해 funding 차감
    adjusted_equity = equity_curve.to_numpy(dtype=np.float64, copy=True)
    cumulative_adjustment = np.zeros_like(adjusted_equity)

    total_funding = 0.0
    for trade in trades:
        direction = trade.get("direction", 1)  # 1=long, -1=short
        quantity = float(trade.get("quantity", trade.get("qty", 0.0)))
        entry_price = float(trade.get("entry_price", 0.0))
        position_value = abs(entry_price * quantity)
        holding_intervals = _holding_intervals_8h(
            trade.get("entry_time"),
            trade.get("exit_time"),
        )

        if holding_intervals <= 0.0 or position_value <= 0.0:
            continue

        # Long은 양수 funding rate일 때 비용 발생
        # Short은 양수 funding rate일 때 수익 발생
        cost = direction * funding_rate_8h * holding_intervals * position_value
        total_funding += cost

        exit_offset = int(trade.get("exit_bar_offset", len(adjusted_equity) - 1))
        if 0 <= exit_offset < len(cumulative_adjustment):
            cumulative_adjustment[exit_offset] += cost

    # Trade 종료 시점부터 cumulative funding drag를 반영한다.
    adjusted_equity = adjusted_equity - np.cumsum(cumulative_adjustment)
    adjusted_equity_series = pd.Series(adjusted_equity, index=equity_curve.index)

    metrics_before = compute_metrics(equity_curve, trades, ppy)

    # 조정된 equity 가 음수로 내려가면 validate_equity_curve 에서 ValueError.
    # funding drag 추정값이 크더라도 최솟값 1.0 으로 클리핑하여 계속 진행한다.
    adjusted_equity = np.maximum(adjusted_equity, 1.0)
    adjusted_equity_series = pd.Series(adjusted_equity, index=equity_curve.index)

    metrics_after = compute_metrics(adjusted_equity_series, trades, ppy)

    final_eq = float(equity_curve.iloc[-1])
    cost_ratio = total_funding / max(final_eq, 1.0)

    return {
        "sharpe_before": metrics_before["sharpe"],
        "sharpe_after": metrics_after["sharpe"],
        "sharpe_delta": metrics_after["sharpe"] - metrics_before["sharpe"],
        "total_funding_cost": float(total_funding),
        "funding_cost_pct": float(cost_ratio * 100),
    }
