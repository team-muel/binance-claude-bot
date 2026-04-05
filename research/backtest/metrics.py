"""
성과 지표 집계: Sharpe, MDD, Calmar, turnover 등.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# 타임프레임 → 연간 봉 수 매핑
TF_PERIODS_PER_YEAR = {
    "1m": 365.25 * 24 * 60,
    "5m": 365.25 * 24 * 12,
    "15m": 365.25 * 24 * 4,
    "30m": 365.25 * 24 * 2,
    "1h": 365.25 * 24,
    "4h": 365.25 * 6,
    "1d": 365.25,
}


def validate_equity_curve(equity_curve: pd.Series):
    """수익률 기반 메트릭이 성립하는지 검증한다."""
    if equity_curve.empty:
        return

    values = equity_curve.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("equity_curve contains non-finite values.")

    if np.any(values <= 0.0):
        raise ValueError("equity_curve must stay strictly positive for return-based metrics.")


def compute_metrics(
    equity_curve: pd.Series,
    trades: List[Dict[str, Any]],
    periods_per_year: float = TF_PERIODS_PER_YEAR["30m"],
) -> Dict[str, float]:
    """백테스트 결과로부터 성과 지표를 계산한다."""
    if equity_curve.empty:
        return {
            "sharpe": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
            "n_trades": len(trades),
            "turnover_annual": 0.0,
        }

    validate_equity_curve(equity_curve)
    returns = equity_curve.pct_change(fill_method=None).dropna()

    # Sharpe Ratio (annualized)
    if len(returns) == 0 or returns.std() < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    # Maximum Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    mdd = float(drawdown.min())

    # Calmar Ratio
    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
    if total_return < -1.0 - 1e-9:
        raise ValueError("total_return fell below -100%, which violates no-leverage portfolio accounting.")

    if mdd < -1.0 - 1e-9:
        raise ValueError("maximum drawdown fell below -100%, which violates no-leverage portfolio accounting.")

    if abs(mdd) < 1e-12:
        calmar = 0.0
    else:
        calmar = float(total_return / abs(mdd))

    # Turnover (평균 거래 빈도)
    n_bars = len(equity_curve)
    turnover = len(trades) / max(n_bars, 1) * periods_per_year if trades else 0.0

    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "total_return": total_return,
        "n_trades": len(trades),
        "turnover_annual": float(turnover),
    }
