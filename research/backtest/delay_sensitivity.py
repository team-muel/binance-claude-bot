"""
Execution delay sensitivity 분석.

Signal 발생 시점 기준 +1, +2 bar 지연 진입 시
OOS 성능 변화를 측정한다.
"""
from typing import Any, Dict, List, cast

import pandas as pd

from research.backtest.engine import run_backtest
from research.backtest.metrics import TF_PERIODS_PER_YEAR, compute_metrics
from research.strategy.rules import generate_signals


def apply_signal_delay(
    signals_df: pd.DataFrame,
    delay_bars: int = 1,
) -> pd.DataFrame:
    """
    Signal 컬럼을 delay_bars만큼 지연시킨 DataFrame을 반환한다.

    Parameters
    ----------
    signals_df : DataFrame
        generate_signals()의 출력.
    delay_bars : int
        지연시킬 바 수.

    Returns
    -------
    DataFrame
        signal/exit 컬럼이 delay_bars만큼 shift된 DataFrame.
    """
    delayed = signals_df.copy()
    delayed["signal"] = delayed["signal"].shift(delay_bars).fillna(0).astype(int)
    delayed["exit_long"] = delayed["exit_long"].shift(delay_bars).fillna(False)
    delayed["exit_short"] = delayed["exit_short"].shift(delay_bars).fillna(False)
    return delayed


def delay_sensitivity_analysis(
    df: pd.DataFrame,
    params: Dict[str, Any],
    cost_config: Dict[str, Any],
    delays: List[int] | None = None,
    timeframe: str = "30m",
) -> List[Dict[str, Any]]:
    """
    여러 지연 수준에서 백테스트를 수행하고 성과를 비교한다.

    Parameters
    ----------
    df : DataFrame
        OHLCV 데이터.
    params : dict
        전략 파라미터.
    cost_config : dict
        commission_rate, slippage_rate, initial_capital, position_size_fraction
    delays : list[int]
        테스트할 지연 수준. 기본값 [0, 1, 2].
    timeframe : str
        데이터 타임프레임.

    Returns
    -------
    list[dict]
        각 지연 수준별 {delay_bars, sharpe, mdd, n_trades, ...}
    """
    if delays is None:
        delays = [0, 1, 2]

    ppy = TF_PERIODS_PER_YEAR.get(timeframe, TF_PERIODS_PER_YEAR["30m"])
    signals = generate_signals(df, params)
    results: List[Dict[str, Any]] = []

    for delay in delays:
        if delay > 0:
            delayed_signals = apply_signal_delay(signals, delay)
        else:
            delayed_signals = signals

        bt_result = run_backtest(delayed_signals, cost_config)
        metrics = cast(Dict[str, Any], compute_metrics(bt_result["equity_curve"], bt_result["trades"], ppy))
        metrics["delay_bars"] = delay
        results.append(metrics)

    return results
