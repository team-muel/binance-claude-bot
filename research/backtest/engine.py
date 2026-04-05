"""
결정론적 OHLCV 리플레이 백테스트 엔진.

시그널 DataFrame을 받아 포지션을 시뮬레이션하고 거래 기록을 반환한다.
numpy 배열 직접 접근으로 최적화 (iterrows 대비 ~50-100x).
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def run_backtest(
    signals_df: pd.DataFrame,
    cost_config: Dict[str, Any],
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    시그널 기반 백테스트를 실행한다.

    Parameters
    ----------
    signals_df : pd.DataFrame
        generate_signals()의 출력. signal, exit_long, exit_short,
        sl_distance, tp_distance 컬럼 필요.
    cost_config : dict
        commission_rate, slippage_rate,
        initial_capital, position_size_fraction(optional)
    initial_capital : float
        초기 자본금

    Returns
    -------
    dict
        equity_curve (pd.Series), trades (list[dict]), final_equity (float)
    """
    if signals_df.empty:
        return {
            "equity_curve": pd.Series(dtype=float),
            "trades": [],
            "final_equity": initial_capital,
        }

    # numpy 배열로 변환 — iterrows 대비 ~50-100x 빠름
    close = signals_df["close"].values.astype(np.float64)
    high = signals_df["high"].values.astype(np.float64)
    low = signals_df["low"].values.astype(np.float64)
    signal = signals_df["signal"].values
    exit_long = signals_df["exit_long"].values
    exit_short = signals_df["exit_short"].values
    sl_dist = signals_df["sl_distance"].values.astype(np.float64)
    tp_dist = signals_df["tp_distance"].values.astype(np.float64)
    timestamps = signals_df["timestamp"].values

    n = len(close)
    last_idx = n - 1
    per_side_rate = cost_config.get("commission_rate", 0.0004) + cost_config.get("slippage_rate", 0.0001)
    position_size_fraction = float(cost_config.get("position_size_fraction", 1.0))
    initial_capital = float(cost_config.get("initial_capital", initial_capital))

    if not 0.0 < position_size_fraction <= 1.0:
        raise ValueError("position_size_fraction must be in the interval (0, 1].")

    cash = initial_capital
    position = 0  # +1 long, -1 short, 0 flat
    position_qty = 0.0
    entry_price = 0.0
    entry_time_val = None
    entry_bar_offset = -1
    entry_fee = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trades: List[Dict[str, Any]] = []
    equity = np.empty(n, dtype=np.float64)

    def close_position(exit_price: float, exit_reason: str, exit_time: Any, exit_bar_offset: int):
        nonlocal cash, position, position_qty, entry_price, entry_time_val, entry_bar_offset, entry_fee, sl_price, tp_price

        qty = abs(position_qty)
        if qty <= 0.0:
            return

        exit_notional = qty * exit_price
        exit_fee = exit_notional * per_side_rate
        gross_pnl = position * qty * (exit_price - entry_price)

        if position == 1:
            cash += exit_notional - exit_fee
        else:
            cash -= exit_notional + exit_fee

        pnl = gross_pnl - entry_fee - exit_fee
        trades.append({
            "entry_time": entry_time_val,
            "exit_time": exit_time,
            "entry_bar_offset": entry_bar_offset,
            "exit_bar_offset": exit_bar_offset,
            "direction": position,
            "quantity": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "pnl": pnl,
            "exit_reason": exit_reason,
        })

        position = 0
        position_qty = 0.0
        entry_price = 0.0
        entry_time_val = None
        entry_bar_offset = -1
        entry_fee = 0.0
        sl_price = 0.0
        tp_price = 0.0

    for i in range(n):
        exited_this_bar = False

        if position != 0:
            lo = low[i]
            hi = high[i]
            hit_sl = (position == 1 and lo <= sl_price) or \
                     (position == -1 and hi >= sl_price)
            hit_tp = (position == 1 and hi >= tp_price) or \
                     (position == -1 and lo <= tp_price)

            if hit_sl:
                close_position(sl_price, "sl", timestamps[i], i)
                exited_this_bar = True
            elif hit_tp:
                close_position(tp_price, "tp", timestamps[i], i)
                exited_this_bar = True
            else:
                exit_by_rule = (position == 1 and exit_long[i]) or \
                               (position == -1 and exit_short[i])
                if exit_by_rule or i == last_idx:
                    reason = "rule" if exit_by_rule else "eod"
                    close_position(close[i], reason, timestamps[i], i)
                    exited_this_bar = True

        if position == 0 and not exited_this_bar and signal[i] != 0:
            entry_price = close[i]
            target_notional = cash * position_size_fraction
            direction = int(signal[i])

            if target_notional > 0.0 and entry_price > 0.0:
                if direction == 1:
                    qty = target_notional / (entry_price * (1.0 + per_side_rate))
                    entry_notional = qty * entry_price
                    entry_fee = entry_notional * per_side_rate
                    cash -= entry_notional + entry_fee
                    position_qty = qty
                else:
                    qty = target_notional / entry_price
                    entry_notional = qty * entry_price
                    entry_fee = entry_notional * per_side_rate
                    cash += entry_notional - entry_fee
                    position_qty = -qty

                position = direction
                entry_time_val = timestamps[i]
                entry_bar_offset = i
                if position == 1:
                    sl_price = entry_price - sl_dist[i]
                    tp_price = entry_price + tp_dist[i]
                else:
                    sl_price = entry_price + sl_dist[i]
                    tp_price = entry_price - tp_dist[i]

        equity[i] = cash + position_qty * close[i]

    return {
        "equity_curve": pd.Series(equity, index=signals_df.index),
        "trades": trades,
        "final_equity": float(equity[-1]),
    }
