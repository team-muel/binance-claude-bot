"""
EMA-RSI-ATR 규칙전략 진입/청산 규칙.

파라미터는 외부에서 dict로 주입받고, 이 모듈은 규칙 로직만 담당한다.
"""
from typing import Dict

import pandas as pd

from research.strategy.indicators import ema, rsi, atr


def generate_signals(df: pd.DataFrame, params: Dict[str, int | float]) -> pd.DataFrame:
    """
    OHLCV DataFrame에 규칙전략 시그널 컬럼을 추가한다.

    Parameters
    ----------
    df : pd.DataFrame
        columns: timestamp, open, high, low, close, volume
    params : dict
        ema_fast, ema_slow, rsi_period, rsi_entry, rsi_short_entry,
        atr_period, sl_multiplier, tp_multiplier

    Returns
    -------
    pd.DataFrame
        원본 + ema_fast_val, ema_slow_val, rsi_val, atr_val,
        signal (+1 long / -1 short / 0 neutral),
        exit_long, exit_short, sl_distance, tp_distance
    """
    close = df["close"]
    ema_fast_val = ema(close, params["ema_fast"])
    ema_slow_val = ema(close, params["ema_slow"])
    rsi_val = rsi(close, params["rsi_period"])
    atr_val = atr(df["high"], df["low"], close, params["atr_period"])

    # 진입 조건
    fast_above = ema_fast_val > ema_slow_val
    long_entry = fast_above & (rsi_val < params["rsi_entry"])
    short_entry = (~fast_above) & (rsi_val > params["rsi_short_entry"])

    # 청산 조건: 반대 크로스 또는 RSI 역전
    long_exit = (~fast_above) | (rsi_val > params["rsi_short_entry"])
    short_exit = fast_above | (rsi_val < params["rsi_entry"])

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[long_entry] = 1
    signal[short_entry] = -1

    # SL/TP 거리
    sl_distance = atr_val * params["sl_multiplier"]
    tp_distance = atr_val * params["tp_multiplier"]

    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["signal"] = signal.values
    out["exit_long"] = long_exit.values
    out["exit_short"] = short_exit.values
    out["sl_distance"] = sl_distance.values
    out["tp_distance"] = tp_distance.values

    return out
