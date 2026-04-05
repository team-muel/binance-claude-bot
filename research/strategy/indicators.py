"""
기술지표 계산: EMA, RSI, ATR.

pandas 기반으로 구현하며, 결정론적 재현성을 보장한다.
"""
import pandas as pd


def ema(series: pd.Series, period: int | float) -> pd.Series:
    """지수이동평균."""
    return series.ewm(span=int(period), adjust=False).mean()


def rsi(series: pd.Series, period: int | float = 14) -> pd.Series:
    """Relative Strength Index."""
    period = int(period)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # avg_loss == 0 인 경우 RSI = 100 (완전 상승 구간), 0 으로 나누기 방지
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100 - (100 / (1 + rs.fillna(float("inf"))))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int | float = 14) -> pd.Series:
    """Average True Range."""
    period = int(period)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
