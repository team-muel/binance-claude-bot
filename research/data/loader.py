"""
로컬 OHLCV 데이터 로더.

Parquet 또는 CSV 형식의 OHLCV 데이터를 DataFrame으로 변환한다.
"""
import os

import pandas as pd


EXPECTED_INTERVALS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """OHLCV 데이터 품질을 검증하고 정규화한다."""
    required_numeric = ["open", "high", "low", "close", "volume"]

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df[required_numeric] = df[required_numeric].apply(pd.to_numeric, errors="coerce")

    if df["timestamp"].isna().any():
        raise ValueError("timestamp 컬럼에 파싱할 수 없는 값이 있습니다.")

    if df[required_numeric].isna().any().any():
        invalid_columns = [column for column in required_numeric if df[column].isna().any()]
        raise ValueError(f"OHLCV 컬럼에 결측 또는 비수치 값이 있습니다: {invalid_columns}")

    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"timestamp 중복이 {duplicate_count}개 있습니다.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp가 단조 증가하지 않습니다.")

    expected_delta = EXPECTED_INTERVALS.get(timeframe)
    if expected_delta is not None and len(df) > 1:
        deltas = df["timestamp"].diff().dropna()
        invalid_deltas = deltas[deltas != expected_delta]
        if not invalid_deltas.empty:
            first_bad_idx = int(invalid_deltas.index[0])
            previous_ts = df.loc[first_bad_idx - 1, "timestamp"]
            current_ts = df.loc[first_bad_idx, "timestamp"]
            raise ValueError(
                "예상한 봉 간격과 다른 구간이 있습니다: "
                f"{previous_ts} -> {current_ts} ({invalid_deltas.iloc[0]})"
            )

    return df


def load_ohlcv(
    symbol: str = "BTCUSDT",
    timeframe: str = "30m",
    data_dir: str = "research/data/raw",
) -> pd.DataFrame:
    """로컬 Parquet/CSV에서 OHLCV DataFrame을 로드한다."""
    parquet_path = os.path.join(data_dir, f"{symbol}_{timeframe}.parquet")
    csv_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        raise FileNotFoundError(
            f"No data file found for {symbol} {timeframe} in {data_dir}. "
            f"Run downloader.py first."
        )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return validate_ohlcv(df, timeframe)
