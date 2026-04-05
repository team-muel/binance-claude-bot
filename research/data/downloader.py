"""
Binance OHLCV 데이터 다운로드.

ccxt를 사용해 Binance Futures OHLCV 캔들 데이터를 다운로드하고
로컬 Parquet 파일로 저장한다.
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Any, List, cast

import ccxt  # type: ignore[import-untyped]
import pandas as pd


def download_ohlcv(
    symbol: str = "BTCUSDT",
    timeframe: str = "30m",
    since: str = "2020-01-01",
    output_dir: str = "research/data/raw",
) -> str:
    """Binance Futures OHLCV를 다운로드해 Parquet으로 저장한다."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    since_ts: int = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: List[List[Any]] = []
    limit = 1500

    print(f"Downloading {symbol} {timeframe} since {since} ...")
    while True:
        candles = cast(List[List[Any]], exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit))  # type: ignore[no-untyped-call]
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = int(candles[-1][0])
        if last_ts <= since_ts:
            break  # 진행이 없으면 중단
        since_ts = last_ts + 1
        print(f"  {len(all_candles):>7,} bars collected (last: {candles[-1][0]})", end="\r")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol}_{timeframe}.parquet"
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} rows → {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="30m")
    parser.add_argument("--since", default="2020-01-01")
    parser.add_argument("--output-dir", default="research/data/raw")
    args = parser.parse_args()
    download_ohlcv(args.symbol, args.timeframe, args.since, args.output_dir)
