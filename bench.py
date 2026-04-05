"""Quick benchmark for optimized pipeline."""
import time
from dataclasses import asdict
from research.data.loader import load_ohlcv
from research.strategy.rules import generate_signals
from research.backtest.engine import run_backtest
from research.backtest.metrics import compute_metrics
from research.config import CostConfig

df = load_ohlcv("BTCUSDT", "30m")
is_df = df.iloc[:26000]
cost = asdict(CostConfig())
params = {
    "ema_fast": 10, "ema_slow": 60, "rsi_period": 14,
    "rsi_entry": 35, "rsi_short_entry": 65,
    "atr_period": 14, "sl_multiplier": 2.0, "tp_multiplier": 3.0,
}

# Warm up
signals = generate_signals(is_df, params)
bt = run_backtest(signals, cost)
m = compute_metrics(bt["equity_curve"], bt["trades"])

# Benchmark 100 iterations
t0 = time.perf_counter()
for _ in range(100):
    signals = generate_signals(is_df, params)
    bt = run_backtest(signals, cost)
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"100 iterations: {elapsed:.2f}s  ({elapsed/100*1000:.1f}ms per eval)")
print(f"Estimated 500 evals: {elapsed/100*500:.1f}s")
print(f"Estimated full (47500 evals): {elapsed/100*47500:.0f}s = {elapsed/100*47500/60:.1f}min")
print(f"Trades: {len(bt['trades'])}, Sharpe: {m['sharpe']:.3f}")
print(f"Exit reasons: {set(t['exit_reason'] for t in bt['trades'])}")
