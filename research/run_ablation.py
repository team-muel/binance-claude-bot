"""
Ablation study 실행기.

QA와 Classical SA의 핵심 하이퍼파라미터를 체계적으로 변경하며
BTC 30m walk-forward 실험을 반복한다.
"""
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from research.config import (
    PARAM_SPACE,
    ExperimentConfig,
    MIN_TRADES_FOR_VALID_OBJECTIVE,
)
from research.backtest.engine import run_backtest
from research.backtest.metrics import TF_PERIODS_PER_YEAR, compute_metrics
from research.data.loader import load_ohlcv
from research.evaluation.walk_forward import Fold, generate_folds, split_fold_data
from research.evaluation.stability import local_robustness, parameter_dispersion
from research.optimizers.quantum_annealing import QuantumAnnealingOptimizer
from research.optimizers.classical_sa import ClassicalSAOptimizer
from research.strategy.rules import generate_signals


# ---------------------------------------------------------------------------
# QA ablation 변이
# ---------------------------------------------------------------------------
QA_ABLATION_VARIANTS: List[Dict[str, Any]] = [
    # num_reads sweep
    {"name": "qa_reads_10",  "num_reads": 10,  "qubo_penalty": 10.0},
    {"name": "qa_reads_25",  "num_reads": 25,  "qubo_penalty": 10.0},
    {"name": "qa_reads_50",  "num_reads": 50,  "qubo_penalty": 10.0},  # default
    {"name": "qa_reads_100", "num_reads": 100, "qubo_penalty": 10.0},
    {"name": "qa_reads_200", "num_reads": 200, "qubo_penalty": 10.0},
    # penalty sweep
    {"name": "qa_penalty_1",  "num_reads": 50, "qubo_penalty": 1.0},
    {"name": "qa_penalty_5",  "num_reads": 50, "qubo_penalty": 5.0},
    {"name": "qa_penalty_10", "num_reads": 50, "qubo_penalty": 10.0},  # default
    {"name": "qa_penalty_20", "num_reads": 50, "qubo_penalty": 20.0},
    {"name": "qa_penalty_50", "num_reads": 50, "qubo_penalty": 50.0},
]

# ---------------------------------------------------------------------------
# SA ablation 변이
# ---------------------------------------------------------------------------
SA_ABLATION_VARIANTS: List[Dict[str, Any]] = [
    # temp_start sweep
    {"name": "sa_temp_1",  "temp_start": 1.0,  "temp_end": 0.01, "cooling_schedule": "exponential"},
    {"name": "sa_temp_5",  "temp_start": 5.0,  "temp_end": 0.01, "cooling_schedule": "exponential"},
    {"name": "sa_temp_10", "temp_start": 10.0, "temp_end": 0.01, "cooling_schedule": "exponential"},  # default
    {"name": "sa_temp_50", "temp_start": 50.0, "temp_end": 0.01, "cooling_schedule": "exponential"},
    # cooling schedule sweep
    {"name": "sa_cool_exp",  "temp_start": 10.0, "temp_end": 0.01, "cooling_schedule": "exponential"},  # default
    {"name": "sa_cool_lin",  "temp_start": 10.0, "temp_end": 0.01, "cooling_schedule": "linear"},
    {"name": "sa_cool_log",  "temp_start": 10.0, "temp_end": 0.01, "cooling_schedule": "logarithmic"},
]


def make_is_objective(is_df: pd.DataFrame, config: ExperimentConfig):
    """IS backtest objective function (with min_trades penalty)."""
    ppy = TF_PERIODS_PER_YEAR.get(config.timeframe, TF_PERIODS_PER_YEAR["30m"])
    cost_config: Dict[str, float] = {
        "commission_rate": config.cost.commission_rate,
        "slippage_rate": config.cost.slippage_rate,
        "initial_capital": config.backtest.initial_capital,
        "position_size_fraction": config.backtest.position_size_fraction,
    }

    def objective(params: Dict[str, Any]) -> float:
        signals = generate_signals(is_df, params)
        result = run_backtest(signals, cost_config)
        metrics = compute_metrics(result["equity_curve"], result["trades"], ppy)
        if metrics["n_trades"] < MIN_TRADES_FOR_VALID_OBJECTIVE:
            return -100.0
        return metrics["sharpe"]
    return objective


def make_oos_objective(oos_df: pd.DataFrame, config: ExperimentConfig):
    """OOS backtest objective function (no penalty)."""
    ppy = TF_PERIODS_PER_YEAR.get(config.timeframe, TF_PERIODS_PER_YEAR["30m"])
    cost_config: Dict[str, float] = {
        "commission_rate": config.cost.commission_rate,
        "slippage_rate": config.cost.slippage_rate,
        "initial_capital": config.backtest.initial_capital,
        "position_size_fraction": config.backtest.position_size_fraction,
    }

    def objective(params: Dict[str, Any]) -> float:
        signals = generate_signals(oos_df, params)
        result = run_backtest(signals, cost_config)
        metrics = compute_metrics(result["equity_curve"], result["trades"], ppy)
        return metrics["sharpe"]
    return objective


def run_single_ablation(
    variant: Dict[str, Any],
    optimizer_type: str,
    folds: List[Fold],
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """단일 ablation 변이에 대해 전체 fold를 실행한다."""
    name = variant["name"]
    print(f"  Running ablation: {name}")

    fold_results: List[Dict[str, Any]] = []
    for fold in folds:
        is_df, oos_df = split_fold_data(df, fold)
        is_obj = make_is_objective(is_df, config)
        oos_obj = make_oos_objective(oos_df, config)

        if optimizer_type == "qa":
            opt = QuantumAnnealingOptimizer(
                eval_budget=config.eval_budget,
                seed=config.seed,
                num_reads=variant["num_reads"],
                qubo_penalty=variant["qubo_penalty"],
            )
        else:
            opt = ClassicalSAOptimizer(
                eval_budget=config.eval_budget,
                seed=config.seed,
                temp_start=variant["temp_start"],
                temp_end=variant["temp_end"],
                cooling_schedule=variant["cooling_schedule"],
            )

        result = opt.optimize(is_obj, PARAM_SPACE)
        best_params = result["best_params"]
        is_sharpe = result["best_score"]

        if best_params is None:
            oos_sharpe = 0.0
        else:
            oos_sharpe = oos_obj(best_params)

        robustness = {}
        if best_params is not None:
            robustness = local_robustness(best_params, oos_obj, PARAM_SPACE)

        fold_results.append({
            "fold_id": fold.fold_id,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "best_params": best_params,
            "robustness": robustness,
        })

    oos_scores = [fr["oos_sharpe"] for fr in fold_results]
    all_params = [fr["best_params"] for fr in fold_results if fr["best_params"]]
    dispersion = parameter_dispersion(all_params) if len(all_params) >= 2 else 0.0

    degradations = [
        fr["robustness"].get("mean_degradation", 0)
        for fr in fold_results if fr["robustness"]
    ]

    return {
        "variant_name": name,
        "optimizer_type": optimizer_type,
        "variant_config": {k: v for k, v in variant.items() if k != "name"},
        "median_oos_sharpe": float(np.median(oos_scores)),
        "mean_oos_sharpe": float(np.mean(oos_scores)),
        "std_oos_sharpe": float(np.std(oos_scores, ddof=1)) if len(oos_scores) > 1 else 0.0,
        "param_dispersion": dispersion,
        "mean_degradation": float(np.mean(degradations)) if degradations else 0.0,
        "n_folds": len(folds),
        "fold_results": fold_results,
    }


def run_ablation(config: ExperimentConfig | None = None):
    """전체 ablation study를 실행한다."""
    if config is None:
        config = ExperimentConfig()

    print(f"=== Ablation Study: {config.symbol} {config.timeframe} ===")
    print(f"Budget: {config.eval_budget}, Seed: {config.seed}")

    df = load_ohlcv(config.symbol, config.timeframe)
    folds = generate_folds(df, config)
    print(f"Folds: {len(folds)}")

    if len(folds) < 6:
        print("ERROR: Need at least 6 folds for ablation. Aborting.")
        return

    all_results: Dict[str, List[Dict[str, Any]]] = {"qa": [], "sa": []}

    print("\n--- QA Ablation ---")
    for variant in QA_ABLATION_VARIANTS:
        result = run_single_ablation(variant, "qa", folds, df, config)
        all_results["qa"].append(result)
        print(f"    {variant['name']}: median_oos={result['median_oos_sharpe']:.3f}, "
              f"dispersion={result['param_dispersion']:.3f}")

    print("\n--- SA Ablation ---")
    for variant in SA_ABLATION_VARIANTS:
        result = run_single_ablation(variant, "sa", folds, df, config)
        all_results["sa"].append(result)
        print(f"    {variant['name']}: median_oos={result['median_oos_sharpe']:.3f}, "
              f"dispersion={result['param_dispersion']:.3f}")

    # 결과 저장
    artifact: Dict[str, Any] = {
        "schema_version": 5,
        "experiment_type": "ablation",
        "config": {
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "eval_budget": config.eval_budget,
            "seed": config.seed,
            "n_folds": len(folds),
        },
        "qa_ablation": all_results["qa"],
        "sa_ablation": all_results["sa"],
    }

    os.makedirs(config.artifacts_dir, exist_ok=True)
    path = os.path.join(
        config.artifacts_dir,
        f"ablation_{config.symbol}_{config.timeframe}_budget{config.eval_budget}_seed{config.seed}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nArtifact saved: {path}")


if __name__ == "__main__":
    run_ablation()
