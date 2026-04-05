"""
실험 진입점.

Walk-forward fold를 생성하고, 각 fold에서 모든 optimizer를 동일 budget으로 실행한 뒤
OOS 성과와 안정성 지표를 비교·저장한다.
"""
import io
import json
import os
import platform
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Type

import numpy as np
import pandas as pd

from research.config import ExperimentConfig, GRID_SAMPLING_STRATEGY, MIN_TRADES_FOR_VALID_OBJECTIVE, PARAM_SPACE
from research.data.loader import load_ohlcv
from research.strategy.rules import generate_signals
from research.backtest.engine import run_backtest
from research.backtest.metrics import compute_metrics
from research.evaluation.walk_forward import generate_folds, split_fold_data
from research.evaluation.stability import local_robustness, parameter_dispersion
from research.evaluation.overfitting import (
    deflated_sharpe_proxy,
    is_oos_decay,
    is_oos_correlation,
    probability_of_backtest_overfitting,
)
from research.evaluation.statistics import (
    cliffs_delta,
    holm_bonferroni_correction,
    paired_bootstrap_test,
    wilcoxon_test,
)
from research.evaluation.nonstationarity import label_all_folds, regime_conditional_summary
from research.evaluation.convergence import extract_convergence_curve, convergence_statistics

from research.optimizers.grid_search import GridSearchOptimizer
from research.optimizers.random_search import RandomSearchOptimizer
from research.optimizers.tpe_search import TPESearchOptimizer
from research.optimizers.classical_sa import ClassicalSAOptimizer
from research.optimizers.quantum_annealing import QuantumAnnealingOptimizer


OPTIMIZER_CLASSES: Dict[str, Type[Any]] = {
    "grid": GridSearchOptimizer,
    "random": RandomSearchOptimizer,
    "tpe": TPESearchOptimizer,
    "classical_sa": ClassicalSAOptimizer,
    "quantum_annealing": QuantumAnnealingOptimizer,
}


class TeeStream(io.TextIOBase):
    """stdout/stderr를 콘솔과 파일에 동시에 기록한다."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def build_log_path(config: ExperimentConfig, run_started_at: datetime) -> str:
    """실험별 고유 persistent log 파일 경로를 만든다."""
    timestamp = run_started_at.strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"{timestamp}_{config.symbol}_{config.timeframe}_budget{config.eval_budget}_"
        f"seed{config.seed}_v{config.artifact_schema_version}.log"
    )
    return os.path.join(config.logs_dir, filename)


def make_objective(is_df: pd.DataFrame, cost_config: Dict[str, Any]):
    """In-sample 데이터에 대한 objective function을 만든다.

    거래 수가 MIN_TRADES_FOR_VALID_OBJECTIVE 미만이면 큰 벌점(-100)을
    부과하여 무거래/극소거래 파라미터가 선택되지 않도록 한다.
    """
    def objective(params: Dict[str, Any]) -> float:
        signals = generate_signals(is_df, params)
        result = run_backtest(signals, cost_config)
        metrics = compute_metrics(result["equity_curve"], result["trades"])
        if metrics["n_trades"] < MIN_TRADES_FOR_VALID_OBJECTIVE:
            return -100.0
        return metrics["sharpe"]
    return objective


def make_oos_objective(oos_df: pd.DataFrame, cost_config: Dict[str, Any]):
    """오에스 데이터에 대한 평가 함수를 만든다."""
    def objective(params: Dict[str, Any]) -> float:
        signals = generate_signals(oos_df, params)
        result = run_backtest(signals, cost_config)
        metrics = compute_metrics(result["equity_curve"], result["trades"])
        return metrics["sharpe"]
    return objective


def summarize_optimizer_results(
    results: list[Dict[str, Any]],
    params_list: list[Dict[str, Any]],
    eval_budget: int,
) -> Dict[str, Any]:
    """optimizer별 fold 결과를 요약한다."""
    oos_sharpes = [float(result["oos_sharpe"]) for result in results]
    is_sharpes = [float(result["is_sharpe"]) for result in results]
    local_mean_signed_deltas = [float(result["local_mean_signed_delta"]) for result in results]
    local_mean_degradations = [float(result["local_mean_degradation"]) for result in results]
    local_worst_degradations = [float(result["local_worst_degradation"]) for result in results]
    local_mean_improvements = [float(result["local_mean_improvement"]) for result in results]
    local_neighbors = [float(result["local_n_neighbors"]) for result in results]
    mean_oos_sharpe = float(pd.Series(oos_sharpes).mean())

    # IS-OOS correlation
    is_oos_corr = is_oos_correlation(is_sharpes, oos_sharpes) if len(is_sharpes) >= 3 else {
        "spearman_r": float("nan"), "spearman_p": float("nan"),
        "pearson_r": float("nan"), "n_obs": len(is_sharpes),
    }

    # IS-OOS decay 평균
    decays = [float(result.get("is_oos_decay", 0)) for result in results]
    mean_decay = float(np.mean(decays))

    return {
        "median_oos_sharpe": float(pd.Series(oos_sharpes).median()),
        "mean_oos_sharpe": mean_oos_sharpe,
        "std_oos_sharpe": float(pd.Series(oos_sharpes).std()),
        "param_dispersion": parameter_dispersion(params_list),
        "mean_local_signed_delta": float(pd.Series(local_mean_signed_deltas).mean()),
        "mean_local_degradation": float(pd.Series(local_mean_degradations).mean()),
        "mean_local_worst_degradation": float(pd.Series(local_worst_degradations).mean()),
        "mean_local_improvement": float(pd.Series(local_mean_improvements).mean()),
        "mean_local_neighbor_count": float(pd.Series(local_neighbors).mean()),
        "mean_is_oos_decay": mean_decay,
        "is_oos_correlation": is_oos_corr,
        "pbo_proxy": probability_of_backtest_overfitting(is_sharpes, oos_sharpes),
        "deflated_sharpe_proxy": deflated_sharpe_proxy(
            observed_sharpe=mean_oos_sharpe,
            n_trials=eval_budget,
            n_obs=len(oos_sharpes),
        ),
        "n_folds": len(results),
    }


def compare_qa_to_baselines(
    all_results: Dict[str, list[Dict[str, Any]]],
    seed: int,
) -> Dict[str, Any]:
    """QA와 baseline들의 통계 검정을 수행한다."""
    tests: Dict[str, Any] = {}
    qa_results = all_results.get("quantum_annealing", [])
    if not qa_results:
        return tests

    qa_sharpes = [float(result["oos_sharpe"]) for result in qa_results]
    print("\n=== Statistical Tests (QA vs baselines) ===")

    baseline_names = [name for name in OPTIMIZER_CLASSES if name != "quantum_annealing"]
    raw_p_values: list[float] = []
    baseline_order: list[str] = []

    for baseline in baseline_names:
        baseline_results = all_results.get(baseline, [])
        if not baseline_results:
            continue

        baseline_sharpes = [float(result["oos_sharpe"]) for result in baseline_results]
        n = min(len(qa_sharpes), len(baseline_sharpes))
        if n < 3:
            continue

        median_bootstrap = paired_bootstrap_test(
            qa_sharpes[:n],
            baseline_sharpes[:n],
            seed=seed,
            statistic="median",
        )
        mean_bootstrap = paired_bootstrap_test(
            qa_sharpes[:n],
            baseline_sharpes[:n],
            seed=seed,
            statistic="mean",
        )
        wilcoxon = wilcoxon_test(qa_sharpes[:n], baseline_sharpes[:n])
        effect_size = cliffs_delta(qa_sharpes[:n], baseline_sharpes[:n])

        tests[baseline] = {
            "n_folds": n,
            "primary_endpoint": "median_oos_sharpe",
            "paired_bootstrap_median": median_bootstrap,
            "paired_bootstrap_mean": mean_bootstrap,
            "wilcoxon": wilcoxon,
            "cliffs_delta": effect_size,
        }

        raw_p_values.append(float(median_bootstrap["p_value"]))
        baseline_order.append(baseline)

        wilcoxon_p = wilcoxon.get("p_value", float("nan"))
        print(
            f"  QA vs {baseline:15s} | median diff: {median_bootstrap['observed_diff']:+.3f} "
            f"[{median_bootstrap['ci_lower']:+.3f}, {median_bootstrap['ci_upper']:+.3f}] "
            f"bootstrap p={median_bootstrap['p_value']:.4f} | mean diff: {mean_bootstrap['observed_diff']:+.3f} | "
            f"wilcoxon p={wilcoxon_p:.4f} | Cliff's δ={effect_size['delta']:+.3f} ({effect_size['interpretation']})"
        )

    # Holm-Bonferroni correction
    if raw_p_values:
        corrected = holm_bonferroni_correction(raw_p_values)
        for i, baseline in enumerate(baseline_order):
            tests[baseline]["holm_corrected_p"] = corrected[i]
        print("\n  Holm-Bonferroni corrected p-values:")
        for i, baseline in enumerate(baseline_order):
            print(f"    QA vs {baseline:15s}: raw p={raw_p_values[i]:.4f} → corrected p={corrected[i]:.4f}")

    return tests


def run_experiment(config: ExperimentConfig | None = None):
    """전체 실험을 실행한다."""
    if config is None:
        config = ExperimentConfig()

    run_started_at = datetime.now(timezone.utc)
    log_path = build_log_path(config, run_started_at)
    os.makedirs(config.logs_dir, exist_ok=True)

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file, \
            redirect_stdout(TeeStream(sys.stdout, log_file)), \
            redirect_stderr(TeeStream(sys.stderr, log_file)):
        try:
            print(f"Persistent log → {log_path}")
            print(f"Run started at (UTC): {run_started_at.isoformat()}")

            print(f"=== Experiment: {config.symbol} {config.timeframe} ===")
            print(f"    Budget: {config.eval_budget}, Seed: {config.seed}")

            # 데이터 로드
            df = load_ohlcv(config.symbol, config.timeframe)
            print(f"    Data: {len(df)} rows, {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

            # Walk-forward fold 생성
            folds = generate_folds(df, config)
            print(f"    Folds: {len(folds)}")

            if not folds:
                print("    ERROR: No folds generated. Check data range and walk-forward config.")
                return

            if len(folds) < 6:
                raise ValueError("논문용 통계 검정을 위해서는 최소 6개의 fold가 필요합니다.")

            # Regime labeling
            fold_regimes = label_all_folds(df, folds, timeframe=config.timeframe)
            regime_map = {info["fold_id"]: info for info in fold_regimes}
            for info in fold_regimes:
                print(f"    Fold {info['fold_id']}: regime={info['regime_label']}, "
                      f"mean_vol={info['mean_oos_vol']:.4f}" if isinstance(info['mean_oos_vol'], float) else "")

            execution_config = {**asdict(config.cost), **asdict(config.backtest)}
            all_results: Dict[str, list[Dict[str, Any]]] = {name: [] for name in OPTIMIZER_CLASSES}
            all_best_params: Dict[str, list[Dict[str, Any]]] = {name: [] for name in OPTIMIZER_CLASSES}

            # 각 fold에서 optimizer 실행
            for fold in folds:
                print(f"\n  --- Fold {fold.fold_id}: IS [{fold.is_start} → {fold.is_end}], "
                      f"OOS [{fold.oos_start} → {fold.oos_end}] ---")

                is_df, oos_df = split_fold_data(df, fold)
                if len(is_df) < 100 or len(oos_df) < 50:
                    print(f"    SKIP: insufficient data (IS={len(is_df)}, OOS={len(oos_df)})")
                    continue

                is_objective = make_objective(is_df, execution_config)
                oos_objective = make_oos_objective(oos_df, execution_config)

                for opt_name, opt_class in OPTIMIZER_CLASSES.items():
                    optimizer = opt_class(eval_budget=config.eval_budget, seed=config.seed)
                    result = optimizer.optimize(is_objective, PARAM_SPACE)

                    if result["best_params"] is None:
                        print(f"    {opt_name}: no valid solution found")
                        continue

                    # OOS 평가
                    oos_score = oos_objective(result["best_params"])
                    decay = is_oos_decay(result["best_score"], oos_score)
                    robustness = local_robustness(result["best_params"], oos_objective, PARAM_SPACE)

                    # OOS 메트릭 전체
                    oos_signals = generate_signals(oos_df, result["best_params"])
                    oos_bt = run_backtest(oos_signals, execution_config)
                    oos_metrics = compute_metrics(oos_bt["equity_curve"], oos_bt["trades"])

                    # Convergence trace
                    conv_curve = extract_convergence_curve(result.get("history", []))
                    conv_stats = convergence_statistics(conv_curve)

                    # Regime info for this fold
                    regime_info = regime_map.get(fold.fold_id, {})

                    fold_result: Dict[str, Any] = {
                        "fold_id": fold.fold_id,
                        "optimizer": opt_name,
                        "is_sharpe": result["best_score"],
                        "oos_sharpe": oos_score,
                        "is_oos_decay": decay,
                        "regime_label": regime_info.get("regime_label", "unknown"),
                        "mean_oos_vol": regime_info.get("mean_oos_vol", None),
                        "convergence": conv_stats,
                        "local_mean_signed_delta": robustness["mean_signed_delta"],
                        "local_mean_degradation": robustness["mean_degradation"],
                        "local_worst_degradation": robustness["worst_degradation"],
                        "local_mean_improvement": robustness["mean_improvement"],
                        "local_n_neighbors": robustness["n_neighbors"],
                        "n_evals": result["n_evals"],
                        **{f"oos_{k}": v for k, v in oos_metrics.items()},
                        "best_params": result["best_params"],
                    }

                    all_results[opt_name].append(fold_result)
                    all_best_params[opt_name].append(result["best_params"])

                    print(f"    {opt_name:20s} | IS Sharpe: {result['best_score']:+.3f} | "
                          f"OOS Sharpe: {oos_score:+.3f} | Decay: {decay:.2f}")

            # 집계
            print("\n=== Summary ===")
            summary: Dict[str, Any] = {}
            for opt_name in OPTIMIZER_CLASSES:
                results = all_results[opt_name]
                params_list = all_best_params[opt_name]
                if not results:
                    continue

                summary[opt_name] = summarize_optimizer_results(results, params_list, config.eval_budget)

                print(
                    f"  {opt_name:20s} | Median OOS Sharpe: {summary[opt_name]['median_oos_sharpe']:+.3f} | "
                    f"Dispersion: {summary[opt_name]['param_dispersion']:.4f} | "
                    f"Local degradation: {summary[opt_name]['mean_local_degradation']:+.4f}"
                )

            statistical_tests = compare_qa_to_baselines(all_results, seed=config.seed)

            # Regime-conditional summary
            fold_oos_scores: Dict[str, object] = {}
            for opt_name in OPTIMIZER_CLASSES:
                fold_oos_scores[opt_name] = all_results[opt_name]
            regime_summary = regime_conditional_summary(fold_regimes, fold_oos_scores)

            # 결과 저장
            os.makedirs(config.artifacts_dir, exist_ok=True)
            artifact_path = os.path.join(config.artifacts_dir, f"{config.symbol}_{config.seed}.json")
            artifact_filename = (
                f"{config.symbol}_{config.timeframe}_budget{config.eval_budget}_seed{config.seed}_"
                f"v{config.artifact_schema_version}.json"
            )
            artifact_path = os.path.join(config.artifacts_dir, artifact_filename)

            # Environment info for reproducibility
            env_info = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            }
            try:
                import scipy  # type: ignore[import-untyped]
                env_info["scipy_version"] = scipy.__version__
            except ImportError:
                pass
            try:
                import neal  # type: ignore[import-untyped]
                env_info["neal_version"] = getattr(neal, "__version__", "unknown")
            except ImportError:
                pass
            try:
                import optuna  # type: ignore[import-untyped]
                env_info["optuna_version"] = str(getattr(optuna, "__version__", "unknown"))
            except ImportError:
                pass

            artifact: Dict[str, Any] = {
                "schema_version": config.artifact_schema_version,
                "log_file": log_path,
                "environment": env_info,
                "config": {
                    "symbol": config.symbol,
                    "timeframe": config.timeframe,
                    "eval_budget": config.eval_budget,
                    "seed": config.seed,
                    "data_start": str(df["timestamp"].iloc[0]),
                    "data_end": str(df["timestamp"].iloc[-1]),
                    "n_folds": len(folds),
                    "cost": asdict(config.cost),
                    "backtest": asdict(config.backtest),
                    "objective": {
                        "min_trades_threshold": MIN_TRADES_FOR_VALID_OBJECTIVE,
                        "penalty_score": -100.0,
                        "note": "IS n_trades < min_trades_threshold 시 penalty_score 반환",
                    },
                    "grid_baseline": {
                        "sampling_strategy": GRID_SAMPLING_STRATEGY,
                        "note": "전체 유효 파라미터에서 eval_budget 크기 비복원 무작위 추출 (seed 고정)",
                    },
                    "regime_classification": {
                        "method": "rolling_volatility_quantile",
                        "lookback_days": 90,
                        "quantiles": [0.33, 0.67],
                        "labels": ["low", "medium", "high"],
                    },
                    "logs_dir": config.logs_dir,
                    "run_started_at_utc": run_started_at.isoformat(),
                },
                "summary": summary,
                "statistical_tests": statistical_tests,
                "regime_analysis": {
                    "fold_regimes": fold_regimes,
                    "regime_conditional_summary": regime_summary,
                },
                "fold_results": {k: v for k, v in all_results.items()},
            }

            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, default=str)

            print(f"\nArtifacts saved → {artifact_path}")
            return artifact
        except Exception:
            print("\nExperiment failed. Traceback follows:", file=sys.stderr)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    run_experiment()
