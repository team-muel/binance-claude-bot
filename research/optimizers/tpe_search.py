"""
Bayesian/TPE optimizer (Optuna 래퍼).

이산 조합공간에 적합한 Tree-structured Parzen Estimator를 사용한다.
"""
from typing import Any, Callable, Dict, cast

import optuna  # type: ignore[import-untyped]

from research.optimizers.base import BaseOptimizer


class TPESearchOptimizer(BaseOptimizer):

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # type: ignore[attr-defined]
        sampler: Any = optuna.samplers.TPESampler(seed=self.seed)  # type: ignore[attr-defined]
        study: Any = optuna.create_study(direction="maximize", sampler=sampler)  # type: ignore[attr-defined]

        def _objective(trial: Any) -> float:
            params: Dict[str, Any] = {}
            for k, v in param_space.items():
                val = trial.suggest_categorical(k, v)
                params[k] = val
            # EMA_fast < EMA_slow 제약
            if params["ema_fast"] >= params["ema_slow"]:
                return float("-inf")
            score = objective_fn(params)
            self._record(params, score)
            return score

        study.optimize(_objective, n_trials=self.eval_budget, show_progress_bar=False)  # type: ignore[union-attr]

        best: Any = cast(Any, study.best_trial)  # type: ignore[union-attr]
        return {
            "best_params": best.params,
            "best_score": best.value,
            "n_evals": len(self.history),
            "history": self.history,
        }
