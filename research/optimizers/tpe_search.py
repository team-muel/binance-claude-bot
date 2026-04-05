"""
Bayesian/TPE optimizer (Optuna 래퍼).

이산 조합공간에 적합한 Tree-structured Parzen Estimator를 사용한다.
EMA 제약 위반 trial은 budget을 소비하지 않도록 ask/tell API를 사용한다.
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

        evals_done = 0
        while evals_done < self.eval_budget:
            trial = study.ask()  # type: ignore[union-attr]
            params: Dict[str, Any] = {}
            for k, v in param_space.items():
                params[k] = trial.suggest_categorical(k, v)

            # EMA_fast < EMA_slow 제약 — 위반 시 budget 소비 없이 prune
            if params["ema_fast"] >= params["ema_slow"]:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # type: ignore[attr-defined]
                continue

            score = objective_fn(params)
            self._record(params, score)
            study.tell(trial, score)  # type: ignore[union-attr]
            evals_done += 1

        best: Any = cast(Any, study.best_trial)  # type: ignore[union-attr]
        return {
            "best_params": best.params,
            "best_score": best.value,
            "n_evals": len(self.history),
            "history": self.history,
        }
