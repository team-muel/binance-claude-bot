"""
Random Search optimizer.

Bergstra & Bengio (2012) 기준의 무작위 탐색.
"""
import random
from typing import Any, Callable, Dict

from research.optimizers.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        best_score = float("-inf")
        best_params = None
        evals_done = 0

        while evals_done < self.eval_budget:
            params = {k: rng.choice(v) for k, v in param_space.items()}
            # EMA_fast < EMA_slow 제약
            if params["ema_fast"] >= params["ema_slow"]:
                continue
            score = objective_fn(params)
            self._record(params, score)
            evals_done += 1
            if score > best_score:
                best_score = score
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_evals": len(self.history),
            "history": self.history,
        }
