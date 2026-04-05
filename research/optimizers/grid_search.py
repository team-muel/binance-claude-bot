"""
Grid Search optimizer.

전수탐색이 불가능할 경우 seed 기반 random subset으로 eval_budget만큼만
평가한다. 이전의 고정 stride 방식 대신 무작위 서브샘플링을 사용하여
특정 파라미터 영역이 체계적으로 누락되는 문제를 방지한다.
"""
import random
from typing import Any, Callable, Dict

from research.optimizers.base import BaseOptimizer
from research.strategy.params import get_all_valid_params


class GridSearchOptimizer(BaseOptimizer):

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        from research.strategy.params import get_all_valid_params as _get_all
        all_params = _get_all(param_space)

        # budget보다 조합이 많으면 seed 기반 random subset
        if len(all_params) > self.eval_budget:
            rng = random.Random(self.seed)
            candidates = rng.sample(all_params, self.eval_budget)
        else:
            candidates = all_params

        best_score = float("-inf")
        best_params = None

        for params in candidates:
            score = objective_fn(params)
            self._record(params, score)
            if score > best_score:
                best_score = score
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_evals": len(self.history),
            "history": self.history,
        }
