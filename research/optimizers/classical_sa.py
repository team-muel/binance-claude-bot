"""
Classical Simulated Annealing optimizer.

동일 QUBO 공간에서 양자성을 제거한 기준(baseline)으로 사용한다.
"""
import math
import random
from typing import Any, Callable, Dict, Literal

from research.optimizers.base import BaseOptimizer

CoolingSchedule = Literal["exponential", "linear", "logarithmic"]


class ClassicalSAOptimizer(BaseOptimizer):

    def __init__(
        self,
        eval_budget: int,
        seed: int = 42,
        temp_start: float = 10.0,
        temp_end: float = 0.01,
        cooling_schedule: CoolingSchedule = "exponential",
    ):
        super().__init__(eval_budget, seed)
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.cooling_schedule = cooling_schedule

    def _temperature(self, evals_done: int) -> float:
        """현재 온도를 cooling schedule에 따라 계산한다."""
        progress = evals_done / self.eval_budget
        if self.cooling_schedule == "exponential":
            return self.temp_start * (self.temp_end / self.temp_start) ** progress
        elif self.cooling_schedule == "linear":
            return self.temp_start + (self.temp_end - self.temp_start) * progress
        elif self.cooling_schedule == "logarithmic":
            return self.temp_start / (1 + math.log(1 + evals_done))
        else:
            return self.temp_start * (self.temp_end / self.temp_start) ** progress

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        keys = list(param_space.keys())

        # 초기 해
        current = {k: rng.choice(v) for k, v in param_space.items()}
        while current["ema_fast"] >= current["ema_slow"]:
            current = {k: rng.choice(v) for k, v in param_space.items()}

        current_score = objective_fn(current)
        self._record(current, current_score)
        best_params = current.copy()
        best_score = current_score
        evals_done = 1

        while evals_done < self.eval_budget:
            # 온도 스케줄
            t = self._temperature(evals_done)

            # 이웃 해: 랜덤 하나의 파라미터를 변경
            neighbor = current.copy()
            key = rng.choice(keys)
            neighbor[key] = rng.choice(param_space[key])

            # EMA 제약 위반 시 재시도
            if neighbor["ema_fast"] >= neighbor["ema_slow"]:
                continue

            neighbor_score = objective_fn(neighbor)
            self._record(neighbor, neighbor_score)
            evals_done += 1

            delta = neighbor_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / max(t, 1e-10)):
                current = neighbor
                current_score = neighbor_score

            if current_score > best_score:
                best_score = current_score
                best_params = current.copy()

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_evals": len(self.history),
            "history": self.history,
        }
