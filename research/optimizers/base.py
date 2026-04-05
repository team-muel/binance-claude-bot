"""
Optimizer 공통 인터페이스.

모든 optimizer는 이 ABC를 상속하고 optimize()를 구현한다.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class BaseOptimizer(ABC):
    """파라미터 탐색기의 공통 인터페이스."""

    def __init__(self, eval_budget: int, seed: int = 42):
        self.eval_budget = eval_budget
        self.seed = seed
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        """
        Objective function을 최대화하는 파라미터를 찾는다.

        Parameters
        ----------
        objective_fn : callable
            params dict → float (높을수록 좋음)
        param_space : dict
            파라미터명 → 이산 후보 리스트

        Returns
        -------
        dict
            best_params, best_score, n_evals, history
        """
        ...

    def _record(self, params: Dict[str, Any], score: float):
        self.history.append({"params": params.copy(), "score": score})
