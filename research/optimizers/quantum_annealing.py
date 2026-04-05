"""
Quantum Annealing optimizer (neal 시뮬레이터).

파라미터 공간을 one-hot binary 변수로 인코딩한 뒤
QUBO 형태로 변환해 neal SimulatedAnnealingSampler로 풀이한다.

현재 구현은 D-Wave Ocean SDK의 neal 패키지를 사용한 simulated quantum annealing이다.
실제 D-Wave QPU나 hybrid solver를 쓰지 않으므로 논문에서 이 점을 반드시 명시해야 한다.
(실제 양자 하드웨어 접근은 future work로 분류)
"""
import random
from typing import Any, Callable, Dict, Optional

from research.optimizers.base import BaseOptimizer


class QuantumAnnealingOptimizer(BaseOptimizer):

    def __init__(
        self,
        eval_budget: int,
        seed: int = 42,
        num_reads: int = 50,
        qubo_penalty: float = 10.0,
    ):
        super().__init__(eval_budget, seed)
        self.num_reads = num_reads
        self.qubo_penalty = qubo_penalty

    def _encode_params_onehot(self, param_space: Dict[str, list[Any]]) -> Dict[str, Any]:
        """파라미터 공간을 one-hot 인코딩 메타데이터로 변환한다."""
        encoding: Dict[str, Any] = {}
        offset = 0
        for key, values in param_space.items():
            n = len(values)
            encoding[key] = {
                "offset": offset,
                "n_bits": n,
                "values": values,
            }
            offset += n
        return encoding

    def _decode_sample(
        self, sample: Any, encoding: Dict[str, Any], param_space: Dict[str, list[Any]]
    ) -> Optional[Dict[str, Any]]:
        """바이너리 샘플을 파라미터 dict로 디코딩한다."""
        params: Dict[str, Any] = {}
        for key, meta in encoding.items():
            bits = [sample.get(meta["offset"] + i, 0) for i in range(meta["n_bits"])]
            active = [i for i, b in enumerate(bits) if b == 1]
            if len(active) != 1:
                return None  # one-hot 위반
            params[key] = meta["values"][active[0]]
        return params

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        try:
            import neal  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("neal 패키지가 필요합니다: pip install dwave-neal")

        encoding = self._encode_params_onehot(param_space)
        sampler: Any = neal.SimulatedAnnealingSampler()  # type: ignore[attr-defined]
        rng = random.Random(self.seed)

        best_score = float("-inf")
        best_params = None
        evals_done = 0

        while evals_done < self.eval_budget:
            reads = min(self.num_reads, self.eval_budget - evals_done)

            # One-hot 제약을 penalty로 넣은 QUBO 구성
            Q: Dict[tuple[int, int], float] = {}
            penalty = self.qubo_penalty
            for _key, meta in encoding.items():
                off = meta["offset"]
                n = meta["n_bits"]
                # one-hot: sum(x_i) = 1 → penalty * (sum(x_i) - 1)^2
                for i in range(n):
                    Q[(off + i, off + i)] = Q.get((off + i, off + i), 0) - penalty
                    for j in range(i + 1, n):
                        Q[(off + i, off + j)] = Q.get((off + i, off + j), 0) + 2 * penalty

            response = sampler.sample_qubo(Q, num_reads=reads, seed=rng.randint(0, 2**31))  # type: ignore[arg-type]

            for sample in response.samples():  # type: ignore[no-untyped-call]
                if evals_done >= self.eval_budget:
                    break
                params = self._decode_sample(sample, encoding, param_space)
                if params is None:
                    continue
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
