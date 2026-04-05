"""
연구 실험 설정.

파라미터 범위, walk-forward fold 설정, 탐색 예산, 거래비용 가정 등
모든 실험 공통 설정을 한 곳에서 관리한다.
"""
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# 파라미터 공간 (이산화)
# ---------------------------------------------------------------------------
PARAM_SPACE: Dict[str, List[int] | List[float]] = {
    "ema_fast":       list(range(5, 55, 5)),          # 5,10,...,50  → 10개
    "ema_slow":       list(range(20, 210, 20)),        # 20,40,...,200 → 10개
    "rsi_period":     [7, 10, 14, 17, 21],             # 5개
    "rsi_entry":      [25, 30, 35, 40, 45],            # 5개 (Long 진입 RSI 상단 임계값)
    "rsi_short_entry": [55, 60, 65, 70, 75],           # 5개 (Short 진입 RSI 하단 임계값)
    "atr_period":     [7, 10, 14, 17, 21],             # 5개
    "sl_multiplier":  [1.0, 1.5, 2.0, 2.5, 3.0],      # 5개
    "tp_multiplier":  [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # 8개
}


# ---------------------------------------------------------------------------
# Walk-forward 설정
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    """Rolling walk-forward fold 설정."""
    is_window_months: int = 18       # in-sample 윈도우 (개월)
    oos_window_months: int = 3       # out-of-sample 윈도우 (개월)
    step_months: int = 3             # fold 이동 간격 (개월)


# ---------------------------------------------------------------------------
# 거래비용 가정
# ---------------------------------------------------------------------------
@dataclass
class CostConfig:
    """수수료와 슬리피지 가정."""
    commission_rate: float = 0.0004   # 편도 수수료 (taker 0.04%)
    slippage_rate: float = 0.0001     # 편도 슬리피지 가정


@dataclass
class BacktestConfig:
    """포트폴리오 수준 백테스트 가정."""
    initial_capital: float = 10_000.0
    position_size_fraction: float = 1.0


# ---------------------------------------------------------------------------
# Objective 설정
# ---------------------------------------------------------------------------
MIN_TRADES_FOR_VALID_OBJECTIVE: int = 5
"""IS 구간에서 이 수 미만 거래 시 objective에 큰 벌점(-100)을 부과한다."""

GRID_SAMPLING_STRATEGY: str = "random_subset_without_replacement"
"""
Grid baseline의 탐색 방식.
전체 유효 파라미터 집합에서 eval_budget 크기만큼 seed 기반으로
비복원 무작위 추출(random subset without replacement)을 사용한다.
이전의 고정 stride subsampling(v2)을 대체하며, 특정 파라미터 영역이
체계적으로 누락되는 문제를 방지한다.
"""


# ---------------------------------------------------------------------------
# 실험 공통 설정
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """실험 전체 설정."""
    # 데이터
    symbol: str = "BTCUSDT"
    timeframe: str = "30m"
    replication_symbols: List[str] = field(default_factory=lambda: ["ETHUSDT", "SOLUSDT"])

    # walk-forward
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    # 비용
    cost: CostConfig = field(default_factory=CostConfig)

    # 백테스트 자본/사이징
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # 탐색 예산 (objective evaluation count)
    eval_budget: int = 500

    # 재현성
    seed: int = 42

    # 결과 저장
    artifacts_dir: str = "research/artifacts"
    logs_dir: str = "logs/experiments"
    artifact_schema_version: int = 5


# ---------------------------------------------------------------------------
# 비용 민감도 시나리오
# ---------------------------------------------------------------------------
COST_SENSITIVITY_SCENARIOS = [
    {"name": "low_cost",  "commission_rate": 0.0002, "slippage_rate": 0.00005},
    {"name": "base",      "commission_rate": 0.0004, "slippage_rate": 0.0001},
    {"name": "high_cost", "commission_rate": 0.0008, "slippage_rate": 0.0003},
]
