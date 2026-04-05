"""ETH + SOL 복제 실험 실행."""
from research.config import ExperimentConfig
from research.run_experiment import run_experiment


def run_replication():
    """ETHUSDT와 SOLUSDT 복제 실험을 순차 실행한다."""
    for symbol in ["ETHUSDT", "SOLUSDT"]:
        print(f"\n{'='*60}")
        print(f"  Starting replication: {symbol}")
        print(f"{'='*60}\n")
        cfg = ExperimentConfig(symbol=symbol)
        run_experiment(cfg)
        print(f"\n  Finished: {symbol}\n")


if __name__ == "__main__":
    run_replication()
