# 데이터, Walk-Forward 설계, 평가 프로토콜

## 자산 및 타임프레임

- 본 실험: BTCUSDT 30m, BTCUSDT 1h
- 복제 자산: ETHUSDT 30m / 1h, SOLUSDT 30m / 1h
- 데이터 소스: Binance Futures OHLCV
- 총 6개 asset-timeframe 조합

## 권장 데이터 기간

- 사용 가능한 최소 기간: 36개월
- 권장 기간: 48~60개월
- 이유: fold 수준 OOS 비교와 안정성 분석을 위해 충분한 walk-forward fold 수가 필요함

## Walk-Forward 설정

- In-sample 윈도우: 18개월
- Out-of-sample 윈도우: 3개월
- Step 크기: 3개월
- 원칙: OOS 데이터는 탐색 과정에 절대 노출되지 않아야 함
- 30m과 1h 모두 동일한 calendar period 사용 (타임프레임이 아니라 기간 기준)

## 비교 optimizer

- Grid Search
- Random Search
- TPE
- Classical Simulated Annealing
- Quantum Annealing

## 공정성 기준

- 모든 optimizer에 동일한 목적함수 평가 예산 적용
- 동일한 seed 정책
- 동일한 데이터 분할
- 동일한 비용 가정(수수료·슬리피지)
- wall-clock time은 보조 지표로만 기록

## 비용 모델 범위

- 첫 논문에서는 수수료와 슬리피지만 반영
- funding 비용은 범위 밖으로 두고 future work로 분류

## 백테스트 자본/사이징 가정

- 단일 포지션만 보유한다
- 기본 가정은 full-capital, no-leverage, mark-to-market equity curve다
- long/short 모두 동일한 포지션 사이징 규칙을 사용한다
- Sharpe, MDD, Calmar, total return은 포트폴리오 equity 기준으로 계산한다

## Objective 함수 규칙 (schema v4 이후)

- **최소 거래 수 기준 (min_trades_threshold = 5)**: IS 구간에서 거래 수가 5 미만인 파라미터 조합은 목적함수 값으로 -100.0을 반환한다. 이를 통해 무거래·극소거래 파라미터가 최적 해로 선택되는 것을 방지한다.
- 이 기준은 IS 최적화에만 적용된다. OOS 평가는 그대로 Sharpe를 기록한다.
- min_trades_threshold 값과 penalty_score는 아티팩트 config.objective 필드에 기록된다.

## Grid Baseline 정의 (schema v4 이후)

- **sampling_strategy: random_subset_without_replacement**: 전체 유효 파라미터 조합에서 eval_budget 크기만큼 seed 기반 비복원 무작위 추출을 수행한다.
- v2의 고정 stride subsampling(`all_params[::step]`)을 대체한다. 고정 stride는 특정 파라미터 영역이 체계적으로 누락될 수 있었다.
- 사용된 sampling 방식과 설명은 아티팩트 config.grid_baseline 필드에 기록된다.

## 통계 검정 방법 (schema v4 이후)

- **Median bootstrap (primary endpoint)**: fold index를 재표본추출 후 `median(a[idx]) - median(b[idx])`를 계산한다. `median(a_i - b_i)` 방식과 달리, 논문의 1차 지표인 "fold-전체 median OOS Sharpe 차이"를 직접 bootstrap한다.
- Mean bootstrap은 보조 지표로 기존 방식(`mean(a_i - b_i)`)을 유지한다. Mean의 경우 두 방식이 수학적으로 동일하다.
- Wilcoxon signed-rank test: 비모수 보조 검정으로 유지한다.

## 주요 지표

- Median OOS Sharpe
- MDD
- Calmar
- Turnover
- Parameter dispersion
- One-step perturbation 하에서의 local robustness
- IS→OOS decay
- PBO proxy (current implementation)
- Deflated Sharpe proxy (current implementation)

## 출력 아티팩트

- fold별 최적 파라미터
- fold별 OOS 지표
- optimizer별 요약 통계
- 통계 검정 결과

## 시장 국면 분류 (schema v5 이후)

- **방법**: Rolling realized volatility (log return 기준) → 전체 기간 분위수 기반 3분류
  - High volatility: top 33% quantile 초과
  - Medium volatility: 33%–67% quantile
  - Low volatility: bottom 33% quantile 이하
- **Lookback**: 90일 rolling window
- 각 OOS fold에 해당 기간의 평균 실현 변동성을 기준으로 regime label(high/medium/low) 부여
- Regime label은 아티팩트의 fold_results 내 각 fold에 기록

## Ablation Study 설계 (schema v5 이후)

### QA Ablation

BTC 30m에서만 실행, budget=500 고정. 아래 변수를 단일 변수씩 변경하고 나머지는 기본값 유지:

| 변수 | 기본값 | 변이 | 목적 |
| ---- | ------ | ---- | ---- |
| num_reads | 50 | [10, 25, 50, 100, 200] | 샘플링 충분성 |
| QUBO penalty | 10.0 | [1.0, 5.0, 10.0, 20.0, 50.0] | 제약 강도 민감도 |

### Classical SA Ablation

BTC 30m에서만 실행, budget=500 고정:

| 변수 | 기본값 | 변이 | 목적 |
| ---- | ------ | ---- | ---- |
| temp_start | 10.0 | [1.0, 5.0, 10.0, 50.0] | 탐색 범위 |
| cooling_schedule | exponential | [exponential, linear, logarithmic] | 수렴 행동 |

### 보고

- 각 ablation 변이별 median OOS Sharpe, parameter dispersion, mean degradation
- 기본값 결과를 기준으로 상대 변화

## 비용 민감도 분석 (schema v5 이후)

BTC 30m에서 QA + best baseline만 비교:

| 시나리오 | Commission | Slippage | 설명 |
| -------- | ---------- | -------- | ---- |
| Low cost | 0.02% | 0.005% | Maker-dominant |
| Base | 0.04% | 0.01% | 기본 가정 |
| High cost | 0.08% | 0.03% | 소규모 계좌/높은 impact |

## Execution Delay Sensitivity (schema v5 이후)

- BTC 30m에서 QA + best baseline만 비교
- Signal 발생 시점 기준 +1, +2 bar 지연 진입
- 각 지연 수준별 OOS Sharpe 변화 보고

## Funding Rate Post-hoc 추정 (schema v5 이후)

- Binance BTCUSDT 과거 8h funding rate 데이터 활용
- 보유 기간 × 평균 funding rate로 equity curve에서 차감
- Funding 포함/미포함 Sharpe 비교 (Discussion 절에 보고)

## Convergence Trace 분석 (schema v5 이후)

- 각 optimizer의 eval_count vs best_so_far objective 기록 (fold별)
- Fold 평균 ± stderr convergence curve
- 보고: 50% of final score 도달 eval count, 마지막 100 evals improvement rate

## Cross-Setup Consistency (schema v5 이후)

- 6개 (asset × timeframe) 조합에서 QA 상대 순위
- QA rank 1st 비율, Kendall rank correlation across setups
- 외부 타당성의 핵심 증거

## 통계 검정 방법 확장 (schema v5 이후)

- **Effect size**: Cliff's delta (비모수, 순서형 적합) + 해석 (negligible/small/medium/large)
- **다중비교 보정**: Holm-Bonferroni correction (QA vs 4 baselines)
- **CSCV PBO**: Bailey et al. (2014) Combinatorially Symmetric Cross-Validation
  - IS 기간을 S개 서브셋 분할 → C(S, S/2) 조합 → IS/OOS 역할 교환
  - Best-IS strategy가 OOS median 이하인 비율 = PBO
- **IS-OOS Correlation**: Optimizer별 IS score vs OOS score Spearman 상관계수

## Failure Analysis (schema v5 이후)

- QA가 가장 나쁜 결과를 낸 fold/asset에 대한 case study
- 해당 fold의 regime, 파라미터 선택, IS landscape 특성 분석
- Discussion > "When does QA fail?" 절에 보고
