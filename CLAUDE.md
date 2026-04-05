# qa-param-search

Quantum Annealing 기반 기술지표 파라미터 탐색 — 규칙기반 암호화폐 전략의 OOS 강건성 비교 연구.

## 논문 개요

| 항목 | 내용 |
| ------ | ------ |
| 과제명 | Quantum Annealing-Based Parameter Search for Rule-Based Crypto Trading: Robustness Under Non-Stationarity |
| 핵심 질문 | 비정상 시계열에서 QUBO 기반 양자 어닐링이 고전적 탐색보다 더 강건한 OOS 일반화를 달성하는가? |
| 전략 | EMA-RSI-ATR 고정 규칙전략 |
| 데이터 | Binance BTCUSDT/ETHUSDT/SOLUSDT × 30m/1h |
| 비교군 | Grid / Random / Bayesian(TPE) / Classical SA / **Quantum Annealing** |
| 검증방식 | Walk-forward optimization + strict OOS + regime-conditional evaluation |
| 주요 지표 | IS-OOS decay, parameter stability, OOS Sharpe, CSCV PBO, Cliff's delta |
| 논문 타입 | 방법론 + 실증 비교형 |

## 연구질문과 가설

**RQ.** 비정상적이고 구조 변화가 심한 암호화폐 시장에서, QUBO 인코딩 기반 양자 어닐링 파라미터 탐색이 고전적 방법 대비 walk-forward re-optimization 하에서 더 강건한 OOS 일반화를 달성하는가?

- **H1 (일반화)**: QA 기반 탐색은 fold 전반에서 IS→OOS performance decay가 고전적 baseline보다 유의하게 작다.
- **H2 (파라미터 안정성)**: QA는 rolling fold 간 선택 파라미터의 dispersion이 더 낮고, perturbation sensitivity가 더 작다.
- **H3 (국면 조건부)**: QA의 상대적 이점은 고변동성(high-volatility) 국면에서 더 두드러진다.

> 핵심 서사는 **비정상 시계열에서의 파라미터 일반화 가능성과 OOS 강건성**이다.

## Contributions

1. **QUBO 정식화**: 규칙기반 암호화폐 전략의 파라미터 탐색 문제를 QUBO 또는 유사 조합최적화 문제로 정식화한다.
2. **Equal-budget OOS 비교**: 동일 탐색 예산 하에서 classical 탐색기와 quantum-assisted 탐색기를 walk-forward OOS 기준으로 정면 비교한다.
3. **연구-운용 연결**: 오프라인 최적화 결과를 실시간 봇이 로드하는 구조를 제시해, 연구와 운용 간 연결 가능성을 보여준다.

## 명시적 제외 범위

이번 논문에서 **하지 않는 것**:

- QAOA 동시 비교 (future work)
- QPCA / 상관구조 통제
- 예측 모델 결합 (LSTM, PatchTST, QLSTM 등)
- 멀티에이전트 시스템 아키텍처
- 실시간 엔진 성능 자체를 contribution으로 내세우는 것
- 실제 D-Wave QPU / hybrid solver (neal 시뮬레이터만 사용, limitation으로 명시)

## 디렉터리 구조

```text
qa-param-search/
├── CLAUDE.md                        # 프로젝트 기준 문서 (이 파일)
├── README.md                        # 공개용 프로젝트 소개 문서
├── PAPER_SCOPE.md                   # 논문 질문, 가설, 범위 정리
├── EXPERIMENT_PROTOCOL.md           # 데이터, fold, 비교 기준, 통계 절차
├── MANUSCRIPT_OUTLINE.md            # 원고 섹션, 표, 그림 계획
├── research/
│   ├── config.py                    # 실험 설정 (파라미터 범위, fold 설정, budget 등)
│   ├── run_experiment.py            # 실험 진입점
│   ├── data/
│   │   ├── downloader.py            # Binance OHLCV 다운로드 (ccxt)
│   │   └── loader.py                # 로컬 CSV/Parquet → DataFrame 변환
│   ├── strategy/
│   │   ├── indicators.py            # EMA, RSI, ATR 계산
│   │   ├── rules.py                 # 진입/청산 규칙 (파라미터 주입)
│   │   └── params.py                # 파라미터 공간 정의 및 이산화
│   ├── backtest/
│   │   ├── engine.py                # 결정론적 OHLCV 리플레이 시뮬레이터
│   │   ├── cost_model.py            # 수수료, 슬리피지 비용 가정
│   │   └── metrics.py               # Sharpe, MDD, Calmar, turnover 등 성과 집계
│   ├── optimizers/
│   │   ├── base.py                  # Optimizer 공통 인터페이스
│   │   ├── grid_search.py           # Grid search
│   │   ├── random_search.py         # Random search
│   │   ├── tpe_search.py            # Bayesian/TPE (Optuna 래퍼)
│   │   ├── classical_sa.py          # Classical simulated annealing
│   │   └── quantum_annealing.py     # Quantum annealing (D-Wave / neal 래퍼)
│   ├── evaluation/
│   │   ├── walk_forward.py          # Rolling walk-forward fold 생성 및 실행
│   │   ├── stability.py             # Parameter dispersion, local robustness (perturbation)
│   │   ├── overfitting.py           # IS→OOS decay, PBO, Deflated Sharpe, IS-OOS correlation
│   │   ├── statistics.py            # Paired bootstrap, Wilcoxon, Cliff's delta, Holm correction
│   │   ├── nonstationarity.py       # Rolling vol regime 분류, regime-conditional 분석
│   │   ├── landscape.py             # 2D IS objective heatmap, landscape shift 분석
│   │   ├── convergence.py           # Optimizer convergence trace 분석
│   │   └── pbo_cscv.py              # CSCV 기반 PBO (Bailey et al. 2014)
│   ├── backtest/
│   │   ├── engine.py                # 결정론적 OHLCV 리플레이 시뮬레이터
│   │   ├── cost_model.py            # 수수료, 슬리피지 비용 가정
│   │   ├── metrics.py               # Sharpe, MDD, Calmar, turnover 등 성과 집계
│   │   ├── funding_impact.py        # Funding rate post-hoc 영향 추정
│   │   └── delay_sensitivity.py     # Execution delay sensitivity 분석
│   ├── run_ablation.py              # QA/SA ablation study 진입점
│   └── artifacts/                   # fold별 best params, OOS metrics, 재현용 로그
├── logs/                            # 실험 로그 및 차트
├── requirements.txt                 # Python 의존성
└── .env                             # API 키 (연구 하네스는 이 파일 없이도 동작)
```

## 실험 흐름

```text
OHLCV 데이터 로드 (Binance BTCUSDT 30m)
  └─ walk-forward fold 분할 (rolling window)
       │  예: 12–18개월 in-sample / 3개월 out-of-sample / step 3개월
       │
       └─ 각 fold에서 동일 evaluation budget으로 optimizer 독립 실행
            ├─ Grid Search
            ├─ Random Search
            ├─ Bayesian / TPE
            ├─ Classical Simulated Annealing
            └─ Quantum Annealing (QUBO 인코딩)
                 │
                 └─ fold별 best params 선택 (in-sample objective 기준)
                      └─ strict OOS 평가 (OOS 구간은 탐색에 절대 노출 안 됨)
                           └─ fold별 OOS Sharpe, MDD, turnover, stability 기록
                                └─ 전체 fold 집계
                                     ├─ 통계 검정 (paired bootstrap / Wilcoxon)
                                     ├─ budget sensitivity analysis
                                     └─ 결과 보고
```

## 전략과 파라미터 공간

### 전략 규칙

| 구성요소 | 설명 |
| ---------- | ------ |
| 진입 (Long) | EMA_fast > EMA_slow 且 RSI < RSI_entry_threshold |
| 진입 (Short) | EMA_fast < EMA_slow 且 RSI > RSI_short_entry |
| 청산 | 반대 크로스 또는 RSI 역전 또는 SL/TP 도달 |
| Stop-Loss | ATR × SL_multiplier |
| Take-Profit | ATR × TP_multiplier |

### 파라미터 후보 (이산화)

| 파라미터 | 범위 | 단위 | 후보 수 (예시) |
| ---------- | ------ | ------ | ---------------- |
| EMA_fast | 5–50 | period | ~10 |
| EMA_slow | 20–200 | period | ~10 |
| RSI_period | 7–21 | period | ~5 |
| RSI_entry | 25–45 | 값 | ~5 |
| RSI_short_entry | 55–75 | 값 | ~5 |
| ATR_period | 7–21 | period | ~5 |
| SL_multiplier | 1.0–3.0 | 배수 | ~5 |
| TP_multiplier | 1.5–5.0 | 배수 | ~8 |

- 제약: EMA_fast < EMA_slow (penalty 또는 encoding으로 강제)
- 총 조합 수: ~수십만 (grid 전수탐색은 비현실적 → 탐색기 비교에 적합한 규모)

## 비교군과 공정성 기준

| Optimizer | 유형 | 비고 |
| ----------- | ------ | ------ |
| Grid Search | Exhaustive (서브샘플링) | 약한 baseline, 참고용 |
| Random Search | Stochastic | Bergstra & Bengio (2012) 기준 |
| Bayesian / TPE | Model-based | Optuna TPE (이산 조합공간에 적합) |
| Classical SA | Annealing-family | 동일 QUBO 풀이, 양자성 분리 기준 |
| **Quantum Annealing** | Quantum-assisted | D-Wave Leap 또는 neal (시뮬레이터) |

**공정성 기준**:

- **1차**: objective evaluation count를 모든 optimizer에 동일하게 고정
- **보조**: wall-clock time 기록 (하드웨어·API 차이로 보조지표로만 사용)
- 동일 seed 정책, 동일 거래비용 가정(수수료·슬리피지), 동일 데이터 분할

## 주요 지표 계층

| 계층 | 지표 | 용도 |
| ------ | ------ | ------ |
| 1차 (Primary) | Median OOS Sharpe, IS→OOS decay | 성과 + 일반화 비교의 핵심 endpoint |
| 2차 (Secondary) | MDD, Calmar, turnover, optimization time | 보조 성과 지표 |
| 안정성 | fold 간 parameter dispersion | 파라미터 일관성 |
| 안정성 | local robustness (one-step perturbation OOS drop) | 선택점 주변 평탄도 |
| 과최적화 통제 | CSCV PBO (Bailey et al. 2014) | 정밀 과최적화 확률 |
| 과최적화 통제 | Deflated Sharpe Ratio (DSR) | 다중검정 보정 |
| 과최적화 통제 | IS-OOS Spearman correlation | IS→OOS 일반화 지표 |
| 통계 | Cliff's delta + Holm-Bonferroni correction | Effect size + 다중비교 보정 |
| 국면 분석 | Regime-conditional OOS Sharpe | 비정상성 하 국면별 성능 |

## 데이터

| 구분 | 자산 | 타임프레임 | 역할 |
| ------ | ------ | ----------- | ------ |
| 본 실험 | BTCUSDT | 30m, 1h | Main result table |
| 복제 실험 | ETHUSDT | 30m, 1h | Replication (독립 재실행) |
| 복제 실험 | SOLUSDT | 30m, 1h | Replication (독립 재실행) |

- 데이터 소스: Binance Futures OHLCV (ccxt 또는 공개 데이터셋)
- 외부 자산은 training에 섞지 않고 replication set으로만 다룬다
- 총 6개 asset-timeframe 조합

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 다운로드
python research/data/downloader.py --symbol BTCUSDT --timeframe 30m

# 3. 실험 실행
python research/run_experiment.py

# 4. 결과 확인
# research/artifacts/ 폴더에 fold별 best params, OOS metrics, stability 결과 저장
```

## 주의사항

- **Quantum hardware 접근**: 현재 구현은 `neal` (시뮬레이티드 양자 어닐링) 전용이다. 실제 D-Wave QPU 및 hybrid solver는 future work로 분류한다. 논문에서 `neal`은 simulated quantum annealing임을 반드시 명시해야 한다.
- **재현성**: 모든 실험은 seed를 고정하고, 동일 데이터·동일 budget·동일 분할·동일 비용 가정(수수료·슬리피지)으로 재현 가능해야 한다.
- **실거래 연결**: 이 저장소는 연구용이다. 최적화 결과를 라이브 봇에 적용하려면 충분한 추가 검증이 필요하다.
- **비용**: D-Wave Leap API 호출 시 비용이 발생할 수 있다. budget sensitivity analysis에서 평가 횟수를 조절할 때 참고한다.
