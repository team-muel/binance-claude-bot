# 연구 질문, 가설, 범위

## 작업 제목

Quantum Annealing-Based Parameter Search for Rule-Based Crypto Trading:
Robustness Under Non-Stationarity

## 핵심 연구 질문

**RQ**: 비정상적이고 구조 변화가 심한 암호화폐 시장에서, QUBO 인코딩 기반
양자 어닐링 파라미터 탐색이 고전적 탐색 방법 대비 walk-forward
re-optimization 하에서 더 강건한 OOS 일반화를 달성하는가?

## 가설

- **H1 (일반화)**: QA 기반 탐색은 fold 전반에서 IS→OOS performance decay가
  고전적 baseline보다 유의하게 작다.
- **H2 (파라미터 안정성)**: QA는 rolling fold 간 선택 파라미터의 dispersion이
  더 낮고, one-step perturbation 하에서 OOS 성능 저하가 더 작다.
- **H3 (국면 조건부)**: QA의 상대적 이점은 고변동성(high-volatility) 국면에서
  더 두드러진다.

## 범위 내

- 이산 EMA-RSI-ATR 규칙 전략을 위한 QUBO 방식 파라미터 인코딩
- Grid, Random, TPE, Classical SA와의 동일 예산 비교
- BTCUSDT / ETHUSDT / SOLUSDT × 30m / 1h walk-forward OOS 평가
- Rolling volatility quantile 기반 시장 국면 분류 및 국면 조건부 분석
- QA / Classical SA ablation study (num_reads, penalty, schedule, temperature)
- Convergence trace 분석 및 search space coverage 분석
- 비용 민감도 분석 (수수료/슬리피지 시나리오)
- Funding rate post-hoc 영향 추정
- Execution delay sensitivity 분석
- 안정성, 과최적화, 통계 비교 지표 (Cliff's delta, Holm correction, CSCV PBO)
- IS-OOS correlation 분석
- Parameter trajectory 시각화
- Failure analysis ("When does QA fail?")

## 범위 외

- QAOA 비교 (future work)
- QPCA 기반 리스크 통제
- 딥러닝 예측 모델 (LSTM, PatchTST, QLSTM 등)
- 멀티에이전트 거래 시스템 아키텍처
- 실시간 실행 성능을 연구 기여로 내세우는 것
- 실제 D-Wave QPU / hybrid solver (neal 시뮬레이터만 사용, limitation으로 명시)

## 주요 주장 방향

논문의 핵심 서사는 단순 수익률 극대화가 아니라 **비정상 시계열에서의
파라미터 일반화 가능성과 OOS 강건성**이다.

- 1차 증거: IS→OOS decay 관점의 일반화 비교
- 2차 증거: 파라미터 안정성 (dispersion + perturbation sensitivity)
- 3차 증거: 국면 조건부 성능 차이
- 구조적 증거: Optimization landscape shift 시각화, convergence trace, CSCV PBO
- 외부 타당성: 다자산 × 다타임프레임 × 다국면 consistency
