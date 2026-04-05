# 논문 구조, 표, 그림 계획

## 목표 섹션 순서

1. 서론
2. 관련 연구
3. 방법론
4. 실험 설계
5. 결과
6. 논의
7. 결론

## 세부 섹션 구조

### 1. Introduction

- 비정상 금융 시계열에서 규칙기반 전략 파라미터 최적화의 본질적 어려움
- Optimization landscape shift 동기 (fold별 최적점 이동)
- QUBO 인코딩을 통한 양자/고전 어닐링 비교의 학술적 가치
- 연구 질문(RQ), 가설(H1–H3), 기여 요약

### 2. Related Work

- 양자 컴퓨팅과 금융 최적화 (Rosenberg, Venturelli, Orus 등)
- 규칙기반 전략의 파라미터 최적화 문헌
- Hyperparameter robustness / walk-forward optimization 문헌
- 과최적화 통제: PBO (Bailey et al.), Deflated Sharpe (Harvey & Liu)

### 3. Methods

- 3.1 전략 규칙 및 파라미터 공간 (EMA-RSI-ATR)
- 3.2 QUBO 인코딩 (one-hot binary, penalty 구조)
- 3.3 Optimizer family (Grid, Random, TPE, Classical SA, QA)
- 3.4 Walk-forward protocol (IS/OOS 분할, 공정성 기준)
- 3.5 시장 국면 분류 (rolling volatility quantile, 3분류)
- 3.6 비정상성 정량화 (fold별 regime label, IS-OOS landscape shift)

### 4. Experimental Design

- 4.1 데이터 (3 assets × 2 timeframes = 6 setups)
- 4.2 공정성 기준 (equal budget, same seed, same splits, same costs)
- 4.3 지표 계층 (primary: OOS Sharpe & IS-OOS decay, secondary: MDD/Calmar/turnover)
- 4.4 Ablation design (QA: num_reads, penalty, schedule; SA: temp, cooling)
- 4.5 비용 민감도 설계 (low/base/high cost scenarios)
- 4.6 Execution delay 설계 (±1~2 bar lag)

### 5. Results

- 5.1 Main OOS comparison (30m, 3 assets)
- 5.2 Regime-conditional results (high/medium/low vol)
- 5.3 QA / SA ablation results
- 5.4 Convergence trace analysis
- 5.5 Stability & robustness (dispersion, perturbation, CSCV PBO)
- 5.6 Cross-setup consistency (6 asset-tf combinations)
- 5.7 1h timeframe results
- 5.8 Cost sensitivity & execution delay results
- 5.9 IS-OOS correlation analysis

### 6. Discussion

- 6.1 Interpretation: 비정상성 하에서 QA의 구조적 특성
- 6.2 When does QA fail? (ETH adverse case, regime analysis)
- 6.3 Limitations (neal vs QPU, no funding in main, no QAOA)
- 6.4 Practical implications (재최적화 주기, 비용 경계)
- 6.5 Reproducibility statement

### 7. Conclusion

## 주요 표 (14개)

| # | 표 | 상태 |
| --- | ------ | ------ |
| T1 | 전략 규칙 및 파라미터 공간 | 기존 |
| T2 | Optimizer 비교 및 공정성 기준 | 기존 |
| T3 | Walk-forward fold 요약 (regime label 포함) | 확장 |
| T4 | BTCUSDT 30m 본 실험 OOS 결과 | 기존 |
| T5 | ETHUSDT / SOLUSDT 30m 복제 결과 | 기존 |
| T6 | 1h 타임프레임 결과 (3 assets) | 신규 |
| T7 | QA ablation (num_reads, penalty, schedule) | 신규 |
| T8 | 안정성 진단 (dispersion, perturbation, CSCV PBO) | 확장 |
| T9 | 통계 검정 (bootstrap, Wilcoxon, Cliff's delta, Holm correction) | 확장 |
| T10 | Cross-setup consistency matrix | 신규 |
| T11 | 비용 민감도 (low/base/high cost) | 신규 |
| T12 | Regime-conditional OOS Sharpe | 신규 |
| T13 | IS-OOS correlation by optimizer | 신규 |
| T14 | Funding rate post-hoc 영향 추정 | 신규 |

## 주요 그림 (10개)

| # | 그림 | 상태 |
| --- | ------ | ------ |
| F1 | Walk-forward 도식 (regime 색칠 포함) | 확장 |
| F2 | Optimization landscape shift across folds (2D heatmap) | 신규 |
| F3 | OOS Sharpe 분포 (optimizer별 violin/box) | 기존 |
| F4 | Regime-conditional OOS Sharpe boxplot | 신규 |
| F5 | Convergence traces by optimizer | 신규 |
| F6 | Parameter dispersion 비교 | 기존 |
| F7 | Parameter trajectory across folds | 신규 |
| F8 | IS vs OOS Sharpe scatter by optimizer | 신규 |
| F9 | Local robustness heatmap | 기존 |
| F10 | Budget sensitivity curve | 기존 |

## 작성 순서

1. 코드 모듈 구현 및 테스트
2. 1h 데이터 다운로드 + 전체 실험 재실행
3. Ablation 실험 실행
4. 비용/지연 민감도 실험 실행
5. 통계 검정 및 시각화 최종 확정
6. Results → Discussion → Methods → Introduction 순 작성

## 메모

논문의 핵심 서사는 "비정상 시계열에서 재최적화 후에도 파라미터가 얼마나 안정적인가"와
"IS에서 찾은 최적점이 OOS로 얼마나 일반화되는가"이다.
수익률 자체보다 일반화 가능성(IS-OOS decay)와 구조적 증거(landscape shift, convergence,
CSCV PBO)를 전면에 내세운다.
