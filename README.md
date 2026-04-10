# 목적

현재 작업의 목적은 논문 *GPyro: uncertainty-aware temperature predictions for additive manufacturing* 를 완전 재현하기보다, 계산 시간을 줄인 경량 프로토타입으로 논문 방향성과 일반화 가능성을 먼저 확인하는 것이다.

# 현재까지 만든 방식 정리

## 1. 개발용 프로토타입: 여러 실험을 시간축으로 연결한 방식

대상 파일: `GPyro_development_prototype.ipynb`

이 방식은 `T1~T10` 실험을 시간 순으로 이어붙여 하나의 긴 시계열처럼 만든 뒤, 그 안에서 train/val/test를 시간 구간 기준으로 나눈다.

특징:

- 여러 실험을 합쳐 더 많은 학습 데이터를 확보한다.
- split 기준은 실험 ID가 아니라 시간 순서다.
- 평가 의미는 "같은 merged timeline 내부의 미래 구간 예측"에 가깝다.
- rollout 평가도 포함되어 있다.

장점:

- 빠르게 학습 곡선과 기본 성능을 확인하기 좋다.
- 데이터 양이 늘어나 초기 실험에 유리하다.

한계:

- 실험 단위 일반화 평가가 아니다.
- 논문처럼 "보지 않은 다른 실험"에 대한 성능이라고 해석하기 어렵다.
- 여러 실험을 이어붙이므로 실험 경계가 모델링 관점에서 자연스럽지 않을 수 있다.

## 2. 실험 단위 일반화: 1개 학습, 1개 평가

대상 파일: `GPyro_experiment_generalization_1train_1test.ipynb`

이 방식은 하나의 실험으로만 학습하고, 다른 하나의 실험으로만 평가한다. 예를 들어 `T1`으로 학습하고 `T2`로 평가한다.

특징:

- train experiment와 test experiment가 완전히 분리된다.
- 여러 실험을 하나로 이어붙이지 않는다.
- 시간축 70/15/15 분할을 사용하지 않는다.
- rollout 평가는 제거하고, unseen test experiment 전체에 대한 one-step 성능만 본다.

장점:

- 논문의 평가 철학인 "실험 단위 일반화"에 더 가깝다.
- 빠르게 가능성을 확인하는 축소 실험으로 적합하다.
- 특정 실험 쌍에서 일반화가 되는지 바로 확인할 수 있다.

한계:

- 평가 실험이 1개뿐이라 편차가 매우 클 수 있다.
- 한 번의 결과만으로 논문 성능과 직접 비교하기는 어렵다.

## 3. 실험 단위 일반화: 1개 학습, 여러 개 평가

대상 파일: `1train_3test_evaluation.py`

이 방식은 하나의 실험으로 학습한 뒤, 여러 개의 unseen experiment에 대해 각각 평가해서 DTW-MARE를 여러 개 수집하는 구조다.

특징:

- 예: `T1` 학습 후 `T2`, `T3`, `T4` 평가
- 각 테스트 실험에 대해 개별 DTW-MARE를 계산
- 여러 결과를 모아 평균과 분산 또는 표준편차를 계산 가능

장점:

- 현재 만든 방식 중 논문 평가 방향과 가장 유사하다.
- 단일 test experiment보다 결과 안정성이 더 높다.
- 실험 간 편차를 파악할 수 있다.

한계:

- 여전히 논문 전체 규모보다 작은 축소판이다.
- 테스트 실험 수가 적으면 여전히 통계적 안정성이 부족할 수 있다.

# 방식별 핵심 차이

## 모델 측면

현재 비교 중인 노트북/스크립트들은 모두 본질적으로 같은 경량 프로토타입 모델을 사용한다.

공통점:

- `ThermalMLP` 기반
- 현재 온도장과 torch 상태, boundary feature를 입력으로 사용
- 다음 시점 온도를 예측하는 one-step 구조
- 주 평가 지표로 DTW-MARE 사용

즉, 현재 차이의 핵심은 모델 구조 차이보다 데이터 구성과 평가 프로토콜 차이에 있다.

## 평가 프로토콜 측면

### 개발용 프로토타입

- 여러 실험을 연결한 뒤 시간 구간으로 분할
- 같은 merged timeline 내부에서 미래 구간 성능 평가
- 논문과 직접 비교하기 어려움

### 1 train / 1 test

- 실험 단위 분리
- 보지 않은 다른 실험에 대한 일반화 평가
- 논문 방향과 더 유사하지만 결과가 1개뿐임

### 1 train / multiple test

- 실험 단위 분리
- 여러 unseen experiment에서 성능 수집
- 평균과 분산을 계산할 수 있어 논문 비교에 가장 적합함

# 논문과의 관계

논문은 하나의 실험으로 학습하고, 여러 다른 실험에서 예측 성능을 평가하는 방식으로 읽힌다. 따라서 현재 구현 중에서는 `1 train / multiple test` 구조가 가장 논문에 가깝다.

다만 현재 프로토타입은 논문의 full GPyro 모델 자체를 그대로 재현한 것은 아니다. 따라서 논문과 같은 수준의 수치가 바로 나오지 않더라도, 우선은 다음 두 가지를 확인하는 것이 중요하다.

확인할 것:

- 실험 단위 일반화가 가능한지
- DTW-MARE 기준으로 논문 방향성과 유사한 경향이 보이는지

# 성능 정리

```
개발용 프로토타입: GPyro_development_prototype.ipynb
=== Test (one-step) ===
DTW-MARE (primary): 4.0981 %
MAE: 33.5862 °C   RMSE: 44.2704 °C

=== Test rollout (H=48) ===
DTW-MARE (rollout): 78480653.9907 %
MAE: 1373924270.0273 °C   RMSE: 6464733768.7984 °C
per-node DTW-MARE: min=1.707% max=14.382%

단일 실험 프로토타입: GPyro_single_experiment_prototype.ipynb
=== Test (one-step) ===
DTW-MARE (primary): 7.3538 %
MAE: 52.4554 °C   RMSE: 67.1368 °C

=== Test rollout (H=48) ===
DTW-MARE (rollout): 2223.6974 %
MAE: 11850.1942 °C   RMSE: 28759.5317 °C
per-node DTW-MARE: min=1.351% max=28.805%

실험 단위 일반화: 4_1Train_3Test
=== Test on T2 ===
DTW-MARE: 1.1963 %
MAE: 10.1691 C   RMSE: 14.4019 C

=== Test on T3 ===
DTW-MARE: 1.2266 %
MAE: 9.4777 C   RMSE: 13.3720 C

=== Test on T4 ===
DTW-MARE: 1.0769 %
MAE: 8.2796 C   RMSE: 11.7693 C

=== Summary across all test experiments ===
Train experiment: T1
Test experiments: ['T2', 'T3', 'T4']
DTW-MARE mean: 1.1666 %
DTW-MARE std: 0.0646 %
Individual DTW-MAREs: ['1.20%', '1.23%', '1.08%']

논문 reported result
DTW 기반 mean relative distance: 2.42%
variance: 1.38%
```

## 성능 해석

- `GPyro_development_prototype.ipynb` 는 one-step 기준으로는 4.0981%까지 내려갔지만, rollout 성능이 매우 크게 붕괴했다. 따라서 긴 horizon 예측에는 현재 설정이 안정적이지 않다.
- `GPyro_single_experiment_prototype.ipynb` 는 one-step 성능도 7.3538%로 더 높고, rollout도 크게 불안정하다.
- `4_1Train_3Test` 는 실험 단위 일반화 기준에서 DTW-MARE 평균 1.1666%를 보였고, 세 테스트 실험 간 편차도 작았다.
- 현재까지 얻은 결과 중에서는 `1 train / multiple test` 구조가 가장 논문 방향과 유사하면서도 수치도 가장 안정적으로 보인다.
- 다만 현재 모델은 논문의 full GPyro 자체가 아니라 경량 프로토타입이므로, 이 수치가 곧바로 논문 재현을 의미하지는 않는다.
- 또한 논문은 다수의 검증 실험 전체에서 평균과 분산을 보고했기 때문에, 현재 3개 테스트 결과는 논문보다 더 작은 축소 평가라는 점을 함께 고려해야 한다.
