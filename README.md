# AIproject : 비판적 재평가 텍스트를 통한 Planner 성능 개선 Dual-System VLA 모델

> OpenVLA(Planner)와 SmolVLM 500B(Actor)을 결합한 듀얼 시스템 VLA(Vision-Language-Action) 모델.
> Planner가 생성한 action을 Actor가 **비판적으로 재평가**하여 조작 성능을 개선하는 구조를 제안하고 구현함.

# 프로젝트 소개

기존 Vision-Language-Action(VLA) 모델은 현재 환경 이미지와 작업 명령을 입력받아 로봇 행동(Action Token)을 생성한다.

하지만 대부분의 모델은 생성한 행동을 그대로 수행하며, 행동의 적절성을 스스로 평가하거나 수정하는 과정이 존재하지 않는다.

본 연구는 Planner가 생성한 행동을 Critic Actor가 분석하고, 자연어 기반 비판적 설명(Critique)을 생성한 뒤 수정된 행동(Action Token)을 다시 생성하는 Dual-System 구조를 제안한다.

Critic Actor는 행동의 문제점을 자연어로 설명하고 수정 방향을 제시하며, 강화학습을 통해 행동 수정 정책(Action Refinement Policy)을 지속적으로 개선한다.

실제 로봇 환경 구축 비용을 고려하여 LIBERO 시뮬레이션 환경에서 실험을 수행하였다.

# 시스템 구조

![Figure1](https://github.com/user-attachments/assets/edbb073d-d40e-452a-873f-a4de54bc41b6)
- **1. Model Structure**

전체 시스템은 Planner와 Critic Actor로 구성된다.

Planner는 행동을 생성하고,
Critic Actor는 행동을 분석하여 Critique를 생성한 뒤 수정된 행동을 출력한다.

수정된 행동은 LIBERO 환경에서 평가되며,
GRPO를 통해 행동 수정 정책을 학습한다.

# 핵심 구조

![Figure2](https://github.com/user-attachments/assets/15b2c0a8-c4bb-4968-bf58-09bde7475a99)
- **2. Model Process**


Critic Actor는 다음 세 가지 정보를 동시에 입력받는다.

- 환경 이미지
- 작업 명령(Task Instruction)
- Planner가 생성한 Action Token

이를 기반으로

- 비판적 설명(Critique)
- 수정된 Action Token

을 생성한다.

이를 통해 행동 생성과 행동 평가를 동시에 수행할 수 있다.

![Figure3](https://github.com/user-attachments/assets/cf47ba2e-f4df-4c42-9f32-b77c0ebf5a60)
- **3. Projection Layer**

 OpenVLA의 Action Embedding은 4096차원으로 구성되어 있으며, SmolVLM은 960차원의 임베딩 공간을 사용한다. 따라서 두 모델 간 정보를 공유하기 위해 차원 변환 과정이 필요해 LLaVA의 Projection Layer 구조를 참고함.

---
![Figure4](https://github.com/user-attachments/assets/e5adbf70-5ef6-4e22-94d6-8828ed1dc6be)
- **4. FLow Chart**

 Planner와 Actor는 ZeroMQ를 통해 통신하며, 학습에 사용되는 이미지 및 텍스트 데이터는 모두 LIBERO 환경에서 수집된다.

Actor 초기화 과정에서는 OpenVLA의 256개 Action Token을 추가하고, Action Embedding Table을 생성한 뒤 Projection Layer를 연결하여 Tokenizer를 구성한다.

강화학습 과정은 RLinf 프레임워크를 기반으로 구현되었다.

먼저 collect_rollout() 단계에서 LIBERO 환경으로부터 데이터를 수집하며, Actor는 비판적 텍스트와 수정된 Action Token을 생성한다. 생성된 행동은 환경에서 실행되며, 수행 결과를 바탕으로 Reward와 Loss가 계산된다.

이후 compute_grpo_loss() 단계에서 정책 업데이트가 수행되며, LoRA 파라미터가 학습된다.
 
---
![Figure5](https://github.com/user-attachments/assets/1c0fbf23-fe96-475a-8827-3fd545224f3e)
- **5. Actor Tokenizer Process**

 이미지 데이터는 Vision Encoder를 통과하여 Vision Embedding으로 변환되며, 텍스트 데이터는 Language Model을 통해 Text Embedding으로 변환된다.

Planner에서 전달받은 Action Token은 OpenVLA Action Embedding Table을 통해 임베딩으로 변환된 뒤 Projection Layer를 거쳐 SmolVLM 임베딩 공간으로 매핑된다.

이후 Image Embedding, Text Embedding, Action Embedding은 Input Merge와 Action Injection Hook을 통해 하나의 입력 시퀀스로 결합된다.

결합된 입력은 Transformer를 통과하며, 최종적으로 Critique와 수정된 Action Token을 생성한다.

---
![Figure6](https://github.com/user-attachments/assets/f106f294-3e26-4129-ab98-1bde2140310e)
- **6. Suprevised Fine Tuning**

초기 상태의 모델은 Critique 형식과 Action Token 출력 형식을 학습하지 않은 상태이므로, 바로 GRPO 학습을 수행할 경우 지속적으로 패널티를 받아 학습이 불안정해질 수 있다.

이를 해결하기 위해 SFT를 먼저 수행하여 기본적인 출력 형식을 학습하도록 하였다.

SFT 단계에서는 다음 능력을 우선적으로 학습한다.

Critique 생성 형식 학습
Action Token 출력 형식 학습
이미지 및 명령 기반 행동 분석
행동 수정 패턴 학습

# 연구 기여점

본 연구의 주요 기여는 다음과 같다.

1. Planner와 Critic Actor로 구성된 Dual-System 구조 제안

2. 자연어 기반 Critique를 활용한 행동 수정(Action Refinement) 구조 구현

3. OpenVLA Action Token을 Critic Actor에 이식하기 위한 Tokenizer 확장 및 Projection Layer 구현

4. Critique 생성과 Action Refinement를 동시에 수행하는 학습 구조 설계

5. LoRA 및 4bit Quantization을 적용하여 제한된 GPU 환경에서도 학습 가능하도록 구현

# 실험 환경

| 구성 요소 | 모델 |
|------------|------------|
| Planner | OpenVLA-7B |
| Critic Actor | SmolVLM2-500M |
| 강화학습 | GRPO |
| 시뮬레이션 | LIBERO-10 |
| 파라미터 효율 학습 | LoRA |
| 양자화 | 4bit Quantization |
| 통신 | ZeroMQ |

# 학습 과정

본 연구는 다음과 같은 순서로 학습을 수행하였다.

### 1. Action Embedding 추출

OpenVLA Action Embedding을 추출하여 Actor 모델에 이식

### 2. SFT

Critique 생성 형식과 Action Token 출력 형식을 학습

예시

[CRITIQUE]
Action may collide with the object.
[/CRITIQUE]

[ACTION]
<action_15>
<action_32>
...
[/ACTION]

### 3. GRPO 강화학습

Critique 기반 Action Refinement 정책 학습

학습 흐름

Planner
↓
Critique Generation
↓
Action Refinement
↓
LIBERO Evaluation
↓
Reward Calculation
↓
GRPO Update

# Reward Function

보상 함수는 다음 요소를 조합하여 계산한다.

| 항목 | Reward |
|--------|--------|
| Task Success | +1.0 |
| Collision | -0.5 |
| Stability | +0.3 |
| Invalid Action | -1.0 |
| Time Penalty | -0.01 × overtime |

최종 Reward를 이용하여 GRPO 정책 업데이트를 수행한다.

# 📁 openvla_planner

### openvla_inference_code.py

OpenVLA Planner 추론 서버.

* ZeroMQ 기반 서버 실행
* Planner 전용 추론 수행
* CPU 환경에서 동작
* OpenVLA Action Token 생성

Model

* OpenVLA-7B Finetuned on LIBERO-10

Input

* RGB Image
* Task Instruction

Output

* Planner Action Tokens (7 Tokens)

---

### action_tokenizer.py

OpenVLA 원본 Action Tokenizer.

본 프로젝트에서 사용하는 Action Token 구조를 확인하기 위해 유지하였다.

주요 역할

* Action Token 정의 확인
* Action Embedding 추출
* Token ↔ Action 변환 확인

---

# 📁 qwen_actor

Qwen 기반 Critic Actor 구현.

### actor_action_tokenizer.py

* OpenVLA Action Token 256개 추가
* Planner Action Embedding 로드
* Projection Layer 적용
* Qwen 입력 임베딩과 결합

### projection_layer.py

```text
OpenVLA Hidden Size : 4096
Qwen Hidden Size    : 2048
```

LLaVA Projection Layer 구조를 참고하였다.

### actor_model.py

* Qwen2.5-VL 기반
* 4bit Quantization
* LoRA Fine-Tuning
* ZeroMQ 통신
* Critique 생성
* Action Token 수정

---

# 📁 SmolVLM_actor

Qwen 모델의 높은 VRAM 사용량 문제를 해결하기 위해 구현하였다.

### smol_action_tokenizer.py

* Action Token 등록
* Action Embedding 초기화
* Action Token 변환

### smol_projection_layer.py

```text
OpenVLA Hidden Size : 4096
SmolVLM Hidden Size : 960
```

LLaVA Projection Layer를 참고하였다.

### smol_actor_model.py

* SmolVLM2-500M
* 4bit Quantization
* LoRA
* ZeroMQ
* Critique 생성
* Action Token 수정

---

# 📁 train

강화학습 실행 파일

### train.py

* LIBERO 환경 생성
* Rollout 수집
* Reward 계산
* GRPO 학습
* Actor 업데이트

### smol_train.py

RLinf 구조 참고

주요 함수

```python
collect_rollout()
compute_grpo_loss()
```

### smol_sft.py

GRPO 이전 SFT 단계

목적

* Critique 형식 학습
* Action Token 형식 학습
* RL 안정성 향상

---

# 📁 assets

### make_embeddings.py

OpenVLA Action Embedding 추출

생성 파일

```text
openvla_action_embeddings.pt
```

---

# 📁 checkpoints

### sft

초기 체크포인트

* 비전 인코더 미사용
* 현재 사용하지 않음

### sft2

최종 체크포인트

* 비전 인코더 포함
* Critique 생성 가능
* Action Token 생성 가능

---

# 📁 logs

저장 항목

* Reward
* Success Rate
* Loss
* Learning Rate

---

# 실험 결과

# 한계점

- 실제 로봇 환경 검증 미수행
- Critique 품질에 따른 성능 편차 존재
- 장기 작업(Long Horizon Task)에 대한 추가 검증 필요
- Planner 오류가 심한 경우 Critique만으로 복구가 어려움

- # 향후 연구

- 실제 로봇 환경 적용
- Multi-Step Critique 생성
- Human Feedback 기반 학습
- 대형 Vision-Language 모델 적용
- 계층적 Planner 구조 연구
- Critique 품질 평가 지표 개발
