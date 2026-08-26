# RT-1 ONNX 추론 파이프라인

이 문서는 자연어 지시문과 카메라 프레임이 RT-1의 11개 action으로 변환되는
과정을 코드의 실제 tensor shape와 처리 순서에 맞춰 설명합니다.


## 전체 구조

![RT-1 ONNX inference architecture](../assets/rt1_onnx_architecture.svg)


ONNX 정책 네트워크는 다음 세 모델로 구성됩니다.

| 모델 | 역할 | 주요 출력 shape |
| --- | --- | --- |
| `film_efficientnet.onnx` | 언어 조건이 적용된 시각 특징 추출 | `[B x T, 9, 9, 512]` |
| `token_learner.onnx` | 공간 특징을 프레임당 8개 토큰으로 압축 | `[B x T, 8, 512]` |
| `transformer.onnx` | 시계열 image/action sequence에서 action logits 예측 | `[B, 114, 256]` |

공식 `rt-1-main` 체크포인트의 가중치는 각 ONNX 파일에 포함됩니다. 세 ONNX
모델의 전체 크기는 약 182 MiB이며 Universal Sentence Encoder는 별도의
로컬 SavedModel로 저장됩니다.

## 1. 파이프라인 입력

`RT1ONNXPipeline`은 두 가지 입력 형태를 지원합니다.

### 자연어 지시문을 직접 사용하는 경우

```text
images:       uint8 [B, T, H, W, 3]
instructions: str 또는 문자열 batch
```

`predict_instruction()`은 정확히 6프레임을, `predict_episode_instruction()`은
한 프레임 이상의 전체 episode를 입력으로 받습니다.

### 언어 임베딩을 직접 사용하는 경우

```text
images:             uint8   [B, T, H, W, 3]
language_embedding: float32 [B, 512]
```

이 방식은 언어 인코딩과 정책 네트워크를 분리해서 검증할 때 사용합니다.

## 2. 자연어 인코딩

RT-1은 `Universal Sentence Encoder Large /5`를 사용합니다.

```text
instruction string
        |
        v
USE Large /5
        |
        v
float32 [B, 512]
```

모델은 다음 경로에서 지연 로딩됩니다.

```text
models/universal_sentence_encoder_large/5/
```

자연어 인코더는 아직 ONNX로 변환하지 않았습니다. 따라서 이 단계에는
TensorFlow Hub가 필요합니다. 생성된 하나의 언어 임베딩은 같은 episode의
모든 frame에 반복해서 적용됩니다.

## 3. 이미지 전처리

입력 카메라 프레임은 `uint8` RGB 배열입니다. 공식 RT-1 전처리와 동일한
순서로 다음 과정을 수행합니다.

1. dtype 변환
2. 공식 전처리 규칙에 따른 crop
3. `300 x 300` resize
4. ONNX 입력을 위한 연속적인 `float32` 배열 생성

episode의 batch와 time 차원은 이미지 인코더 실행 전에 다음처럼 합쳐집니다.

```text
[B, T, H, W, 3] -> [B x T, 300, 300, 3]
```

언어 임베딩도 각 timestep에 반복하여 다음 context를 만듭니다.

```text
[B, 512] -> [B x T, 512]
```

전처리의 dtype, crop 및 resize 결과는 공식 TensorFlow 출력과 별도로 비교할
수 있습니다.

## 4. FiLM-EfficientNet

FiLM-EfficientNet은 이미지와 언어 context를 함께 입력받습니다.

```text
image:   [B x T, 300, 300, 3]
context: [B x T, 512]
                       |
                       v
features: [B x T, 9, 9, 512]
```

FiLM conditioning을 통해 같은 이미지라도 자연어 지시문에 따라 다른 시각
특징을 생성할 수 있습니다.

## 5. TokenLearner

TokenLearner는 `9 x 9` 공간 특징을 프레임당 8개의 image token으로
압축합니다.

```text
[B x T, 9, 9, 512]
          |
          v
[B x T, 8, 512]
```

그다음 batch와 time 차원을 복원합니다.

```text
[B x T, 8, 512] -> [B, T, 8, 512]
```

## 6. 6프레임 image history

RT-1은 최근 6 timestep을 Transformer 문맥으로 사용합니다.

```text
TIME_STEPS = 6
IMAGE_TOKENS = 8
EMBEDDING_DIM = 512
```

episode 시작 시 history는 0으로 초기화됩니다. 첫 6프레임 동안 새 image
token이 앞에서부터 채워지며, 7번째 프레임부터는 가장 오래된 프레임을
제거하고 새 프레임을 마지막 위치에 추가합니다.

```text
t < 6:  history[:, t] = current image tokens
t >= 6: history를 왼쪽으로 이동한 뒤 마지막 slot에 새 토큰 추가
```

각 frame의 최종 history shape는 다음과 같습니다.

```text
[B, 6, 8, 512]
```

episode 초반에는 현재까지 채워진 timestep에 해당하는 action 위치에서
출력을 선택하고, 6프레임이 채워진 뒤에는 마지막 timestep의 action을
선택합니다.

## 7. Transformer sequence 구성

각 timestep에는 8개의 image token과 11개의 빈 action slot을 결합합니다.

```text
8 image tokens + 11 action slots = 19 tokens per timestep
```

6 timestep을 연결하면 전체 sequence 길이는 114가 됩니다.

```text
[B, 6, 8, 512]
        +
[B, 6, 11, 512]  # zero-initialized action slots
        |
        v
[B, 6, 19, 512]
        |
        v
[B, 114, 512]
```

상수 정의는 다음과 같습니다.

```text
TOKENS_PER_STEP = 8 + 11 = 19
SEQUENCE_LENGTH = 6 x 19 = 114
```

## 8. Attention mask

Transformer에는 `[114, 114]` 크기의 attention mask가 함께 입력됩니다.
mask는 기본적인 lower-triangular causal 구조를 사용하면서 action slot 사이의
접근을 RT-1의 autoregressive 규칙에 맞게 추가로 제한합니다.

이 mask는 다음을 목적으로 합니다.

- 미래 timestep의 token 참조 방지
- 현재 action을 예측할 때 허용되지 않은 이전 action slot 참조 방지
- 같은 timestep에서 아직 생성되지 않은 action 정보 사용 방지

attention mask는 실행 중 변하지 않으므로 pipeline 초기화 시 한 번 생성하여
재사용합니다.

## 9. Transformer logits

Transformer ONNX 모델의 입력과 출력은 다음과 같습니다.

```text
sequence:       float32 [B, 114, 512]
attention_mask: float32 [114, 114]
                         |
                         v
logits:         float32 [B, 114, 256]
```

`256`은 RT-1 action vocabulary 크기입니다. 각 action 위치에서 가장 큰
logit의 인덱스를 선택하여 이산 action token을 얻습니다.

```text
argmax(logits, axis=-1) -> token in [0, 255]
```

현재 timestep에 해당하는 11개 위치만 선택하므로 최종 token shape는
`[B, 11]`입니다.

## 10. 11개 action token 복원

11개 token은 다음 순서로 해석됩니다.

| Token 위치 | 출력 항목 | 차원 | 복원 범위 |
| ---: | --- | ---: | --- |
| 0 | `terminate_episode` | 1 | 3개 mode의 one-hot 값 |
| 1–3 | `world_vector` | 3 | 각각 `[-1, 1]` |
| 4–6 | `rotation_delta` | 3 | 각각 `[-pi/2, pi/2]` |
| 7 | `gripper_closedness_action` | 1 | `[-1, 1]` |
| 8–9 | `base_displacement_vector` | 2 | 각각 `[-1, 1]` |
| 10 | `base_displacement_vertical_rotation` | 1 | `[-pi, pi]` |

연속 action은 vocabulary token `0–255`를 각 action의 최소·최대 범위로 선형
매핑하여 복원합니다.

```text
normalized = token / 255
value = normalized x (maximum - minimum) + minimum
```

`terminate_episode`는 token `0`, `1`, `2`를 세 가지 mode의 one-hot 값으로
변환합니다. 범위를 벗어난 token은 mode `0`으로 처리합니다.

최종 action dictionary는 다음 항목을 포함합니다.

```text
terminate_episode
world_vector
rotation_delta
gripper_closedness_action
base_displacement_vector
base_displacement_vertical_rotation
```

## 11. 전체 episode 추론

사용자 추론 entry point는 episode의 모든 PNG frame을 읽어 batch 차원을
추가하고, 자연어 지시문과 함께 rolling-history pipeline을 실행합니다.

```bash
python onnx/validation/validate_episode.py \
  --episode-index 1 \
  --instruction "close middle drawer"
```

사용자가 `--instruction`을 생략하면 episode의 `metadata.json`에 저장된
지시문을 사용합니다.

출력 JSON:

```text
validation_artifacts/episode_00001/episode/onnx.json
```

JSON에는 다음 정보가 저장됩니다.

- episode 인덱스
- 실제 추론에 사용한 자연어 지시문
- 전체 frame 수
- frame별 11개 action token
- frame별 복원된 연속 action

## 12. 검증 경계

파이프라인의 각 단계는 공식 TensorFlow 출력과 독립적으로 비교할 수 있도록
구성되어 있습니다.

- dtype 변환과 crop
- `300 x 300` resize
- USE Large `/5` 임베딩
- FiLM-EfficientNet 특징
- TokenLearner image token
- 6프레임 history
- Transformer sequence
- attention mask
- Transformer logits
- action token
- 연속 action

전체 검증 명령, 허용 오차 및 66프레임 end-to-end 결과는
[`../comparison/README.md`](../comparison/README.md)를 참고하세요.

## 현재 범위

- USE Large `/5`는 TensorFlow SavedModel로 실행합니다.
- 세 정책 모델은 ONNX Runtime의 `CPUExecutionProvider`를 사용합니다.
- RT-1 출력은 end-effector와 mobile-base action이며 개별 arm joint 값이
  아닙니다.
- MuJoCo 시각화는 action을 이해하기 위한 도식 모델이며 원본 EDR 로봇의
  실제 관절 구조나 trajectory를 복원하지 않습니다.
