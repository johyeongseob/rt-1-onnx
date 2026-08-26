# TensorFlow-ONNX 비교 검증

이 디렉터리에는 모듈식 ONNX RT-1이 공식 TensorFlow `rt1main`의 동작을
그대로 보존하는지 확인하기 위한 프레임워크 간 비교 도구가 포함되어 있습니다.
모든 명령은 WSL 가상환경을 활성화한 뒤 저장소 최상위 디렉터리에서 실행합니다.

## 검증 방식

각 단계는 다음과 같은 방식으로 검증합니다.

```text
공식 TensorFlow 출력 -> validation_artifacts/.../tensorflow.*
ONNX 출력            -> validation_artifacts/.../onnx.*
출력 비교             -> Match: True
```

## 전처리

```bash
python official/validation/validate_preprocessors.py
python onnx/validation/validate_preprocessors.py
python comparison/preprocessors.py

python official/validation/validate_resize.py
python onnx/validation/validate_resize.py
python comparison/resize.py
```

## 언어 임베딩

USE Large `/5`로 `close middle drawer`를 임베딩한 뒤 episode 1에 저장된
임베딩과 비교합니다.

```bash
python official/validation/validate_use_embedding.py
```

작은 부동소수점 오차는 정상입니다. 검증 결과는 `rtol=1e-5`, `atol=1e-6`
조건에서 일치합니다.

## FiLM-EfficientNet

```bash
python official/validation/validate_film_efficientnet.py
python onnx/validation/validate_film_efficientnet.py
python comparison/film_efficientnet.py
```

예상 출력 형상은 `[1, 9, 9, 512]`입니다.

## TokenLearner

```bash
python official/validation/validate_token_learner.py
python onnx/validation/validate_token_learner.py
python comparison/token_learner.py
```

예상 출력 형상은 `[1, 8, 512]`입니다.

## 6프레임 이미지 히스토리

```bash
python official/validation/validate_image_history.py
python onnx/validation/validate_image_history.py
python comparison/image_history.py
```

예상 출력 형상은 `[1, 6, 8, 512]`입니다.

## Transformer

114개 토큰으로 구성된 입력 시퀀스와 causal attention mask를 생성한 뒤
TensorFlow와 ONNX Transformer의 출력을 비교합니다.

```bash
python comparison/prepare_transformer_input.py
python official/validation/validate_transformer.py
python onnx/validation/validate_transformer.py
python comparison/transformer.py
```

예상 출력 형상은 `[1, 114, 256]`입니다.

ONNX 이미지 히스토리로 입력 시퀀스를 명시적으로 생성하려면 다음과 같이
실행합니다.

```bash
python comparison/prepare_transformer_input.py --source onnx
```

## Action 디코딩

```bash
python official/validation/validate_action_decoder.py
python onnx/validation/validate_action_decoder.py
python comparison/action.py
```

디코더는 다음 값을 생성합니다.

- `terminate_episode`
- `world_vector` (`x`, `y`, `z`)
- `rotation_delta` (`roll`, `pitch`, `yaw`)
- `gripper_closedness_action`
- `base_displacement_vector` (`x`, `y`)
- `base_displacement_vertical_rotation` (`yaw`)

## End-to-end: 최초 6프레임

공식 SavedModel 정책과 연결된 ONNX 파이프라인을 각각 실행한 뒤 최종 action을
비교합니다.

```bash
python official/validation/validate_end_to_end.py
python onnx/validation/validate_end_to_end.py
python comparison/end_to_end_models.py
```

재현 가능한 비교를 위해 `--instruction`을 생략하여 두 파이프라인 모두 episode의
지시문을 사용하게 합니다. ONNX 진입점은 episode에 미리 계산되어 저장된
`language_embedding.npy`를 불러오는 대신, metadata의 지시문을 로컬 USE Large
`/5` SavedModel로 인코딩합니다.

## End-to-end: 전체 episode

episode 1의 전체 66프레임을 각 파이프라인으로 추론하고 생성된 JSON 파일의
모든 action을 비교합니다.

```bash
python official/validation/validate_episode.py
python onnx/validation/validate_episode.py
python comparison/end_to_end_episode.py
```

예상 결과는 다음과 같습니다.

```text
Frames compared: 66
Token mismatch frames: []
Action mismatches: []
Maximum absolute action error: 2.384185791015625e-07
Match: True
```

출력 파일은 다음 경로에 저장됩니다.

```text
validation_artifacts/episode_00001/episode/official.json
validation_artifacts/episode_00001/episode/onnx.json
```
