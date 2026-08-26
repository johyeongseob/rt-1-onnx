# 공식 RT-1 체크포인트의 ONNX 변환과 추론

이 문서는 공식 Google `rt1main` 체크포인트를 세 개의 모듈식 ONNX 모델로
변환하고, 생성된 모델을 연결하여 자연어 지시문으로 전체 episode를 추론하는
방법을 설명합니다.

## 사전 준비

먼저 다음 항목을 준비해야 합니다.

- Ubuntu 22.04 + Python 3.10 기준 환경
- 최상위 `requirements.txt`의 Python 의존성
- 공식 Robotics Transformer 및 Tensor2Robot 소스
- 공식 `rt1main` 체크포인트
- ONNX Universal Sentence Encoder Large `/5`
- 추론할 Fractal episode

관련 문서:

- [환경 설정](setup.md)
- [모델과 데이터 다운로드](downloads.md)
- [공식 TensorFlow 환경](../official/README.md)

공식 소스와 체크포인트는 다음 경로에 있어야 합니다.

```text
official/robotics_transformer/
official/tensor2robot/
official/robotics_transformer/trained_checkpoints/rt1main/
```

## 변환 구조

전체 추론은 ONNX USE와 세 개의 RT-1 정책 모델을 연결합니다.

| ONNX 모델 | 역할 |
| --- | --- |
| `model.onnx` (USE Large `/5`) | 자연어 지시문을 512차원 임베딩으로 변환 |
| `film_efficientnet.onnx` | 언어 조건이 적용된 이미지 특징 추출 |
| `token_learner.onnx` | 프레임당 8개 image token 생성 |
| `transformer.onnx` | 6프레임 sequence에서 action logits 예측 |

이미지 전처리, 6프레임 history, Transformer sequence, attention mask 및
action 복원은 NumPy/Python 코드가 담당합니다. 전체 tensor 흐름은
[`pipeline.md`](pipeline.md)를 참고하세요.

## ONNX 모델 변환

가상환경을 활성화한 뒤 저장소 최상위에서 다음 변환기를 순서대로 실행합니다.

```bash
python onnx/scripts/convert_film_efficientnet.py
python onnx/scripts/convert_token_learner.py
python onnx/scripts/convert_transformer.py
```

각 변환기는 ONNX export 전에 공식 `rt1main` 체크포인트의 해당 가중치를
복원합니다.

출력 경로:

```text
models/film_efficientnet/film_efficientnet.onnx
models/token_learner/token_learner.onnx
models/transformer/transformer.onnx
```

세 모델의 전체 크기는 약 182 MiB입니다. 생성된 ONNX 파일에는 공식
체크포인트에서 변환된 가중치가 포함되며, 용량과 원본 가중치의 배포 조건을
고려하여 Git에서 제외됩니다.

## 사용자 지시문으로 episode 추론

다운로드한 episode의 모든 프레임을 사용자가 지정한 자연어 지시문으로
추론합니다.

```bash
python onnx/validation/validate_episode.py \
  --episode-index 1 \
  --instruction "close middle drawer"
```

처리 순서:

1. episode의 RGB frame을 시간 순서대로 읽습니다.
2. 사용자 지시문을 로컬 ONNX USE Large `/5`로 인코딩합니다.
3. FiLM-EfficientNet, TokenLearner 및 Transformer를 ONNX Runtime에서
   실행합니다.
4. 각 frame의 11개 action token을 선택합니다.
5. token을 연속 robot action으로 복원합니다.
6. 전체 결과를 JSON으로 저장합니다.

결과 경로:

```text
validation_artifacts/episode_00001/episode/onnx.json
```

JSON에는 다음 항목이 포함됩니다.

- `episode_index`
- 실제 추론에 사용한 `instruction`
- `num_frames`
- frame별 `action_tokens`
- frame별 복원된 `actions`

## episode 메타데이터의 지시문 사용

`--instruction`을 생략하면 `metadata.json`에 저장된 episode 지시문을
사용합니다.

```bash
python onnx/validation/validate_episode.py --episode-index 1
```

공식 TensorFlow와 재현 가능한 비교를 수행할 때는 두 경로가 같은 지시문을
사용하도록 `--instruction`을 생략하는 방식을 권장합니다.

## 다른 입출력 경로 사용

추론 entry point는 필요에 따라 다음 경로 옵션을 지원합니다.

```text
--data-dir
--artifacts-dir
--use-model
```

지원하는 정확한 옵션과 기본값은 다음 명령으로 확인할 수 있습니다.

```bash
python onnx/validation/validate_episode.py --help
```

## 변환 정확성 검증

ONNX 파일이 생성되고 추론이 완료되었다는 사실만으로 공식 모델과의 동등성이
보장되지는 않습니다. 변환 또는 pipeline 코드를 변경했다면 모듈별 출력과
전체 episode action을 공식 TensorFlow 기준과 비교해야 합니다.

episode 1 전체 비교:

```bash
python official/validation/validate_episode.py
python onnx/validation/validate_episode.py
python comparison/end_to_end_episode.py
```

현재 기준 결과:

```text
Frames compared: 66
Token mismatch frames: []
Action mismatches: []
Maximum absolute action error: 2.384185791015625e-07
Match: True
```

모듈별 명령, tensor shape와 허용 오차는
[`../comparison/README.md`](../comparison/README.md)를 참고하세요.

## 라이선스와 생성 artifact

변환은 모델 표현과 실행 runtime을 변경하지만 원본 체크포인트 가중치의 출처를
변경하지 않습니다. ONNX 모델을 별도로 배포할 때는 공식 RT-1의 출처와
적용되는 조건을 함께 검토해야 합니다.

자세한 내용은
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)를 참고하세요.
