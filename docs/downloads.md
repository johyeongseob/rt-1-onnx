# 모델과 데이터 다운로드

이 문서는 RT-1 ONNX 추론과 검증에 필요한 Universal Sentence Encoder와
Fractal episode를 전체 원본 데이터셋 없이 선택적으로 준비하는 방법을
설명합니다.

명령은 가상환경을 활성화한 뒤 저장소 최상위에서 실행합니다. 환경 준비는
[`setup.md`](setup.md)를 참고하세요.

## Universal Sentence Encoder Large /5

RT-1은 Universal Sentence Encoder Large `/5`를 사용하여 자연어 지시문을
512차원 언어 임베딩으로 변환합니다.

다운로드 스크립트:

```bash
python scripts/download_use_model.py
```

출력 경로:

```text
models/universal_sentence_encoder_large/5/
|-- saved_model.pb
`-- variables/
```

ONNX episode 추론 시 `RT1ONNXPipeline`은 기본적으로 이 경로의 SavedModel을
TensorFlow Hub로 로드합니다.

다운로드한 모델은 Git에서 제외됩니다. 모델 자체에는 TensorFlow Hub의 원
배포자가 정한 라이선스와 이용 조건이 적용됩니다. 자세한 고지는
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)를 참고하세요.

## Fractal RT-1 episode

검증에는 공개 RT-1 Fractal 데이터셋인 `fractal20220817_data`를 사용합니다.

- Version: `0.1.0`
- TensorFlow Datasets catalog:
  <https://www.tensorflow.org/datasets/catalog/fractal20220817_data>
- Google Cloud Storage source:

```text
gs://gresearch/robotics/fractal20220817_data/0.1.0
```

전체 데이터셋은 약 111 GiB이므로 이 프로젝트의 다운로더는 요청한 episode만
읽어 로컬 샘플 형식으로 저장합니다.

### 하나의 episode 다운로드

episode 인덱스는 0부터 시작합니다.

```bash
python scripts/download_sample_episode.py --episode-index 1
```

위 명령은 `episode_00001`을 생성합니다.

### episode 범위 다운로드

시작과 끝 인덱스를 모두 포함하는 범위를 다운로드합니다.

```bash
python scripts/download_sample_episode.py --start-index 2 --end-index 10
```

### 성공한 episode 자동 검색

인덱스를 생략하면 설정된 검색 범위 안에서 성공으로 표시된 첫 episode를
찾아 저장합니다.

```bash
python scripts/download_sample_episode.py
```

### 출력 구조

각 episode는 별도 디렉터리에 저장됩니다.

```text
data/fractal_samples/episode_00001/
|-- frames/
|   |-- frame_0000.png
|   `-- ...
|-- language_embedding.npy
|-- metadata.json
`-- steps.json
```

각 파일의 역할:

| 파일 | 내용 |
| --- | --- |
| `frames/frame_*.png` | 시간 순서의 RGB 카메라 관측 |
| `language_embedding.npy` | episode에 저장된 512차원 언어 임베딩 |
| `metadata.json` | episode 인덱스, 지시문 및 메타데이터 |
| `steps.json` | 각 timestep의 기록 정보와 action |

이미 존재하는 episode 디렉터리는 덮어쓰지 않고 건너뜁니다. 다른 결과가
필요하면 기존 디렉터리를 직접 확인한 뒤 명시적으로 처리하세요.

다운로드한 데이터는 Git에서 제외되며 원 데이터셋의 이용 조건을 따릅니다.

## 준비 상태 확인

ONNX episode 1 추론에 필요한 최소 경로는 다음과 같습니다.

```text
models/universal_sentence_encoder_large/5/
data/fractal_samples/episode_00001/frames/
data/fractal_samples/episode_00001/metadata.json
```

공식 체크포인트와 ONNX 모델 준비는
[`onnx_conversion.md`](onnx_conversion.md)를 참고하세요.
