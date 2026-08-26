# RT-1 ONNX

Google의 공식 TensorFlow RT-1 정책 네트워크를 모듈식 ONNX 모델로
변환하고, 실제 RT-1 로봇 에피소드에서 두 구현이 동일한 행동을
출력하는지 단계별·end-to-end 방식으로 검증하는 오픈소스 프로젝트입니다.

기존 RT-1은 오래된 TensorFlow 및 Tensor2Robot 환경에 의존합니다. 이
프로젝트는 핵심 정책 네트워크를 개방형 ONNX 형식으로 이전하여 다양한
하드웨어와 실행 환경에서 RT-1 추론을 쉽게 재현하고 확장할 수 있도록 하는
것을 목표로 합니다.

> [!IMPORTANT]
> 현재 자연어 인코더인 Universal Sentence Encoder Large `/5`는 TensorFlow
> SavedModel로 실행합니다. FiLM-EfficientNet, TokenLearner, Transformer로
> 구성된 RT-1 정책 네트워크는 ONNX Runtime으로 실행합니다. 따라서 현재
> 전체 파이프라인은 TensorFlow와 ONNX Runtime을 함께 사용하는 하이브리드
> 구조입니다.

## 핵심 검증 결과

`close middle drawer` 지시문을 포함한 Fractal episode 1의 전체 66프레임을
공식 TensorFlow RT-1과 변환된 ONNX 파이프라인으로 각각 추론했습니다.

```text
비교한 프레임: 66
action token 불일치 프레임: 0
연속 action 불일치: 0
최대 절대 action 오차: 2.384185791015625e-07
결과: 일치
```

이 결과는 기록된 실제 RT-1 관측 데이터에 대해 ONNX 정책 네트워크가 공식
TensorFlow 기준 구현과 사실상 동일한 행동을 출력했음을 의미합니다. 실제
로봇에서의 물리적 안전성이나 작업 성공률을 인증하는 결과는 아닙니다.

## 추론 파이프라인

```text
자연어 지시문
      |
      v
Universal Sentence Encoder Large /5
512차원 언어 임베딩
      |
      +-------------------------------+
                                      |
카메라 프레임                         |
      |                               |
      v                               v
이미지 변환 및 300 x 300 리사이즈 + 언어 임베딩
                  |
                  v
          FiLM-EfficientNet
           [B, 9, 9, 512]
                  |
                  v
             TokenLearner
             [B, 8, 512]
                  |
                  v
       최근 6프레임 이미지 history
           [B, 6, 8, 512]
                  |
                  v
Transformer: 6 x (이미지 토큰 8개 + action slot 11개)
             [B, 114, 256]
                  |
                  v
       11개 action token 및 복원
                  |
                  v
arm xyz/rpy/gripper + base xy/yaw + mode switch
```

ONNX 구현은 다음 세 모델로 분리되어 있습니다.

- `film_efficientnet.onnx`
- `token_learner.onnx`
- `transformer.onnx`
- NumPy/Python 기반 전처리, 파이프라인 연결, history 관리 및 action 복원

공식 `rt1main` 가중치는 각 ONNX 파일 내부에 포함됩니다. 세 ONNX 모델의
전체 크기는 약 182 MiB이며 Universal Sentence Encoder는 별도로 저장됩니다.

## Action 구성

Transformer는 각 시점에서 11개의 action token을 출력합니다.

- arm 이동: `x`, `y`, `z`
- arm 회전: `roll`, `pitch`, `yaw`
- gripper: 열림 및 닫힘
- mobile base 이동: `x`, `y`
- mobile base 회전: `yaw`
- mode: arm 제어, base 제어, episode 종료

## 검증 환경

다음 환경에서 변환과 동등성 검증을 완료했습니다.

- Windows + WSL2
- Ubuntu 22.04
- Python 3.10
- TensorFlow 2.14.0
- TensorFlow Probability 0.22.1
- TF-Agents 0.18.0
- TensorFlow Datasets 4.9.4
- TensorFlow Hub 0.15.0
- ONNX Runtime 1.18.1
- tf2onnx 1.17.0
- MuJoCo 3.11.0

소스 코드, 데이터셋, 체크포인트, ONNX 모델 및 검증 출력은 Windows
저장소에 두고 Python 가상환경은 WSL 내부의
`~/venvs/rt1-tensorflow`에 저장합니다.

WSL 설정, 공식 저장소 준비, Tensor2Robot protobuf 생성, TensorFlow 2.x
호환성 수정 및 공식 tokenizer 테스트 방법은
[`official/README.md`](official/README.md)를 참고하세요.

## 설치

가상환경을 활성화하고 저장소로 이동합니다.

```bash
source ~/venvs/rt1-tensorflow/bin/activate
cd /mnt/c/Users/(your_machine_name)/Desktop/rt-1-onnx
```

TensorFlow, ONNX 및 시각화에 필요한 전체 의존성을 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

공식 소스와 체크포인트는 다음 경로에 있어야 합니다.

```text
official/robotics_transformer/
official/tensor2robot/
official/robotics_transformer/trained_checkpoints/rt1main/
```

두 공식 저장소의 `requirements.txt`는 원본을 유지합니다. 이 프로젝트를
실행할 때는 버전이 고정된 최상위 [`requirements.txt`](requirements.txt)를
사용합니다.

## 언어 인코더 다운로드

RT-1은 Universal Sentence Encoder Large `/5`를 사용하여 자연어 지시문을
512차원 임베딩으로 변환합니다. 다음 명령으로 SavedModel을 다운로드합니다.

```bash
python scripts/download_use_model.py
```

출력 경로:

```text
models/universal_sentence_encoder_large/5/
```

다운로드한 모델 디렉터리는 Git에서 제외됩니다.

## Fractal episode 다운로드

다운로더는 전체 데이터셋을 받지 않고 공개 RT-1 Fractal 데이터셋에서 필요한
episode만 Google Cloud Storage를 통해 읽습니다.

```text
gs://gresearch/robotics/fractal20220817_data/0.1.0
```

0부터 시작하는 정확한 episode 인덱스 하나를 다운로드합니다.

```bash
python scripts/download_sample_episode.py --episode-index 1
```

시작과 끝을 포함하는 범위를 다운로드합니다.

```bash
python scripts/download_sample_episode.py --start-index 2 --end-index 10
```

인덱스를 생략하면 설정된 검색 범위 안에서 성공으로 표시된 첫 episode를
찾습니다.

```bash
python scripts/download_sample_episode.py
```

각 episode는 별도 디렉터리에 저장됩니다.

```text
data/fractal_samples/episode_00001/
|-- frames/frame_0000.png
|-- language_embedding.npy
|-- metadata.json
`-- steps.json
```

이미 존재하는 episode 디렉터리는 덮어쓰지 않고 건너뜁니다. 다운로드한
데이터는 Git에서 제외됩니다.

## 공식 체크포인트를 ONNX로 변환

저장소 최상위에서 다음 변환기를 순서대로 실행합니다.

```bash
python onnx/scripts/convert_film_efficientnet.py
python onnx/scripts/convert_token_learner.py
python onnx/scripts/convert_transformer.py
```

출력 경로:

```text
models/film_efficientnet/film_efficientnet.onnx
models/token_learner/token_learner.onnx
models/transformer/transformer.onnx
```

각 변환기는 ONNX export 전에 공식 `rt1main` 가중치를 복원합니다. 생성된
모델은 용량과 원본 가중치의 배포 조건을 고려하여 Git에서 제외됩니다.

## 사용자 지시문으로 ONNX 추론

다운로드한 episode의 모든 프레임을 사용자가 지정한 자연어 지시문으로
추론합니다.

```bash
python onnx/validation/validate_episode.py \
  --episode-index 1 \
  --instruction "close middle drawer"
```

지시문은 로컬 USE Large `/5` SavedModel로 인코딩합니다. 이후
FiLM-EfficientNet, TokenLearner 및 Transformer는 ONNX Runtime에서
실행됩니다.

결과 JSON:

```text
validation_artifacts/episode_00001/episode/onnx.json
```

JSON에는 episode 인덱스, 사용자 지시문, 전체 프레임 수, 프레임별 action
token과 복원된 연속 action이 저장됩니다.

## TensorFlow-ONNX 동등성 검증

변환 결과는 전처리부터 최종 action까지 단계별로 검증합니다.

- 이미지 dtype 변환
- crop
- 300 x 300 리사이즈
- USE-Large/5 언어 임베딩
- FiLM-EfficientNet 출력
- TokenLearner 출력
- 6프레임 이미지 history
- Transformer 입력
- causal attention mask
- Transformer logits
- action token
- 연속 action

전체 검증 명령, 예상 tensor shape, 허용 오차 및 end-to-end 결과는
[`comparison/README.md`](comparison/README.md)를 참고하세요.

## MuJoCo 시각화

MuJoCo 시각화는 RT-1이 출력한 end-effector action을 이해하기 위한
도구입니다. 원본 Everyday Robots 로봇의 관절 구조, 실제 trajectory 또는
CAD 모델을 복원한 것이 아닙니다.

MuJoCo 설치와 WSLg GUI 지원을 확인합니다.

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m mujoco.viewer
```

### 실시간 position viewer

```bash
python visualization/mujoco/visualize_world_vector.py
```

viewer는 정규화된 RT-1 `world_vector` delta를 누적하여 표시합니다.

### 카메라 GIF

소스 프레임 하나당 표시 프레임 네 개를 사용하여 12 FPS GIF를 생성합니다.

```bash
python visualization/export_episode_frames_gif.py
```

출력:

```text
visualization_artifacts/episode_00001/camera_frames.gif
```

### MuJoCo gripper GIF

카메라 프레임과 동기화된 12 FPS MuJoCo GIF를 생성합니다.

```bash
python visualization/mujoco/export_world_vector_gif.py
```

단순화한 V자형 gripper는 다음 일곱 arm action 값을 표현합니다.

- 누적 `x`, `y`, `z`
- 누적 `roll`, `pitch`, `yaw`
- gripper 열림 및 닫힘

episode 1의 66개 base action 출력은 모두 사실상 정지 상태를 나타내는 중앙
bin 값이므로 base action 시각화는 생략했습니다.

출력:

```text
visualization_artifacts/episode_00001/world_vector.gif
```

### 카메라와 MuJoCo 결과 결합

동기화된 카메라 GIF와 gripper GIF를 좌우로 결합합니다.

```bash
python visualization/combine_camera_vector_gifs.py
```

출력:

```text
visualization_artifacts/episode_00001/camera_and_world_vector.gif
```

## 저장소 구조

```text
rt-1-onnx/
|-- official/                # 공식 TensorFlow 코드와 검증기
|-- onnx/                    # ONNX 변환기, 추론 파이프라인 및 검증기
|-- scripts/                 # 모델 및 episode 다운로드 도구
|-- comparison/              # TensorFlow-ONNX 출력 비교 도구
|-- visualization/           # 카메라 및 MuJoCo 시각화 도구
|-- data/                    # 다운로드한 episode 샘플 (Git 제외)
|-- models/                  # USE 및 생성한 ONNX 모델 (Git 제외)
|-- validation_artifacts/    # 중간 tensor와 JSON 결과 (Git 제외)
|-- visualization_artifacts/ # 생성한 GIF (Git 제외)
`-- requirements.txt         # 버전이 고정된 전체 실행환경
```

## 프로젝트 범위와 안전 고지

- 이 프로젝트는 연구, 변환 재현 및 추론 결과 검증을 목적으로 합니다.
- 공식 RT-1 구현과의 출력 동등성이 실제 로봇의 안전성을 보장하지 않습니다.
- 물리적 로봇에서 사용하기 전에는 별도의 안전 제어, workspace 제한,
  충돌 방지, 비상 정지 및 작업별 검증이 필요합니다.
- MuJoCo 결과는 action을 설명하기 위한 도식적 시각화이며 실제 로봇 동작의
  물리 시뮬레이션이 아닙니다.
- 이 저장소는 Google의 공식 제품 또는 공식 ONNX 포팅이 아닙니다.

## 출처

- [RT-1 논문](https://arxiv.org/abs/2212.06817)
- [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)
- [google-research/tensor2robot](https://github.com/google-research/tensor2robot)
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/5)
- [Fractal RT-1 데이터셋](https://www.tensorflow.org/datasets/catalog/fractal20220817_data)
- [ONNX](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MuJoCo](https://mujoco.org/)

## 라이선스

이 프로젝트에서 자체적으로 작성한 코드는 Apache License 2.0으로
배포됩니다. 자세한 내용은 [LICENSE.md](LICENSE.md)를 확인하세요.

외부 코드, 모델, 체크포인트 및 데이터셋에는 각각의 원 배포 조건이
적용됩니다. 자세한 출처와 고지사항은
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 확인하세요.

`official/` 아래의 Google 원본 코드에는 각 디렉터리에 포함된 기존
라이선스와 저작권 고지가 적용됩니다.
