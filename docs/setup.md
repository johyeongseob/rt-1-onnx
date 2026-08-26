# RT-1 ONNX 환경 설정

이 문서는 RT-1 ONNX의 기준 검증 환경과 WSL2 기반 설치 방법을 설명합니다.

## 기준 검증 환경

다음 환경에서 공식 TensorFlow RT-1의 실행, ONNX 변환, 모듈별 수치 비교 및
전체 episode 동등성 검증을 완료했습니다.

| 구성요소 | 검증 버전 |
| --- | --- |
| Host OS | Windows + WSL2 |
| Linux distribution | Ubuntu 22.04 |
| Python | 3.10 |
| TensorFlow | 2.14.0 |
| TensorFlow Probability | 0.22.1 |
| TF-Agents | 0.18.0 |
| TensorFlow Datasets | 4.9.4 |
| ONNX Runtime | 1.18.1 |
| ONNX Runtime Extensions | 0.15.2 |
| tf2onnx | 1.17.0 |
| MuJoCo | 3.11.0 |

전체 Python 의존성은 버전이 고정된 최상위
[`requirements.txt`](../requirements.txt)에 정의되어 있습니다.

## 권장 파일 배치

프로젝트 소스와 대용량 artifact는 Windows 파일시스템에 두고, Linux Python
가상환경은 WSL 내부에 두는 구성을 권장합니다.

```text
WSL Linux filesystem
`-- ~/venvs/rt1-tensorflow/        # Python 3.10 virtual environment

Windows filesystem mounted in WSL
`-- /mnt/c/Users/<username>/Desktop/rt-1-onnx/
    |-- official/
    |-- onnx/
    |-- scripts/
    |-- models/
    |-- data/
    `-- requirements.txt
```

가상환경을 `/mnt/c` 아래에 만들면 많은 작은 Python 패키지 파일을 Windows
파일시스템에서 읽게 되어 성능과 호환성이 저하될 수 있습니다.

## 1. Ubuntu 22.04 실행

Windows PowerShell에서 설치된 WSL distribution을 확인합니다.

```powershell
wsl --list --verbose
```

Ubuntu 22.04를 지정하여 실행합니다.

```powershell
wsl -d Ubuntu-22.04
```

Ubuntu 내부에서 버전을 확인합니다.

```bash
lsb_release -a
```

> `wsl` 명령은 Windows PowerShell에서 실행합니다. Ubuntu shell 내부에서
> `wsl`을 다시 실행하거나 `sudo apt install wsl`을 수행할 필요가 없습니다.

## 2. 필수 시스템 패키지

Ubuntu 22.04에서 Python 가상환경과 기본 빌드 도구를 준비합니다.

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev build-essential
```

Tensor2Robot protobuf 생성에 필요한 추가 환경과 공식 소스 준비 과정은
[`../official/README.md`](../official/README.md)를 참고하세요.

## 3. Python 가상환경 생성

WSL의 Linux 홈 디렉터리에 가상환경을 생성합니다.

```bash
mkdir -p ~/venvs
python3.10 -m venv ~/venvs/rt1-tensorflow
source ~/venvs/rt1-tensorflow/bin/activate
```

정상적으로 활성화되면 shell prompt 앞에 일반적으로
`(rt1-tensorflow)`이 표시됩니다.

Python 경로와 버전을 확인합니다.

```bash
which python
python --version
```

예상 결과:

```text
/home/<username>/venvs/rt1-tensorflow/bin/python
Python 3.10.x
```

이미 가상환경이 존재한다면 새로 만들지 않고 다음 명령으로 활성화합니다.

```bash
source ~/venvs/rt1-tensorflow/bin/activate
```

## 4. 프로젝트 디렉터리로 이동

Windows 사용자 이름에 맞춰 경로를 지정합니다.

```bash
cd /mnt/c/Users/<username>/Desktop/rt-1-onnx
```

현재 위치를 확인합니다.

```bash
pwd
```

예상 형식:

```text
/mnt/c/Users/<username>/Desktop/rt-1-onnx
```

## 5. Python 의존성 설치

가상환경이 활성화된 상태에서 실행합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

최상위 `requirements.txt`는 TensorFlow, ONNX 변환·추론 및 MuJoCo
시각화 환경을 함께 설치합니다. `official/` 아래 두 upstream 저장소의
`requirements.txt`는 원본 보존용이며 프로젝트 설치 기준으로 사용하지
않습니다.

## 6. 주요 버전 확인

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import onnxruntime as ort; print(ort.__version__)"
python -c "import onnxruntime_extensions as ortx; print(ortx.__version__)"
python -c "import tf2onnx; print(tf2onnx.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

핵심 예상 결과:

```text
TensorFlow:    2.14.0
ONNX Runtime:  1.18.1
ORT Extensions: 0.15.2
tf2onnx:       1.17.0
MuJoCo:        3.11.0
```

## 7. 공식 소스와 체크포인트 확인

다음 경로가 준비되어 있어야 합니다.

```text
official/robotics_transformer/
official/tensor2robot/
official/robotics_transformer/trained_checkpoints/rt1main/
```

공식 소스 준비, Tensor2Robot protobuf 생성, TensorFlow 2.x 호환성 수정 및
공식 tokenizer 테스트는 [`../official/README.md`](../official/README.md)를
따릅니다.

## 8. 언어 모델과 episode 준비

Universal Sentence Encoder Large `/5`와 검증용 Fractal episode 준비 방법은
[`downloads.md`](downloads.md)를 참고하세요.

## 9. 설치 확인용 ONNX 추론

ONNX USE와 세 정책 ONNX 모델을 연결하고 episode 전체 추론으로 설치 상태를
확인하는 방법은
[`onnx_conversion.md`](onnx_conversion.md)를 참고하세요.

TensorFlow-ONNX 전체 동등성 검증 방법은
[`../comparison/README.md`](../comparison/README.md)를 참고하세요.

## 가상환경 종료

```bash
deactivate
```

## 다른 환경 사용

Ubuntu 24.04 또는 Windows native 환경에서도 일부 기능을 실행할 수 있지만,
공식 RT-1 변환과 동등성 검증의 기준 환경은 Ubuntu 22.04와 Python 3.10입니다.
다른 환경에서 얻은 결과는 package version, execution provider 및 수치 오차를
별도로 기록하는 것을 권장합니다.
