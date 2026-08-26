# MuJoCo 시각화

MuJoCo 시각화는 RT-1이 출력한 end-effector action을 이해하기 위한
도구입니다. 원본 Everyday Robots 로봇의 관절 구조, 실제 trajectory 또는
CAD 모델을 복원한 것이 아닙니다.

모든 명령은 WSL 가상환경을 활성화한 뒤 저장소 최상위 디렉터리에서
실행합니다.

## 실행환경 확인

MuJoCo 설치와 WSLg GUI 지원을 확인합니다.

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m mujoco.viewer
```

## 실시간 position viewer

```bash
python visualization/mujoco/visualize_world_vector.py
```

viewer는 정규화된 RT-1 `world_vector` delta를 누적하여 표시합니다.

## 카메라 GIF

소스 프레임 하나당 표시 프레임 네 개를 사용하여 12 FPS GIF를 생성합니다.

```bash
python visualization/export_episode_frames_gif.py
```

출력:

```text
visualization_artifacts/episode_00001/camera_frames.gif
```

## MuJoCo gripper GIF

카메라 프레임과 동기화된 12 FPS MuJoCo GIF를 생성합니다.

```bash
python visualization/mujoco/export_world_vector_gif.py
```

단순화한 gripper와 robot arm은 다음 일곱 arm action 값을 표현합니다.

- 누적 `x`, `y`, `z`
- 누적 `roll`, `pitch`, `yaw`
- gripper 열림 및 닫힘

episode 1의 66개 base action 출력은 모두 사실상 정지 상태를 나타내는 중앙
bin 값이므로 base action 시각화는 생략했습니다.

출력:

```text
visualization_artifacts/episode_00001/world_vector.gif
```

## 카메라와 MuJoCo 결과 결합

동기화된 카메라 GIF와 MuJoCo GIF를 좌우로 결합합니다. 기본 캡션은 왼쪽
`RT-1 Camera`, 오른쪽 `ONNX RT-1 Action (MuJoCo)`입니다.

```bash
python visualization/combine_camera_vector_gifs.py
```

출력:

```text
visualization_artifacts/episode_00001/camera_and_world_vector.gif
```

사용 가능한 경로 및 캡션 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python visualization/combine_camera_vector_gifs.py --help
```

## 시각화 범위

이 시각화는 RT-1의 end-effector action을 설명하기 위한 도식적 표현입니다.
실제 로봇의 링크 길이, 관절값, inverse kinematics, 충돌 또는 물리적 상호작용을
재현하지 않으며 로봇 동작의 안전성이나 작업 성공 여부를 검증하지 않습니다.
