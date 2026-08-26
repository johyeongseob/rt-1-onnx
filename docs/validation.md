# TensorFlow-ONNX 출력 동등성 검증

이 문서는 공식 TensorFlow RT-1과 ONNX RT-1의 출력을 어떤 범위에서
검증했는지 설명합니다. 변환 결과는 입력 준비부터 최종 robot action까지
단계별로 비교합니다.

## 검증 범위

- **이미지 전처리**
  - dtype 변환, crop, 300 x 300 리사이즈
- **Language**
  - ONNX USE Large `/5` 출력과 공식 dataset의 512차원 언어 임베딩
- **Vision**
  - FiLM-EfficientNet 출력, TokenLearner 출력, 6프레임 image history
- **Transformer**
  - 입력 sequence, causal attention mask, logits
- **Action**
  - 11개 action token, 디코딩된 연속 action
- **End-to-end 동등성**
  - 실제 Fractal episode 전체 프레임의 action token 및 연속 action

## End-to-end 검증 결과

`close middle drawer` 지시문을 포함한 Fractal episode 1의 전체 66프레임을
공식 TensorFlow RT-1과 ONNX-only 추론 파이프라인으로 각각 실행했습니다.

```text
Frames compared: 66
Token mismatch frames: []
Action mismatches: []
Maximum absolute action error: 2.384185791015625e-07
Match: True
```

이 결과는 자연어 지시문부터 최종 action까지 실행한 ONNX 파이프라인이 기록된
RT-1 episode 전체에서 공식 기준과 사실상 동일한 action을 출력했음을
의미합니다. 실제 로봇의 안전성이나 작업 성공률을 검증한 결과는 아닙니다.

## 실행 방법

모듈별 검증 명령, 예상 tensor shape, 허용 오차와 artifact 경로는
[`../comparison/README.md`](../comparison/README.md)를 참고하세요.
