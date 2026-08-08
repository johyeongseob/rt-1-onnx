# Copyright 2026 rt-1-lab contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare TensorFlow and ONNX RT-1 Transformer logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compare RT-1 Transformer validation arrays."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  parser.add_argument("--rtol", type=float, default=1e-4)
  parser.add_argument("--atol", type=float, default=1e-4)
  return parser.parse_args()


def _load(path: Path) -> np.ndarray:
  if not path.is_file():
    raise FileNotFoundError(f"Validation output was not found: {path}")
  return np.load(path, allow_pickle=False)


def main() -> None:
  args = _parse_args()
  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
      / "transformer"
  )
  tensorflow_path = artifact_dir / "tensorflow.npy"
  onnx_path = artifact_dir / "onnx.npy"
  tensorflow_output = _load(tensorflow_path)
  onnx_output = _load(onnx_path)

  print(f"TensorFlow: {tensorflow_path}")
  print(f"  shape: {tensorflow_output.shape}")
  print(f"  dtype: {tensorflow_output.dtype}")
  print(f"ONNX: {onnx_path}")
  print(f"  shape: {onnx_output.shape}")
  print(f"  dtype: {onnx_output.dtype}")

  if tensorflow_output.shape != onnx_output.shape:
    raise AssertionError("Transformer output shapes do not match.")
  if tensorflow_output.dtype != onnx_output.dtype:
    raise AssertionError("Transformer output dtypes do not match.")

  absolute_error = np.abs(tensorflow_output - onnx_output)
  match = np.allclose(
      tensorflow_output, onnx_output, rtol=args.rtol, atol=args.atol
  )
  print(f"Max absolute error: {float(absolute_error.max())}")
  print(f"Mean absolute error: {float(absolute_error.mean())}")
  print(f"rtol: {args.rtol}")
  print(f"atol: {args.atol}")
  print(f"Match: {match}")

  if not match:
    raise AssertionError("The Transformer outputs do not match.")


if __name__ == "__main__":
  main()
