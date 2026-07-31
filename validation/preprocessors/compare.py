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
"""Compare saved TensorFlow and ONNX RT-1 preprocessing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compare saved RT-1 preprocessing validation arrays."
  )
  parser.add_argument(
      "--episode-index",
      type=int,
      default=1,
      help="Zero-based episode index. Defaults to 1.",
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
      help="Root directory containing generated validation outputs.",
  )
  parser.add_argument(
      "--rtol",
      type=float,
      default=0.0,
      help="Relative tolerance passed to numpy.allclose. Defaults to 0.",
  )
  parser.add_argument(
      "--atol",
      type=float,
      default=0.0,
      help="Absolute tolerance passed to numpy.allclose. Defaults to 0.",
  )
  return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
  if not path.is_file():
    raise FileNotFoundError(f"Validation output was not found: {path}")
  return np.load(path, allow_pickle=False)


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")
  if args.rtol < 0.0 or args.atol < 0.0:
    raise ValueError("--rtol and --atol must be zero or greater.")

  artifact_dir = (
      args.artifacts_dir
      / f"episode_{args.episode_index:05d}"
      / "preprocessors"
  )
  tensorflow_path = artifact_dir / "tensorflow.npy"
  onnx_path = artifact_dir / "onnx.npy"

  tensorflow_output = _load_array(tensorflow_path)
  onnx_output = _load_array(onnx_path)

  print(f"TensorFlow: {tensorflow_path}")
  print(f"  shape: {tensorflow_output.shape}")
  print(f"  dtype: {tensorflow_output.dtype}")
  print(f"ONNX: {onnx_path}")
  print(f"  shape: {onnx_output.shape}")
  print(f"  dtype: {onnx_output.dtype}")

  if tensorflow_output.shape != onnx_output.shape:
    raise AssertionError(
        "Shape mismatch: "
        f"TensorFlow {tensorflow_output.shape} != ONNX {onnx_output.shape}"
    )
  if tensorflow_output.dtype != onnx_output.dtype:
    raise AssertionError(
        "Dtype mismatch: "
        f"TensorFlow {tensorflow_output.dtype} != ONNX {onnx_output.dtype}"
    )

  absolute_error = np.abs(tensorflow_output - onnx_output)
  max_absolute_error = float(absolute_error.max())
  mean_absolute_error = float(absolute_error.mean())
  matches = np.allclose(
      tensorflow_output,
      onnx_output,
      rtol=args.rtol,
      atol=args.atol,
  )

  print(f"Max absolute error: {max_absolute_error}")
  print(f"Mean absolute error: {mean_absolute_error}")
  print(f"rtol: {args.rtol}")
  print(f"atol: {args.atol}")
  print(f"Match: {matches}")

  if not matches:
    raise AssertionError("The preprocessing outputs do not match.")


if __name__ == "__main__":
  main()
