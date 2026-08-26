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
"""Compare an ONNX USE Large /5 embedding with an RT-1 dataset embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import get_library_path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Embed an RT-1 episode instruction with ONNX USE Large /5 and "
          "compare it with the dataset embedding."
      )
  )
  parser.add_argument(
      "--episode-index",
      type=int,
      default=1,
      help="Zero-based downloaded episode index. Defaults to 1.",
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
      help="Parent directory containing episode_XXXXX directories.",
  )
  parser.add_argument(
      "--onnx-model",
      type=Path,
      default=(
          REPOSITORY_DIR
          / "models"
          / "universal_sentence_encoder_large_onnx"
          / "5"
          / "model.onnx"
      ),
      help="ONNX USE Large /5 model path.",
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
      help="Root directory for generated validation outputs.",
  )
  parser.add_argument("--rtol", type=float, default=1e-5)
  parser.add_argument("--atol", type=float, default=1e-6)
  return parser.parse_args()


def _create_onnx_session(model_path: Path) -> ort.InferenceSession:
  options = ort.SessionOptions()
  options.register_custom_ops_library(get_library_path())
  return ort.InferenceSession(
      str(model_path),
      sess_options=options,
      providers=["CPUExecutionProvider"],
  )


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
  left_flat = left.reshape(-1).astype(np.float64)
  right_flat = right.reshape(-1).astype(np.float64)
  denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
  if denominator == 0.0:
    raise ValueError("Cosine similarity is undefined for a zero vector.")
  return float(np.dot(left_flat, right_flat) / denominator)


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")
  if args.rtol < 0.0 or args.atol < 0.0:
    raise ValueError("--rtol and --atol must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  metadata_path = episode_dir / "metadata.json"
  reference_path = episode_dir / "language_embedding.npy"
  model_path = args.onnx_model.expanduser().resolve()

  if not metadata_path.is_file():
    raise FileNotFoundError(f"Episode metadata was not found: {metadata_path}")
  if not reference_path.is_file():
    raise FileNotFoundError(
        f"Dataset language embedding was not found: {reference_path}"
    )
  if not model_path.is_file():
    raise FileNotFoundError(f"ONNX USE model was not found: {model_path}")

  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  instruction = metadata["instruction"]
  reference = np.load(reference_path, allow_pickle=False)

  session = _create_onnx_session(model_path)
  generated = session.run(["outputs"], {"inputs": [instruction]})[0][0]

  print(f"Episode: {episode_name}")
  print(f"Instruction: {instruction}")
  print(f"ONNX model: {model_path}")
  print(f"Dataset reference: {reference_path}")
  print(f"ONNX shape: {generated.shape}")
  print(f"Dataset shape: {reference.shape}")
  print(f"ONNX dtype: {generated.dtype}")
  print(f"Dataset dtype: {reference.dtype}")
  print(f"ONNX first 5: {generated[:5].tolist()}")
  print(f"Dataset first 5: {reference[:5].tolist()}")

  if generated.shape != reference.shape:
    raise AssertionError(
        f"Shape mismatch: ONNX {generated.shape} != dataset {reference.shape}"
    )
  if generated.dtype != reference.dtype:
    raise AssertionError(
        f"Dtype mismatch: ONNX {generated.dtype} != dataset {reference.dtype}"
    )

  absolute_error = np.abs(generated - reference)
  cosine_similarity = _cosine_similarity(generated, reference)
  match = np.allclose(
      generated,
      reference,
      rtol=args.rtol,
      atol=args.atol,
  )

  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "use_embedding_onnx"
  )
  artifact_dir.mkdir(parents=True, exist_ok=True)
  np.save(artifact_dir / "onnx.npy", generated)
  np.save(artifact_dir / "dataset.npy", reference)

  print(f"Max absolute error: {float(absolute_error.max())}")
  print(f"Mean absolute error: {float(absolute_error.mean())}")
  print(f"Cosine similarity: {cosine_similarity}")
  print(f"rtol: {args.rtol}")
  print(f"atol: {args.atol}")
  print(f"Match: {match}")
  print(f"Exact match: {np.array_equal(generated, reference)}")
  print(f"Artifacts: {artifact_dir}")

  if not match:
    raise AssertionError(
        "The ONNX USE embedding does not match the dataset reference."
    )


if __name__ == "__main__":
  main()
