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
"""Save the ONNX RT-1 FiLM-EfficientNet output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
DEFAULT_MODEL_PATH = (
    REPOSITORY_DIR
    / "models"
    / "film_efficientnet"
    / "film_efficientnet.onnx"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the converted RT-1 FiLM-EfficientNet ONNX model."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  model_path = args.model.expanduser().resolve()
  image_path = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "resize"
      / "onnx.npy"
  )
  context_path = (
      args.data_dir.expanduser().resolve()
      / episode_name
      / "language_embedding.npy"
  )
  if not model_path.is_file():
    raise FileNotFoundError(f"ONNX model was not found: {model_path}")
  if not image_path.is_file():
    raise FileNotFoundError(f"Resized ONNX image was not found: {image_path}")
  if not context_path.is_file():
    raise FileNotFoundError(f"Language embedding was not found: {context_path}")

  image = np.load(image_path, allow_pickle=False)
  context = np.load(context_path, allow_pickle=False)[np.newaxis, ...]
  if image.shape != (1, 300, 300, 3) or image.dtype != np.float32:
    raise ValueError(f"Unexpected image shape or dtype: {image.shape}, {image.dtype}")
  if context.shape != (1, 512) or context.dtype != np.float32:
    raise ValueError(
        f"Unexpected context shape or dtype: {context.shape}, {context.dtype}"
    )

  session = ort.InferenceSession(
      str(model_path), providers=["CPUExecutionProvider"]
  )
  input_names = {model_input.name for model_input in session.get_inputs()}
  if input_names != {"image", "context"}:
    raise ValueError(f"Unexpected ONNX inputs: {sorted(input_names)}")
  model_outputs = session.get_outputs()
  if len(model_outputs) != 1:
    raise ValueError(f"Expected one ONNX output; found {len(model_outputs)}")
  output = session.run(None, {"image": image, "context": context})[0]
  output = np.asarray(output, dtype=np.float32)

  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "film_efficientnet"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "onnx.npy"
  np.save(output_path, output)

  print(f"Model: {model_path}")
  print(f"Image shape: {image.shape}")
  print(f"Context shape: {context.shape}")
  print(f"Output: {output_path}")
  print(f"Output shape: {output.shape}")
  print(f"Output dtype: {output.dtype}")
  print(f"Output range: [{float(output.min())}, {float(output.max())}]")


if __name__ == "__main__":
  main()
