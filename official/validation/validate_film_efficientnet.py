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
"""Save the trained TensorFlow RT-1 FiLM-EfficientNet output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
sys.path.insert(0, str(OFFICIAL_DIR))

from film_efficientnet_checkpoint import restore_encoder  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the trained rt1main FiLM-EfficientNet encoder."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--model-dir", type=Path, default=DEFAULT_MODEL_DIR
  )
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
  model_dir = args.model_dir.expanduser().resolve()
  image_path = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "resize"
      / "tensorflow.npy"
  )
  context_path = (
      args.data_dir.expanduser().resolve()
      / episode_name
      / "language_embedding.npy"
  )
  if not (model_dir / "saved_model.pb").is_file():
    raise FileNotFoundError(f"rt1main SavedModel was not found: {model_dir}")
  if not image_path.is_file():
    raise FileNotFoundError(f"Resized TensorFlow image was not found: {image_path}")
  if not context_path.is_file():
    raise FileNotFoundError(f"Language embedding was not found: {context_path}")

  image = np.load(image_path, allow_pickle=False)
  context = np.load(context_path, allow_pickle=False)[np.newaxis, ...]
  if image.shape != (1, 300, 300, 3) or image.dtype != np.float32:
    raise ValueError(
        f"Expected image shape (1, 300, 300, 3) and float32; "
        f"received {image.shape} and {image.dtype}."
    )
  if context.shape != (1, 512) or context.dtype != np.float32:
    raise ValueError(
        f"Expected context shape (1, 512) and float32; "
        f"received {context.shape} and {context.dtype}."
    )

  image_tensor = tf.convert_to_tensor(image)
  context_tensor = tf.convert_to_tensor(context)
  encoder = restore_encoder(model_dir, image_tensor, context_tensor)
  output = encoder(
      image_tensor, context=context_tensor, training=False
  )
  output_array = np.asarray(output.numpy(), dtype=np.float32)

  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "film_efficientnet"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "tensorflow.npy"
  np.save(output_path, output_array)

  print(f"Model: {model_dir}")
  print(f"Image: {image_path}")
  print(f"Image shape: {image.shape}")
  print(f"Context: {context_path}")
  print(f"Context shape: {context.shape}")
  print(f"Output: {output_path}")
  print(f"Output shape: {output_array.shape}")
  print(f"Output dtype: {output_array.dtype}")
  print(f"Output range: [{float(output_array.min())}, "
        f"{float(output_array.max())}]")


if __name__ == "__main__":
  main()
