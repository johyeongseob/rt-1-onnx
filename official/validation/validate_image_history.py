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
"""Create a six-frame RT-1 image-token history with TensorFlow."""

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
from token_learner_checkpoint import restore_token_learner  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)
HISTORY_LENGTH = 6


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Create six RT-1 image-token timesteps with TensorFlow."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
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
  if args.episode_index < 0 or args.start_frame < 0:
    raise ValueError("Episode and frame indices must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  model_dir = args.model_dir.expanduser().resolve()
  frame_paths = [
      episode_dir / "frames" / f"frame_{index:04d}.png"
      for index in range(args.start_frame, args.start_frame + HISTORY_LENGTH)
  ]
  missing = [path for path in frame_paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"History frames were not found: {missing}")

  context_path = episode_dir / "language_embedding.npy"
  if not context_path.is_file():
    raise FileNotFoundError(f"Language embedding was not found: {context_path}")

  context = np.load(context_path, allow_pickle=False)
  if context.shape != (512,) or context.dtype != np.float32:
    raise ValueError(
        f"Expected context shape (512,) and float32; "
        f"received {context.shape} and {context.dtype}."
    )
  context_tensor = tf.convert_to_tensor(context)[tf.newaxis, :]

  first_image = tf.io.decode_png(
      tf.io.read_file(str(frame_paths[0])), channels=3
  )
  first_image = tf.image.convert_image_dtype(first_image, tf.float32)
  first_image = first_image[tf.newaxis, ...]
  encoder = restore_encoder(model_dir, first_image, context_tensor)
  first_features = encoder(
      first_image, context=context_tensor, training=False
  )
  learner = restore_token_learner(model_dir, first_features)

  frame_tokens = []
  for frame_path in frame_paths:
    image = tf.io.decode_png(tf.io.read_file(str(frame_path)), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[tf.newaxis, ...]
    features = encoder(image, context=context_tensor, training=False)
    frame_tokens.append(learner(features, training=False))

  history = tf.stack(frame_tokens, axis=1)
  history_array = np.asarray(history.numpy(), dtype=np.float32)

  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "image_history"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "tensorflow.npy"
  np.save(output_path, history_array)

  print(f"Frames: {frame_paths[0]} through {frame_paths[-1]}")
  print(f"Instruction embedding: {context_path}")
  print(f"Output: {output_path}")
  print(f"Output shape: {history_array.shape}")
  print(f"Output dtype: {history_array.dtype}")
  print(f"Output range: [{float(history_array.min())}, "
        f"{float(history_array.max())}]")


if __name__ == "__main__":
  main()
