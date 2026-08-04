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
"""Save the TensorFlow RT-1 image resize output for one frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
IMAGE_SIZE = 300


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the TensorFlow resize used before RT-1 EfficientNet-B3."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--frame-index", type=int, default=0)
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0 or args.frame_index < 0:
    raise ValueError("Episode and frame indices must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  frame_path = (
      args.data_dir
      / episode_name
      / "frames"
      / f"frame_{args.frame_index:04d}.png"
  )
  if not frame_path.is_file():
    raise FileNotFoundError(f"Input frame was not found: {frame_path}")

  image = tf.io.decode_png(tf.io.read_file(str(frame_path)), channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image_batch = image[tf.newaxis, ...]
  output = tf.image.resize(image_batch, (IMAGE_SIZE, IMAGE_SIZE))

  artifact_dir = args.output_dir / episode_name / "resize"
  artifact_dir.mkdir(parents=True, exist_ok=True)
  output_path = artifact_dir / "tensorflow.npy"
  np.save(output_path, output.numpy())

  print(f"Input: {frame_path}")
  print(f"Input shape: {tuple(image_batch.shape)}")
  print(f"Output: {output_path}")
  print(f"Output shape: {tuple(output.shape)}")
  print(f"Output dtype: {output.dtype.name}")
  print(f"Output range: [{float(tf.reduce_min(output))}, "
        f"{float(tf.reduce_max(output))}]")


if __name__ == "__main__":
  main()
