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
"""Save the official TensorFlow RT-1 preprocessing output for one frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
sys.path.insert(0, str(OFFICIAL_DIR))

from robotics_transformer.film_efficientnet import preprocessors  # pylint: disable=g-import-not-at-top


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the official RT-1 inference image preprocessing."
  )
  parser.add_argument(
      "--episode-index",
      type=int,
      default=1,
      help="Zero-based downloaded episode index. Defaults to 1.",
  )
  parser.add_argument(
      "--frame-index",
      type=int,
      default=0,
      help="Zero-based frame index within the episode. Defaults to 0.",
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
      help="Parent directory containing episode_XXXXX directories.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
      help="Root directory for generated validation outputs.",
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
  image_batch = image[tf.newaxis, ...]
  output = preprocessors.convert_dtype_and_crop_images(
      image_batch,
      training=False,
      pad_then_crop=True,
      convert_dtype=True,
  )

  artifact_dir = args.output_dir / episode_name / "preprocessors"
  artifact_dir.mkdir(parents=True, exist_ok=True)
  output_path = artifact_dir / "tensorflow.npy"
  np.save(output_path, output.numpy())

  print(f"Input: {frame_path}")
  print(f"Input shape: {tuple(image_batch.shape)}")
  print(f"Input dtype: {image_batch.dtype.name}")
  print(f"Output: {output_path}")
  print(f"Output shape: {tuple(output.shape)}")
  print(f"Output dtype: {output.dtype.name}")
  print(
      "Output range: "
      f"[{float(tf.reduce_min(output))}, {float(tf.reduce_max(output))}]"
  )


if __name__ == "__main__":
  main()
