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
"""Save the trained TensorFlow RT-1 TokenLearner output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
sys.path.insert(0, str(OFFICIAL_DIR))

from token_learner_checkpoint import restore_token_learner  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the trained rt1main TokenLearner."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
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
  artifacts_dir = args.artifacts_dir.expanduser().resolve()
  input_path = artifacts_dir / episode_name / "film_efficientnet" / "tensorflow.npy"
  if not (model_dir / "saved_model.pb").is_file():
    raise FileNotFoundError(f"rt1main SavedModel was not found: {model_dir}")
  if not input_path.is_file():
    raise FileNotFoundError(f"FiLM-EfficientNet output was not found: {input_path}")

  features = np.load(input_path, allow_pickle=False)
  if features.shape != (1, 9, 9, 512) or features.dtype != np.float32:
    raise ValueError(
        f"Expected features shape (1, 9, 9, 512) and float32; "
        f"received {features.shape} and {features.dtype}."
    )

  features_tensor = tf.convert_to_tensor(features)
  learner = restore_token_learner(model_dir, features_tensor)
  output = learner(features_tensor, training=False)
  output_array = np.asarray(output.numpy(), dtype=np.float32)

  output_dir = artifacts_dir / episode_name / "token_learner"
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "tensorflow.npy"
  np.save(output_path, output_array)

  print(f"Model: {model_dir}")
  print(f"Input: {input_path}")
  print(f"Input shape: {features.shape}")
  print(f"Output: {output_path}")
  print(f"Output shape: {output_array.shape}")
  print(f"Output dtype: {output_array.dtype}")
  print(f"Output range: [{float(output_array.min())}, "
        f"{float(output_array.max())}]")


if __name__ == "__main__":
  main()
