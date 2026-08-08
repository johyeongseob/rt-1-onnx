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
"""Save the trained TensorFlow RT-1 Transformer logits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
sys.path.insert(0, str(OFFICIAL_DIR))

from transformer_checkpoint import restore_transformer  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the trained rt1main Transformer."
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

  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
  )
  input_dir = artifact_dir / "transformer_input"
  sequence_path = input_dir / "sequence.npy"
  mask_path = input_dir / "attention_mask.npy"
  sequence = np.load(sequence_path, allow_pickle=False)
  attention_mask = np.load(mask_path, allow_pickle=False)
  if sequence.shape != (1, 114, 512) or sequence.dtype != np.float32:
    raise ValueError(
        f"Unexpected sequence shape or dtype: {sequence.shape}, {sequence.dtype}"
    )
  if attention_mask.shape != (114, 114) or attention_mask.dtype != np.float32:
    raise ValueError(
        "Unexpected attention mask shape or dtype: "
        f"{attention_mask.shape}, {attention_mask.dtype}"
    )

  sequence_tensor = tf.convert_to_tensor(sequence)
  mask_tensor = tf.convert_to_tensor(attention_mask)
  decoder = restore_transformer(
      args.model_dir.expanduser().resolve(), sequence_tensor, mask_tensor
  )
  logits, _ = decoder(
      sequence_tensor, training=False, attention_mask=mask_tensor
  )
  logits_array = np.asarray(logits.numpy(), dtype=np.float32)

  output_dir = artifact_dir / "transformer"
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "tensorflow.npy"
  np.save(output_path, logits_array)

  print(f"Sequence: {sequence_path}")
  print(f"Attention mask: {mask_path}")
  print(f"Output: {output_path}")
  print(f"Output shape: {logits_array.shape}")
  print(f"Output dtype: {logits_array.dtype}")
  print(f"Output range: [{float(logits_array.min())}, "
        f"{float(logits_array.max())}]")


if __name__ == "__main__":
  main()
