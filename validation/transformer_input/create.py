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
"""Create the RT-1 Transformer sequence and attention mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
TIME_STEPS = 6
IMAGE_TOKENS = 8
ACTION_TOKENS = 11
EMBEDDING_DIM = 512
TOKENS_PER_STEP = IMAGE_TOKENS + ACTION_TOKENS
SEQUENCE_LENGTH = TIME_STEPS * TOKENS_PER_STEP


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Assemble RT-1 image and zero action-token slots."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--source",
      choices=("tensorflow", "onnx"),
      default="tensorflow",
      help="Image-token history implementation to assemble.",
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def _action_index(position: int) -> int:
  if position % TOKENS_PER_STEP < IMAGE_TOKENS:
    return -1
  return position // TOKENS_PER_STEP


def _create_attention_mask() -> np.ndarray:
  mask = np.tril(
      np.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=np.float32)
  )
  for query in range(SEQUENCE_LENGTH):
    query_action = _action_index(query)
    if query_action == -1:
      continue
    for key in range(SEQUENCE_LENGTH):
      key_action = _action_index(key)
      if key_action == -1:
        continue
      if key_action < query_action:
        mask[query, key] = 0.0
      if key_action == query_action and key <= query:
        mask[query, key] = 0.0
  return mask


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")

  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
  )
  history_path = artifact_dir / "image_history" / f"{args.source}.npy"
  if not history_path.is_file():
    raise FileNotFoundError(f"Image-token history was not found: {history_path}")

  image_history = np.load(history_path, allow_pickle=False)
  expected_shape = (1, TIME_STEPS, IMAGE_TOKENS, EMBEDDING_DIM)
  if image_history.shape != expected_shape or image_history.dtype != np.float32:
    raise ValueError(
        f"Expected image history shape {expected_shape} and float32; "
        f"received {image_history.shape} and {image_history.dtype}."
    )

  action_slots = np.zeros(
      (1, TIME_STEPS, ACTION_TOKENS, EMBEDDING_DIM), dtype=np.float32
  )
  sequence = np.concatenate([image_history, action_slots], axis=2)
  sequence = np.ascontiguousarray(
      sequence.reshape(1, SEQUENCE_LENGTH, EMBEDDING_DIM)
  )
  attention_mask = _create_attention_mask()

  output_dir = artifact_dir / "transformer_input"
  output_dir.mkdir(parents=True, exist_ok=True)
  sequence_name = (
      "sequence.npy" if args.source == "tensorflow" else "onnx_sequence.npy"
  )
  sequence_path = output_dir / sequence_name
  mask_path = output_dir / "attention_mask.npy"
  np.save(sequence_path, sequence)
  np.save(mask_path, attention_mask)

  print(f"Image history: {history_path}")
  print(f"Image history shape: {image_history.shape}")
  print(f"Sequence: {sequence_path}")
  print(f"Sequence shape: {sequence.shape}")
  print(f"Attention mask: {mask_path}")
  print(f"Attention mask shape: {attention_mask.shape}")
  print(f"Tokens per timestep: {IMAGE_TOKENS} image + {ACTION_TOKENS} action")


if __name__ == "__main__":
  main()
