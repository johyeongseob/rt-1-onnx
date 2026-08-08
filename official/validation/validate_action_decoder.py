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
"""Decode RT-1 TensorFlow Transformer logits into robot actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
sys.path.insert(0, str(OFFICIAL_DIR))

from robotics_transformer.tokenizers import action_tokenizer  # pylint: disable=g-import-not-at-top
from tensor2robot.utils import tensorspec_utils  # pylint: disable=g-import-not-at-top


ACTION_ORDER = [
    "terminate_episode",
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
    "base_displacement_vector",
    "base_displacement_vertical_rotation",
]


def _create_tokenizer() -> action_tokenizer.RT1ActionTokenizer:
  spec = tensorspec_utils.TensorSpecStruct()
  spec.base_displacement_vector = tensor_spec.BoundedTensorSpec(
      (2,), tf.float32, -1.0, 1.0, "base_displacement_vector"
  )
  spec.base_displacement_vertical_rotation = tensor_spec.BoundedTensorSpec(
      (1,), tf.float32, -np.pi, np.pi,
      "base_displacement_vertical_rotation",
  )
  spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
      (1,), tf.float32, -1.0, 1.0, "gripper_closedness_action"
  )
  spec.rotation_delta = tensor_spec.BoundedTensorSpec(
      (3,), tf.float32, -np.pi / 2.0, np.pi / 2.0, "rotation_delta"
  )
  spec.terminate_episode = tensor_spec.BoundedTensorSpec(
      (3,), tf.int32, 0, 1, "terminate_episode"
  )
  spec.world_vector = tensor_spec.BoundedTensorSpec(
      (3,), tf.float32, -1.0, 1.0, "world_vector"
  )
  return action_tokenizer.RT1ActionTokenizer(
      spec, vocab_size=256, action_order=ACTION_ORDER
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Decode the TensorFlow RT-1 action output."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--artifacts-dir", type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
  )
  logits_path = artifact_dir / "transformer" / "tensorflow.npy"
  logits = np.load(logits_path, allow_pickle=False)
  if logits.shape != (1, 114, 256):
    raise ValueError(f"Unexpected Transformer logits shape: {logits.shape}")

  action_logits = logits[:, 102:113, :]
  tokens = tf.argmax(action_logits, axis=-1, output_type=tf.int32)
  actions = _create_tokenizer().detokenize(tokens)
  arrays = {key: np.asarray(value.numpy()) for key, value in actions.items()}

  output_dir = artifact_dir / "action"
  output_dir.mkdir(parents=True, exist_ok=True)
  token_path = output_dir / "tensorflow_tokens.npy"
  action_path = output_dir / "tensorflow.npz"
  np.save(token_path, tokens.numpy())
  np.savez(action_path, **arrays)

  print(f"Logits: {logits_path}")
  print(f"Action tokens: {token_path}")
  print(f"Tokens: {tokens.numpy().tolist()}")
  print(f"Actions: {action_path}")
  for key in ACTION_ORDER:
    print(f"  {key}: {arrays[key].tolist()}")


if __name__ == "__main__":
  main()
