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
"""Run six frames through the exported RT-1 TensorFlow policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents
from tf_agents.trajectories import time_step as ts


del tfp, tf_agents  # Imports register SavedModel TypeSpecs.

OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR / "robotics_transformer" / "trained_checkpoints" / "rt1main"
)
ACTION_ORDER = [
    "terminate_episode",
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
    "base_displacement_vector",
    "base_displacement_vertical_rotation",
]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the official rt1main policy end to end for six frames."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--history-length", type=int, default=6)
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument(
      "--data-dir", type=Path, default=REPOSITORY_DIR / "data"
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def _observation(
    image: np.ndarray, embedding: np.ndarray, instruction: str
) -> dict[str, tf.Tensor]:
  """Build the complete observation expected by the exported policy."""
  return {
      "base_pose_tool_reached": tf.zeros((1, 7), tf.float32),
      "gripper_closed": tf.zeros((1, 1), tf.float32),
      "gripper_closedness_commanded": tf.zeros((1, 1), tf.float32),
      "height_to_bottom": tf.zeros((1, 1), tf.float32),
      "image": tf.convert_to_tensor(image[None, ...], tf.uint8),
      "natural_language_embedding": tf.convert_to_tensor(
          embedding[None, ...], tf.float32
      ),
      "natural_language_instruction": tf.constant([instruction]),
      "orientation_box": tf.zeros((1, 2, 3), tf.float32),
      "orientation_start": tf.zeros((1, 4), tf.float32),
      "robot_orientation_positions_box": tf.zeros((1, 3, 3), tf.float32),
      "rotation_delta_to_go": tf.zeros((1, 3), tf.float32),
      "src_rotation": tf.zeros((1, 4), tf.float32),
      "vector_to_go": tf.zeros((1, 3), tf.float32),
      "workspace_bounds": tf.zeros((1, 3, 3), tf.float32),
  }


def _time_step(observation: dict[str, tf.Tensor], first: bool) -> ts.TimeStep:
  step_type = ts.StepType.FIRST if first else ts.StepType.MID
  return ts.TimeStep(
      step_type=tf.fill((1,), tf.cast(step_type, tf.int32)),
      reward=tf.zeros((1,), tf.float32),
      discount=tf.ones((1,), tf.float32),
      observation=observation,
  )


def main() -> None:
  args = _parse_args()
  if args.history_length != 6:
    raise ValueError("rt1main was exported with a six-frame history.")

  model_dir = args.model_dir.expanduser().resolve()
  episode_dir = (
      args.data_dir.expanduser().resolve()
      / "fractal_samples"
      / f"episode_{args.episode_index:05d}"
  )
  metadata = json.loads((episode_dir / "metadata.json").read_text("utf-8"))
  instruction = metadata["instruction"]
  embedding = np.load(
      episode_dir / "language_embedding.npy", allow_pickle=False
  ).astype(np.float32)
  embedding = embedding.reshape(512)

  logging.set_verbosity(logging.ERROR)
  policy = tf.saved_model.load(str(model_dir))
  policy_state = policy.get_initial_state(tf.constant(1, tf.int32))

  action_history: dict[str, list[np.ndarray]] = {
      key: [] for key in ACTION_ORDER
  }
  frame_paths = []
  policy_step = None
  for offset in range(args.history_length):
    frame_index = args.start_frame + offset
    frame_path = episode_dir / "frames" / f"frame_{frame_index:04d}.png"
    if not frame_path.is_file():
      raise FileNotFoundError(f"Frame was not found: {frame_path}")
    image = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
    if image.shape != (256, 320, 3):
      raise ValueError(f"Unexpected frame shape {image.shape}: {frame_path}")

    time_step = _time_step(
        _observation(image, embedding, instruction), first=(offset == 0)
    )
    policy_step = policy.action(time_step, policy_state)
    policy_state = policy_step.state
    frame_paths.append(frame_path)
    for key in ACTION_ORDER:
      action_history[key].append(np.asarray(policy_step.action[key].numpy()))

  if policy_step is None:
    raise RuntimeError("No policy step was produced.")

  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
      / "end_to_end"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  action_path = output_dir / "official.npz"
  history_path = output_dir / "official_history.npz"
  token_path = output_dir / "official_tokens.npy"

  final_actions = {
      key: np.asarray(policy_step.action[key].numpy()) for key in ACTION_ORDER
  }
  histories = {
      key: np.concatenate(values, axis=0)
      for key, values in action_history.items()
  }
  action_tokens = np.asarray(policy_state["action_tokens"].numpy())
  final_tokens = action_tokens[:, -1, :, 0, 0]

  np.savez(action_path, **final_actions)
  np.savez(history_path, **histories)
  np.save(token_path, final_tokens)

  print(f"Model: {model_dir}")
  print(f"Frames: {frame_paths[0]} through {frame_paths[-1]}")
  print(f"Instruction: {instruction}")
  print(f"Final action tokens: {final_tokens.tolist()}")
  print(f"Final actions: {action_path}")
  for key in ACTION_ORDER:
    print(f"  {key}: {final_actions[key].tolist()}")
  print(f"Action history: {history_path}")


if __name__ == "__main__":
  main()
