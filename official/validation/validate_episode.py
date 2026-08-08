"""Run every frame of an episode through the official RT-1 policy."""

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

from validate_end_to_end import ACTION_ORDER, _observation, _time_step


del tfp, tf_agents
OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR / "robotics_transformer/trained_checkpoints/rt1main"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run a complete episode through official rt1main."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument(
      "--data-dir", type=Path, default=REPOSITORY_DIR / "data/fractal_samples"
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  metadata = json.loads((episode_dir / "metadata.json").read_text("utf-8"))
  instruction = metadata["instruction"]
  embedding = np.load(
      episode_dir / "language_embedding.npy", allow_pickle=False
  ).astype(np.float32, copy=False).reshape(512)
  frame_paths = sorted((episode_dir / "frames").glob("frame_*.png"))
  if not frame_paths:
    raise FileNotFoundError(f"No episode frames were found in {episode_dir}")

  logging.set_verbosity(logging.ERROR)
  policy = tf.saved_model.load(str(args.model_dir.expanduser().resolve()))
  policy_state = policy.get_initial_state(tf.constant(1, tf.int32))
  steps = []
  for index, frame_path in enumerate(frame_paths):
    image = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
    time_step = _time_step(
        _observation(image, embedding, instruction), first=(index == 0)
    )
    policy_step = policy.action(time_step, policy_state)
    policy_state = policy_step.state
    state_tokens = np.asarray(policy_state["action_tokens"].numpy())
    token_index = min(index, 5)
    tokens = state_tokens[0, token_index, :, 0, 0]
    steps.append({
        "frame_index": index,
        "action_tokens": tokens.tolist(),
        "actions": {
            key: np.asarray(policy_step.action[key].numpy())[0].tolist()
            for key in ACTION_ORDER
        },
    })

  result = {
      "episode_index": args.episode_index,
      "instruction": instruction,
      "num_frames": len(frame_paths),
      "steps": steps,
  }
  output_dir = (
      args.artifacts_dir.expanduser().resolve() / episode_name / "episode"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "official.json"
  output_path.write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8"
  )
  print(f"Episode: {episode_name}")
  print(f"Instruction: {instruction}")
  print(f"Frames: {len(frame_paths)}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
