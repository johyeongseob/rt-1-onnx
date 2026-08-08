"""Run every frame of an episode through the connected ONNX pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
sys.path.insert(0, str(ONNX_DIR))

from rt1_pipeline import RT1ONNXPipeline


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run a complete episode through the ONNX RT-1 pipeline."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--instruction",
      type=str,
      default=None,
      help=(
          "Natural-language instruction to run. If omitted, use the "
          "instruction stored in the episode metadata."
      ),
  )
  parser.add_argument(
      "--data-dir", type=Path, default=REPOSITORY_DIR / "data/fractal_samples"
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  parser.add_argument(
      "--use-model",
      type=Path,
      default=(
          REPOSITORY_DIR / "models/universal_sentence_encoder_large/5"
      ),
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  metadata = json.loads((episode_dir / "metadata.json").read_text("utf-8"))
  frame_paths = sorted((episode_dir / "frames").glob("frame_*.png"))
  if not frame_paths:
    raise FileNotFoundError(f"No episode frames were found in {episode_dir}")
  images = np.stack([
      np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
      for path in frame_paths
  ])[np.newaxis]
  instruction = (
      args.instruction
      if args.instruction is not None
      else metadata["instruction"]
  )

  pipeline = RT1ONNXPipeline(
      REPOSITORY_DIR / "models/film_efficientnet/film_efficientnet.onnx",
      REPOSITORY_DIR / "models/token_learner/token_learner.onnx",
      REPOSITORY_DIR / "models/transformer/transformer.onnx",
      args.use_model,
  )
  tokens, actions = pipeline.predict_episode_instruction(images, instruction)
  steps = []
  for index in range(len(frame_paths)):
    steps.append({
        "frame_index": index,
        "action_tokens": tokens[0, index].tolist(),
        "actions": {
            key: value[0, index].tolist() for key, value in actions.items()
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
  output_path = output_dir / "onnx.json"
  output_path.write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8"
  )
  print(f"Episode: {episode_name}")
  print(f"Instruction: {instruction}")
  print(f"Frames: {len(frame_paths)}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
