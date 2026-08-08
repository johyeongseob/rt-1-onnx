"""Run the connected RT-1 ONNX modules end to end."""

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

from rt1_pipeline import RT1ONNXPipeline  # pylint: disable=g-import-not-at-top


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run RT-1 end to end through the connected ONNX modules."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument(
      "--film-model",
      type=Path,
      default=REPOSITORY_DIR / "models/film_efficientnet/film_efficientnet.onnx",
  )
  parser.add_argument(
      "--token-learner-model",
      type=Path,
      default=REPOSITORY_DIR / "models/token_learner/token_learner.onnx",
  )
  parser.add_argument(
      "--transformer-model",
      type=Path,
      default=REPOSITORY_DIR / "models/transformer/transformer.onnx",
  )
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
  frame_paths = [
      episode_dir / "frames" / f"frame_{index:04d}.png"
      for index in range(args.start_frame, args.start_frame + 6)
  ]
  missing = [path for path in frame_paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"History frames were not found: {missing}")

  images = np.stack([
      np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
      for path in frame_paths
  ])[np.newaxis, ...]
  embedding = np.load(
      episode_dir / "language_embedding.npy", allow_pickle=False
  ).astype(np.float32, copy=False)[np.newaxis, ...]
  metadata = json.loads((episode_dir / "metadata.json").read_text("utf-8"))

  pipeline = RT1ONNXPipeline(
      args.film_model, args.token_learner_model, args.transformer_model
  )
  tokens, actions = pipeline.predict(images, embedding)

  output_dir = (
      args.artifacts_dir.expanduser().resolve() / episode_name / "end_to_end"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  token_path = output_dir / "onnx_tokens.npy"
  action_path = output_dir / "onnx.npz"
  np.save(token_path, tokens)
  np.savez(action_path, **actions)

  print(f"Frames: {frame_paths[0]} through {frame_paths[-1]}")
  print(f"Instruction: {metadata['instruction']}")
  print(f"Action tokens: {token_path}")
  print(f"Tokens: {tokens.tolist()}")
  print(f"Actions: {action_path}")
  for key, value in actions.items():
    print(f"  {key}: {value.tolist()}")


if __name__ == "__main__":
  main()
