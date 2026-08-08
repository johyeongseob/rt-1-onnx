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
"""Create a six-frame RT-1 image-token history with ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
sys.path.insert(0, str(ONNX_DIR / "film_efficientnet"))

from preprossors import convert_dtype_and_crop_images, resize_images  # pylint: disable=g-import-not-at-top


DEFAULT_FILM_MODEL = (
    REPOSITORY_DIR
    / "models"
    / "film_efficientnet"
    / "film_efficientnet.onnx"
)
DEFAULT_TOKEN_MODEL = (
    REPOSITORY_DIR / "models" / "token_learner" / "token_learner.onnx"
)
HISTORY_LENGTH = 6


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Create six RT-1 image-token timesteps with ONNX Runtime."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--film-model", type=Path, default=DEFAULT_FILM_MODEL)
  parser.add_argument("--token-model", type=Path, default=DEFAULT_TOKEN_MODEL)
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
  )
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.episode_index < 0 or args.start_frame < 0:
    raise ValueError("Episode and frame indices must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  frame_paths = [
      episode_dir / "frames" / f"frame_{index:04d}.png"
      for index in range(args.start_frame, args.start_frame + HISTORY_LENGTH)
  ]
  missing = [path for path in frame_paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"History frames were not found: {missing}")

  film_model = args.film_model.expanduser().resolve()
  token_model = args.token_model.expanduser().resolve()
  for model_path in (film_model, token_model):
    if not model_path.is_file():
      raise FileNotFoundError(f"ONNX model was not found: {model_path}")

  context_path = episode_dir / "language_embedding.npy"
  context = np.load(context_path, allow_pickle=False)
  if context.shape != (512,) or context.dtype != np.float32:
    raise ValueError(
        f"Expected context shape (512,) and float32; "
        f"received {context.shape} and {context.dtype}."
    )

  images = np.stack([
      np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
      for path in frame_paths
  ])
  images = convert_dtype_and_crop_images(images)
  images = resize_images(images, (300, 300))
  context_batch = np.repeat(context[np.newaxis, :], HISTORY_LENGTH, axis=0)

  film_session = ort.InferenceSession(
      str(film_model), providers=["CPUExecutionProvider"]
  )
  token_session = ort.InferenceSession(
      str(token_model), providers=["CPUExecutionProvider"]
  )
  features = film_session.run(
      None, {"image": images, "context": context_batch}
  )[0]
  tokens = token_session.run(None, {"features": features})[0]
  history = np.asarray(
      tokens.reshape(1, HISTORY_LENGTH, 8, 512), dtype=np.float32
  )

  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / episode_name
      / "image_history"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "onnx.npy"
  np.save(output_path, history)

  print(f"Frames: {frame_paths[0]} through {frame_paths[-1]}")
  print(f"Instruction embedding: {context_path}")
  print(f"Output: {output_path}")
  print(f"Output shape: {history.shape}")
  print(f"Output dtype: {history.dtype}")
  print(f"Output range: [{float(history.min())}, {float(history.max())}]")


if __name__ == "__main__":
  main()
