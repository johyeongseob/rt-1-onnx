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
"""Compare a USE Large /5 embedding with an RT-1 dataset embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow_hub as hub


OFFICIAL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = OFFICIAL_DIR.parent
RTOL = 1e-5
ATOL = 1e-6


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Embed an RT-1 episode instruction with Universal Sentence "
          "Encoder Large /5 and compare it with the dataset embedding."
      )
  )
  parser.add_argument(
      "--episode-index",
      type=int,
      default=1,
      help="Zero-based downloaded episode index. Defaults to 1.",
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
      help="Parent directory containing episode_XXXXX directories.",
  )
  parser.add_argument(
      "--model-dir",
      type=Path,
      default=(
          REPOSITORY_DIR
          / "models"
          / "universal_sentence_encoder_large"
          / "5"
      ),
      help="Local TensorFlow Hub directory containing USE Large /5.",
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
  if args.episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")

  episode_name = f"episode_{args.episode_index:05d}"
  episode_dir = args.data_dir.expanduser().resolve() / episode_name
  metadata_path = episode_dir / "metadata.json"
  reference_path = episode_dir / "language_embedding.npy"
  model_dir = args.model_dir.expanduser().resolve()

  if not metadata_path.is_file():
    raise FileNotFoundError(f"Episode metadata was not found: {metadata_path}")
  if not reference_path.is_file():
    raise FileNotFoundError(
        f"Dataset language embedding was not found: {reference_path}"
    )
  if not model_dir.is_dir():
    raise FileNotFoundError(f"Local USE model was not found: {model_dir}")

  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  instruction = metadata["instruction"]
  reference = np.load(reference_path, allow_pickle=False)

  model = hub.load(str(model_dir))
  generated = model([instruction]).numpy()[0]

  if generated.shape != reference.shape:
    raise ValueError(
        "Embedding shape mismatch: "
        f"generated={generated.shape}, reference={reference.shape}"
    )

  absolute_error = np.abs(generated - reference)
  match = np.allclose(generated, reference, rtol=RTOL, atol=ATOL)

  artifact_dir = (
      args.output_dir.expanduser().resolve() / episode_name / "use_embedding"
  )
  artifact_dir.mkdir(parents=True, exist_ok=True)
  output_path = artifact_dir / "use_large_v5.npy"
  np.save(output_path, generated)

  print(f"Instruction: {instruction}")
  print(f"Model: {model_dir}")
  print(f"Reference: {reference_path}")
  print(f"Generated output: {output_path}")
  print(f"Generated shape: {generated.shape}")
  print(f"Reference shape: {reference.shape}")
  print(f"Generated dtype: {generated.dtype}")
  print(f"Reference dtype: {reference.dtype}")
  print(f"Max absolute error: {float(absolute_error.max())}")
  print(f"Mean absolute error: {float(absolute_error.mean())}")
  print(f"rtol: {RTOL}")
  print(f"atol: {ATOL}")
  print(f"Match: {match}")
  print(f"Exact match: {np.array_equal(generated, reference)}")


if __name__ == "__main__":
  main()
