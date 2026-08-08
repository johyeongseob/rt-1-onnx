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
"""Save the ONNX RT-1 Transformer logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
DEFAULT_MODEL_PATH = (
    REPOSITORY_DIR / "models" / "transformer" / "transformer.onnx"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run the converted RT-1 Transformer ONNX model."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  artifact_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
  )
  input_dir = artifact_dir / "transformer_input"
  sequence_path = input_dir / "onnx_sequence.npy"
  mask_path = input_dir / "attention_mask.npy"
  sequence = np.load(sequence_path, allow_pickle=False)
  attention_mask = np.load(mask_path, allow_pickle=False)

  model_path = args.model.expanduser().resolve()
  if not model_path.is_file():
    raise FileNotFoundError(f"ONNX model was not found: {model_path}")
  session = ort.InferenceSession(
      str(model_path), providers=["CPUExecutionProvider"]
  )
  input_names = {model_input.name for model_input in session.get_inputs()}
  if input_names != {"sequence", "attention_mask"}:
    raise ValueError(f"Unexpected ONNX inputs: {sorted(input_names)}")
  logits = session.run(
      None,
      {"sequence": sequence, "attention_mask": attention_mask},
  )[0]
  logits = np.asarray(logits, dtype=np.float32)

  output_dir = artifact_dir / "transformer"
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "onnx.npy"
  np.save(output_path, logits)

  print(f"Model: {model_path}")
  print(f"Sequence shape: {sequence.shape}")
  print(f"Attention mask shape: {attention_mask.shape}")
  print(f"Output: {output_path}")
  print(f"Output shape: {logits.shape}")
  print(f"Output dtype: {logits.dtype}")
  print(f"Output range: [{float(logits.min())}, {float(logits.max())}]")


if __name__ == "__main__":
  main()
