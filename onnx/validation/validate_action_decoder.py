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
"""Decode RT-1 ONNX Transformer logits into robot actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
sys.path.insert(0, str(ONNX_DIR))

from action_decoder import decode_action_tokens, extract_action_tokens  # pylint: disable=g-import-not-at-top


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Decode the ONNX RT-1 action output."
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
  logits_path = artifact_dir / "transformer" / "onnx.npy"
  logits = np.load(logits_path, allow_pickle=False)
  tokens = extract_action_tokens(logits)
  actions = decode_action_tokens(tokens)

  output_dir = artifact_dir / "action"
  output_dir.mkdir(parents=True, exist_ok=True)
  token_path = output_dir / "onnx_tokens.npy"
  action_path = output_dir / "onnx.npz"
  np.save(token_path, tokens)
  np.savez(action_path, **actions)

  print(f"Logits: {logits_path}")
  print(f"Action tokens: {token_path}")
  print(f"Tokens: {tokens.tolist()}")
  print(f"Actions: {action_path}")
  for key, value in actions.items():
    print(f"  {key}: {value.tolist()}")


if __name__ == "__main__":
  main()
