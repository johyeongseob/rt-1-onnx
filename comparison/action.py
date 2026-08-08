"""Compare TensorFlow and NumPy-decoded RT-1 actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Compare RT-1 decoded actions.")
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--artifacts-dir", type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  parser.add_argument("--rtol", type=float, default=1e-5)
  parser.add_argument("--atol", type=float, default=1e-6)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  action_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
      / "action"
  )
  tf_tokens = np.load(action_dir / "tensorflow_tokens.npy", allow_pickle=False)
  onnx_tokens = np.load(action_dir / "onnx_tokens.npy", allow_pickle=False)
  token_match = np.array_equal(tf_tokens, onnx_tokens)
  print(f"TensorFlow tokens: {tf_tokens.tolist()}")
  print(f"ONNX tokens: {onnx_tokens.tolist()}")
  print(f"Token match: {token_match}")

  all_match = token_match
  with np.load(action_dir / "tensorflow.npz") as tensorflow_actions, \
       np.load(action_dir / "onnx.npz") as onnx_actions:
    if set(tensorflow_actions.files) != set(onnx_actions.files):
      raise AssertionError("Decoded action keys do not match.")
    for key in tensorflow_actions.files:
      tf_value = tensorflow_actions[key]
      onnx_value = onnx_actions[key]
      match = np.allclose(
          tf_value, onnx_value, rtol=args.rtol, atol=args.atol
      )
      max_error = float(np.max(np.abs(tf_value - onnx_value)))
      print(f"{key}: max error={max_error}, match={match}")
      all_match = all_match and match

  print(f"Match: {all_match}")
  if not all_match:
    raise AssertionError("The decoded RT-1 actions do not match.")


if __name__ == "__main__":
  main()
