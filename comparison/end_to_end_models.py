"""Compare official TensorFlow and connected ONNX end-to-end outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compare RT-1 TensorFlow and ONNX end-to-end actions."
  )
  parser.add_argument("--episode-index", type=int, default=1)
  parser.add_argument(
      "--artifacts-dir",
      type=Path,
      default=REPOSITORY_DIR / "validation_artifacts",
  )
  parser.add_argument("--rtol", type=float, default=1e-5)
  parser.add_argument("--atol", type=float, default=1e-6)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  output_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
      / "end_to_end"
  )
  official_tokens = np.load(
      output_dir / "official_tokens.npy", allow_pickle=False
  )
  onnx_tokens = np.load(output_dir / "onnx_tokens.npy", allow_pickle=False)
  token_match = np.array_equal(official_tokens, onnx_tokens)
  print(f"Official TensorFlow tokens: {official_tokens.tolist()}")
  print(f"End-to-end ONNX tokens: {onnx_tokens.tolist()}")
  print(f"Token match: {token_match}")

  all_match = token_match
  with np.load(output_dir / "official.npz") as official_actions, \
       np.load(output_dir / "onnx.npz") as onnx_actions:
    if set(official_actions.files) != set(onnx_actions.files):
      raise AssertionError("Action keys do not match.")
    for key in official_actions.files:
      official_value = official_actions[key]
      onnx_value = onnx_actions[key]
      error = np.abs(official_value - onnx_value)
      match = official_value.shape == onnx_value.shape and np.allclose(
          official_value, onnx_value, rtol=args.rtol, atol=args.atol
      )
      print(f"{key}: max error={float(error.max())}, match={match}")
      all_match = all_match and match

  print(f"Match: {all_match}")
  if not all_match:
    raise AssertionError(
        "Official TensorFlow and end-to-end ONNX outputs do not match."
    )


if __name__ == "__main__":
  main()
