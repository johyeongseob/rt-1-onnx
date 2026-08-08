"""Compare official end-to-end RT-1 with the modular TensorFlow path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compare official and modular TensorFlow RT-1 actions."
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
  episode_dir = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
  )
  official_dir = episode_dir / "end_to_end"
  modular_dir = episode_dir / "action"

  official_tokens = np.load(
      official_dir / "official_tokens.npy", allow_pickle=False
  )
  modular_tokens = np.load(
      modular_dir / "tensorflow_tokens.npy", allow_pickle=False
  )
  token_match = np.array_equal(official_tokens, modular_tokens)
  print(f"Official tokens: {official_tokens.tolist()}")
  print(f"Modular TensorFlow tokens: {modular_tokens.tolist()}")
  print(f"Token match: {token_match}")

  all_match = token_match
  with np.load(official_dir / "official.npz") as official_actions, \
       np.load(modular_dir / "tensorflow.npz") as modular_actions:
    if set(official_actions.files) != set(modular_actions.files):
      raise AssertionError("Action keys do not match.")
    for key in official_actions.files:
      official_value = official_actions[key]
      modular_value = modular_actions[key]
      if official_value.shape != modular_value.shape:
        raise AssertionError(
            f"{key} shape mismatch: {official_value.shape} != "
            f"{modular_value.shape}"
        )
      error = np.abs(official_value - modular_value)
      match = np.allclose(
          official_value, modular_value, rtol=args.rtol, atol=args.atol
      )
      print(f"{key}: max error={float(error.max())}, match={match}")
      all_match = all_match and match

  print(f"Match: {all_match}")
  if not all_match:
    raise AssertionError(
        "Official end-to-end and modular TensorFlow actions do not match."
    )


if __name__ == "__main__":
  main()
