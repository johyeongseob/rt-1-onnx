"""Compare complete-episode TensorFlow and ONNX JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
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
  directory = (
      args.artifacts_dir.expanduser().resolve()
      / f"episode_{args.episode_index:05d}"
      / "episode"
  )
  official = json.loads((directory / "official.json").read_text("utf-8"))
  onnx = json.loads((directory / "onnx.json").read_text("utf-8"))
  for key in ("episode_index", "instruction", "num_frames"):
    if official[key] != onnx[key]:
      raise AssertionError(f"Metadata mismatch for {key}: {official[key]!r} != {onnx[key]!r}")
  if len(official["steps"]) != len(onnx["steps"]):
    raise AssertionError("The number of output steps does not match.")

  token_mismatches = []
  action_mismatches = []
  max_error = 0.0
  for tf_step, onnx_step in zip(official["steps"], onnx["steps"]):
    index = tf_step["frame_index"]
    if index != onnx_step["frame_index"]:
      raise AssertionError("Frame indices do not match.")
    if tf_step["action_tokens"] != onnx_step["action_tokens"]:
      token_mismatches.append(index)
    if set(tf_step["actions"]) != set(onnx_step["actions"]):
      raise AssertionError(f"Action keys do not match at frame {index}.")
    for key in tf_step["actions"]:
      tf_value = np.asarray(tf_step["actions"][key])
      onnx_value = np.asarray(onnx_step["actions"][key])
      error = float(np.max(np.abs(tf_value - onnx_value)))
      max_error = max(max_error, error)
      if not np.allclose(
          tf_value, onnx_value, rtol=args.rtol, atol=args.atol
      ):
        action_mismatches.append((index, key, error))

  match = not token_mismatches and not action_mismatches
  print(f"Frames compared: {official['num_frames']}")
  print(f"Token mismatch frames: {token_mismatches}")
  print(f"Action mismatches: {action_mismatches}")
  print(f"Maximum absolute action error: {max_error}")
  print(f"Match: {match}")
  if not match:
    raise AssertionError("Complete-episode outputs do not match.")


if __name__ == "__main__":
  main()
