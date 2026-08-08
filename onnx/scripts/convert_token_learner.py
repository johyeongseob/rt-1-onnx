"""Convert the trained rt1main TokenLearner to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tensorflow as tf
import tf2onnx


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
OFFICIAL_DIR = REPOSITORY_DIR / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from token_learner_checkpoint import restore_token_learner  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_DIR / "models" / "token_learner" / "token_learner.onnx"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Restore rt1main TokenLearner and convert it to ONNX."
  )
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
  parser.add_argument("--opset", type=int, default=17)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  model_dir = args.model_dir.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  if args.opset < 1:
    raise ValueError("--opset must be positive.")
  if not (model_dir / "saved_model.pb").is_file():
    raise FileNotFoundError(f"rt1main SavedModel was not found: {model_dir}")

  features_spec = tf.TensorSpec(
      [None, 9, 9, 512], tf.float32, name="features"
  )
  dummy_features = tf.zeros([1, 9, 9, 512], dtype=tf.float32)
  learner = restore_token_learner(model_dir, dummy_features)

  @tf.function(input_signature=[features_spec])
  def inference(features: tf.Tensor) -> dict[str, tf.Tensor]:
    flattened = tf.reshape(features, [tf.shape(features)[0], 81, 512])
    selected = learner.layernorm(flattened)
    selected = learner.mlp(selected, is_training=False)
    selected = tf.transpose(selected, [0, 2, 1])
    selected = tf.nn.softmax(selected, axis=-1)
    return {"tokens": tf.matmul(selected, flattened)}

  output_path.parent.mkdir(parents=True, exist_ok=True)
  tf2onnx.convert.from_function(
      inference,
      input_signature=[features_spec],
      opset=args.opset,
      output_path=str(output_path),
  )
  print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
  main()
