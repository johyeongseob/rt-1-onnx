"""Convert the trained rt1main Transformer to ONNX."""

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

from transformer_checkpoint import restore_transformer  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_DIR / "models" / "transformer" / "transformer.onnx"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Restore rt1main Transformer and convert it to ONNX."
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

  sequence_spec = tf.TensorSpec(
      [None, 114, 512], tf.float32, name="sequence"
  )
  mask_spec = tf.TensorSpec([114, 114], tf.float32, name="attention_mask")
  dummy_sequence = tf.zeros([1, 114, 512], dtype=tf.float32)
  dummy_mask = tf.ones([114, 114], dtype=tf.float32)
  decoder = restore_transformer(model_dir, dummy_sequence, dummy_mask)

  @tf.function(input_signature=[sequence_spec, mask_spec])
  def inference(
      sequence: tf.Tensor, attention_mask: tf.Tensor
  ) -> dict[str, tf.Tensor]:
    logits, _ = decoder(
        sequence, training=False, attention_mask=attention_mask
    )
    return {"logits": logits}

  output_path.parent.mkdir(parents=True, exist_ok=True)
  tf2onnx.convert.from_function(
      inference,
      input_signature=[sequence_spec, mask_spec],
      opset=args.opset,
      output_path=str(output_path),
  )
  print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
  main()
