"""Convert the trained rt1main FiLM-EfficientNet to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import tensorflow as tf
import tf2onnx


ONNX_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = ONNX_DIR.parent
OFFICIAL_DIR = REPOSITORY_DIR / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from film_efficientnet_checkpoint import restore_encoder  # pylint: disable=g-import-not-at-top


DEFAULT_MODEL_DIR = (
    OFFICIAL_DIR
    / "robotics_transformer"
    / "trained_checkpoints"
    / "rt1main"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_DIR
    / "models"
    / "film_efficientnet"
    / "film_efficientnet.onnx"
)

# tf2onnx exposes these fixed values as inputs. They reflect the tensors used
# by the restored RT-1 graph: zero centering and ImageNet standard-deviation
# rescaling.
_CAPTURED_PREPROCESSING_CONSTANTS = {
    "rescaling_3/mul/y:0": np.asarray(
        1.0, dtype=np.float32
    ) / np.sqrt(np.asarray([0.229, 0.224, 0.225], dtype=np.float32)),
    "normalization_1/sub/y:0": np.asarray(
        [0.0, 0.0, 0.0], dtype=np.float32
    ),
}

_UNUSED_CAPTURED_INPUTS = {
    "normalization_1/Sqrt/x:0",
}


def _freeze_captured_preprocessing_constants(model_path: Path) -> None:
  """Replace tf2onnx-captured preprocessing inputs with ONNX constants."""
  model = onnx.load(str(model_path))
  graph_inputs = {value.name: value for value in model.graph.input}
  expected_suffixes = (
      set(_CAPTURED_PREPROCESSING_CONSTANTS) | _UNUSED_CAPTURED_INPUTS
  )
  captured_inputs = {}
  for suffix in expected_suffixes:
    matches = [name for name in graph_inputs if name.endswith(suffix)]
    if len(matches) != 1:
      raise ValueError(
          f"Expected one captured input ending with {suffix!r}; "
          f"found {matches}"
      )
    captured_inputs[suffix] = matches[0]

  for suffix, value in _CAPTURED_PREPROCESSING_CONSTANTS.items():
    input_name = captured_inputs[suffix]
    model.graph.initializer.append(
        numpy_helper.from_array(value, name=input_name)
    )

  retained_inputs = [
      value
      for value in model.graph.input
      if value.name not in captured_inputs.values()
  ]
  del model.graph.input[:]
  model.graph.input.extend(retained_inputs)
  onnx.checker.check_model(model)
  onnx.save(model, str(model_path))


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Restore rt1main FiLM-EfficientNet and convert it to ONNX."
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

  image_spec = tf.TensorSpec(
      [None, 300, 300, 3], tf.float32, name="image"
  )
  context_spec = tf.TensorSpec([None, 512], tf.float32, name="context")
  dummy_image = tf.zeros([1, 300, 300, 3], dtype=tf.float32)
  dummy_context = tf.zeros([1, 512], dtype=tf.float32)
  encoder = restore_encoder(model_dir, dummy_image, dummy_context)

  @tf.function(input_signature=[image_spec, context_spec])
  def inference(image: tf.Tensor, context: tf.Tensor) -> dict[str, tf.Tensor]:
    return {
        "features": encoder(image, context=context, training=False),
    }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  tf2onnx.convert.from_function(
      inference,
      input_signature=[image_spec, context_spec],
      opset=args.opset,
      output_path=str(output_path),
  )
  _freeze_captured_preprocessing_constants(output_path)
  print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
  main()
