"""Run one RT-1 SavedModel action step with deterministic dummy inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp


def zeros_for_spec(spec: tf.TensorSpec, batch_size: int) -> tf.Tensor:
    shape = [batch_size if size is None else size for size in spec.shape]
    if spec.dtype == tf.string:
        return tf.fill(shape, "")
    return tf.zeros(shape, dtype=spec.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    # Register the distribution TypeSpec stored in the exported policy.
    tfp.distributions.Deterministic(loc=tf.constant(0.0))

    model_dir = args.model_dir.resolve()
    print(f"Loading SavedModel: {model_dir}")
    model = tf.saved_model.load(str(model_dir))

    batch_size = 1
    initial_state = model.signatures["get_initial_state"](
        batch_size=tf.constant(batch_size, dtype=tf.int32)
    )
    print("Initial state:")
    for name, value in sorted(initial_state.items()):
        print(f"  {name}: shape={value.shape}, dtype={value.dtype.name}")

    action_fn = model.signatures["action"]
    input_specs = action_fn.structured_input_signature[1]
    inputs = {
        name: zeros_for_spec(spec, batch_size)
        for name, spec in input_specs.items()
    }

    for state_name, value in initial_state.items():
        inputs[f"1/{state_name}"] = value

    # Use valid identity quaternions instead of an all-zero rotation.
    inputs["0/observation/orientation_start"] = tf.constant(
        [[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32
    )
    inputs["0/observation/src_rotation"] = tf.constant(
        [[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32
    )
    inputs["0/discount"] = tf.ones([batch_size], dtype=tf.float32)
    inputs["0/observation/natural_language_instruction"] = tf.constant(
        ["dummy instruction"]
    )

    print("Running one action step...")
    outputs = action_fn(**inputs)
    print("Inference completed.")

    for name, value in sorted(outputs.items()):
        if name.startswith("action/") or name.startswith("info/"):
            print(
                f"{name}: shape={value.shape}, dtype={value.dtype.name}, "
                f"value={value.numpy().tolist()}"
            )


if __name__ == "__main__":
    main()
