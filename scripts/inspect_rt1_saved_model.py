"""Inspect an RT-1 SavedModel before constructing a smoke-test input."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing saved_model.pb.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    # Saved RT-1 policies expose a TFP Deterministic distribution. Constructing
    # one registers its CompositeTensor TypeSpec before SavedModel decoding.
    tfp.distributions.Deterministic(loc=tf.constant(0.0))
    print(f"Loading SavedModel: {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    print("SavedModel loaded.")

    signature_names = list(model.signatures)
    print(f"Signatures: {signature_names}")
    for name, function in model.signatures.items():
        print(f"\nSignature: {name}")
        print(f"Inputs: {function.structured_input_signature}")
        print(f"Outputs: {function.structured_outputs}")

    variables = list(getattr(model, "variables", ()))
    print(f"\nVariables exposed by SavedModel: {len(variables)}")
    for variable in variables[:20]:
        print(f"{variable.name}: shape={variable.shape}, dtype={variable.dtype.name}")


if __name__ == "__main__":
    main()
