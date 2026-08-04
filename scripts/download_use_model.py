"""Download the pinned Universal Sentence Encoder used for validation."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub


MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
REPOSITORY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_DIR / "models" / "universal_sentence_encoder_large" / "5"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Download Universal Sentence Encoder Large /5 from TensorFlow Hub "
          "and store an explicit local copy for reproducible RT-1 validation."
      )
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=DEFAULT_OUTPUT_DIR,
      help="Directory in which to store the TensorFlow SavedModel.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  output_dir = args.output_dir.expanduser().resolve()

  if output_dir.exists():
    raise FileExistsError(
        f"Output directory already exists: {output_dir}\n"
        "Remove it explicitly before downloading the model again."
    )

  output_dir.parent.mkdir(parents=True, exist_ok=True)
  cached_model_dir = Path(hub.resolve(MODEL_URL))

  with tempfile.TemporaryDirectory(
      prefix=f".{output_dir.name}-", dir=output_dir.parent
  ) as temporary_dir:
    temporary_model_dir = Path(temporary_dir) / "model"
    shutil.copytree(cached_model_dir, temporary_model_dir)

    metadata = {
        "model_url": MODEL_URL,
        "tensorflow_version": tf.__version__,
        "tensorflow_hub_version": hub.__version__,
    }
    (temporary_model_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary_model_dir.rename(output_dir)

  print(f"Saved USE Large /5 to {output_dir}")


if __name__ == "__main__":
  main()
