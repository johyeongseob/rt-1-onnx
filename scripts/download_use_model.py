"""Download the pinned ONNX USE Large /5 model used by RT-1 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODEL_URL = (
    "https://huggingface.co/SamLowe/"
    "universal-sentence-encoder-large-5-onnx/resolve/main/model.onnx"
)
MODEL_SHA256 = (
    "d267b0955793f866593e49ba58474fb57f5314cf757ec0945127c521c569b22f"
)
REPOSITORY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_DIR
    / "models"
    / "universal_sentence_encoder_large_onnx"
    / "5"
    / "model.onnx"
)
CHUNK_SIZE = 1024 * 1024


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Download the pinned community ONNX conversion of Universal "
          "Sentence Encoder Large /5."
      )
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=DEFAULT_OUTPUT_PATH,
      help="Destination path for model.onnx.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  output_path = args.output.expanduser().resolve()
  if output_path.exists():
    raise FileExistsError(
        f"Output file already exists: {output_path}\n"
        "Remove it explicitly before downloading the model again."
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  digest = hashlib.sha256()
  with tempfile.NamedTemporaryFile(
      prefix=f".{output_path.name}-",
      dir=output_path.parent,
      delete=False,
  ) as temporary_file:
    temporary_path = Path(temporary_file.name)
    try:
      with urlopen(MODEL_URL) as response:  # nosec B310: pinned HTTPS URL
        while chunk := response.read(CHUNK_SIZE):
          temporary_file.write(chunk)
          digest.update(chunk)
    except Exception:
      temporary_path.unlink(missing_ok=True)
      raise

  actual_sha256 = digest.hexdigest()
  if actual_sha256 != MODEL_SHA256:
    temporary_path.unlink(missing_ok=True)
    raise ValueError(
        "Downloaded model checksum mismatch: "
        f"expected {MODEL_SHA256}, received {actual_sha256}"
    )
  temporary_path.replace(output_path)

  metadata = {
      "source": MODEL_URL,
      "sha256": MODEL_SHA256,
      "conversion": (
          "SamLowe/universal-sentence-encoder-large-5-onnx"
      ),
      "original_model": (
          "https://tfhub.dev/google/universal-sentence-encoder-large/5"
      ),
  }
  metadata_path = output_path.with_name("download_metadata.json")
  metadata_path.write_text(
      json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
  )

  print(f"Saved ONNX USE Large /5 to {output_path}")
  print(f"SHA-256: {actual_sha256}")


if __name__ == "__main__":
  main()
