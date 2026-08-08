"""Export episode camera frames as an interpolated comparison GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_DIR = (
    REPOSITORY_DIR / "data" / "fractal_samples" / "episode_00001" / "frames"
)
DEFAULT_OUTPUT = (
    REPOSITORY_DIR
    / "visualization_artifacts"
    / "episode_00001"
    / "camera_frames.gif"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Create an interpolated GIF from an episode's camera frames."
  )
  parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--fps", type=int, default=12)
  parser.add_argument("--source-hz", type=float, default=3.0)
  return parser.parse_args()


def _load_frames(frames_dir: Path) -> list[Image.Image]:
  frame_paths = sorted(frames_dir.glob("frame_*.png"))
  if not frame_paths:
    raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

  frames = []
  for path in frame_paths:
    with Image.open(path) as image:
      frames.append(image.convert("RGB"))

  sizes = {frame.size for frame in frames}
  if len(sizes) != 1:
    raise ValueError(f"All source frames must have the same size, got {sizes}")
  return frames


def _interpolate_frames(
    source_frames: list[Image.Image], frames_per_source: int
) -> list[Image.Image]:
  output_frames = []

  for index, target in enumerate(source_frames):
    start = source_frames[index - 1] if index > 0 else target
    for subframe in range(frames_per_source):
      fraction = subframe / frames_per_source
      blended = Image.blend(start, target, fraction)
      output_frames.append(
          blended.convert("P", palette=Image.ADAPTIVE)
      )

  output_frames.append(
      source_frames[-1].convert("P", palette=Image.ADAPTIVE)
  )
  return output_frames


def main() -> None:
  args = _parse_args()
  if args.fps <= 0 or args.source_hz <= 0:
    raise ValueError("--fps and --source-hz must be greater than zero")

  frames_dir = args.frames_dir.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  frames_per_source = max(1, round(args.fps / args.source_hz))

  source_frames = _load_frames(frames_dir)
  output_frames = _interpolate_frames(source_frames, frames_per_source)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_frames[0].save(
      output_path,
      save_all=True,
      append_images=output_frames[1:],
      duration=round(1000 / args.fps),
      loop=0,
      disposal=2,
  )

  print(f"Frames directory: {frames_dir}")
  print(f"Source frames: {len(source_frames)}")
  print(f"Interpolated frames per source frame: {frames_per_source}")
  print(f"GIF frames: {len(output_frames)}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
