"""Place the episode camera GIF and RT-1 vector GIF side by side."""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

from PIL import Image, ImageSequence


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_DIR / "visualization_artifacts" / "episode_00001"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Combine synchronized camera and world-vector GIFs."
  )
  parser.add_argument(
      "--camera-gif",
      type=Path,
      default=DEFAULT_ARTIFACT_DIR / "camera_frames.gif",
  )
  parser.add_argument(
      "--vector-gif",
      type=Path,
      default=DEFAULT_ARTIFACT_DIR / "world_vector.gif",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=DEFAULT_ARTIFACT_DIR / "camera_and_world_vector.gif",
  )
  return parser.parse_args()


def _read_gif(path: Path) -> tuple[list[Image.Image], list[int], int]:
  with Image.open(path) as image:
    loop = int(image.info.get("loop", 0))
    frames = []
    durations = []
    for frame in ImageSequence.Iterator(image):
      frames.append(frame.convert("RGB"))
      durations.append(int(frame.info.get("duration", image.info.get("duration", 100))))
  return frames, durations, loop


def _frame_at_time(
    frames: list[Image.Image], cumulative_ends: list[int], time_ms: int
) -> Image.Image:
  index = bisect.bisect_right(cumulative_ends, time_ms)
  return frames[min(index, len(frames) - 1)]


def main() -> None:
  args = _parse_args()
  camera_path = args.camera_gif.expanduser().resolve()
  vector_path = args.vector_gif.expanduser().resolve()
  output_path = args.output.expanduser().resolve()

  camera_frames, camera_durations, loop = _read_gif(camera_path)
  vector_frames, vector_durations, _ = _read_gif(vector_path)

  if camera_frames[0].size != vector_frames[0].size:
    raise ValueError(
        "GIF sizes differ: "
        f"camera={camera_frames[0].size}, vector={vector_frames[0].size}. "
        "Regenerate the vector GIF with the current default settings."
    )

  positive_durations = [
      duration
      for duration in camera_durations + vector_durations
      if duration > 0
  ]
  if not positive_durations:
    raise ValueError("The GIFs do not contain valid frame durations.")

  frame_duration = min(positive_durations)
  camera_ends = []
  vector_ends = []
  total = 0
  for duration in camera_durations:
    total += duration
    camera_ends.append(total)
  camera_total = total
  total = 0
  for duration in vector_durations:
    total += duration
    vector_ends.append(total)
  vector_total = total
  common_duration = min(camera_total, vector_total)
  timestamps = range(0, common_duration, frame_duration)

  width, height = camera_frames[0].size
  combined_frames = []
  for timestamp in timestamps:
    camera_frame = _frame_at_time(camera_frames, camera_ends, timestamp)
    vector_frame = _frame_at_time(vector_frames, vector_ends, timestamp)
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(camera_frame, (0, 0))
    combined.paste(vector_frame, (width, 0))
    combined_frames.append(
        combined.convert("P", palette=Image.ADAPTIVE)
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  combined_frames[0].save(
      output_path,
      save_all=True,
      append_images=combined_frames[1:],
      duration=frame_duration,
      loop=loop,
      disposal=2,
  )

  print(f"Camera: {camera_path}")
  print(f"Vector: {vector_path}")
  print(f"Decoded camera frames: {len(camera_frames)}")
  print(f"Decoded vector frames: {len(vector_frames)}")
  print(f"Synchronized frames: {len(combined_frames)}")
  print(f"Frame duration: {frame_duration} ms")
  print(f"Frame size: {width}x{height}")
  print(f"Output size: {width * 2}x{height}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
