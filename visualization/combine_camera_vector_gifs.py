"""Place the episode camera GIF and RT-1 vector GIF side by side."""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


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
  parser.add_argument(
      "--camera-caption",
      default="RT-1 Camera",
      help="Caption shown above the camera GIF.",
  )
  parser.add_argument(
      "--vector-caption",
      default="ONNX RT-1 Action (MuJoCo)",
      help="Caption shown above the MuJoCo GIF.",
  )
  parser.add_argument(
      "--no-captions",
      action="store_true",
      help="Combine the GIFs without the caption bar.",
  )
  parser.add_argument(
      "--font",
      type=Path,
      help="Optional TrueType/OpenType font used for captions.",
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


def _caption_font(
    font_size: int, requested_font: Path | None
) -> ImageFont.FreeTypeFont:
  if requested_font is not None:
    font_path = requested_font.expanduser().resolve()
    if not font_path.is_file():
      raise FileNotFoundError(f"Caption font not found: {font_path}")
    return ImageFont.truetype(str(font_path), font_size)

  candidates = (
      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
      Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
      Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
      Path("C:/Windows/Fonts/arialbd.ttf"),
  )
  for font_path in candidates:
    if font_path.is_file():
      return ImageFont.truetype(str(font_path), font_size)

  raise FileNotFoundError(
      "No TrueType caption font was found. Install DejaVu Sans or pass "
      "--font with a .ttf/.otf path."
  )


def _draw_centered_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    center_y: int,
    font: ImageFont.ImageFont,
) -> None:
  bounds = draw.textbbox((0, 0), text, font=font)
  text_width = bounds[2] - bounds[0]
  text_height = bounds[3] - bounds[1]
  draw.text(
      (center_x - text_width / 2, center_y - text_height / 2 - bounds[1]),
      text,
      font=font,
      fill=(245, 247, 250),
  )


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
  caption_height = 0 if args.no_captions else max(30, round(height * 0.13))
  font = _caption_font(
      max(15, round(caption_height * 0.52)), args.font
  )
  combined_frames = []
  for timestamp in timestamps:
    camera_frame = _frame_at_time(camera_frames, camera_ends, timestamp)
    vector_frame = _frame_at_time(vector_frames, vector_ends, timestamp)
    combined = Image.new(
        "RGB", (width * 2, height + caption_height), (18, 21, 26)
    )
    combined.paste(camera_frame, (0, caption_height))
    combined.paste(vector_frame, (width, caption_height))
    if caption_height:
      draw = ImageDraw.Draw(combined)
      draw.line(
          (width, 0, width, caption_height), fill=(70, 76, 86), width=1
      )
      _draw_centered_caption(
          draw, args.camera_caption, width // 2, caption_height // 2, font
      )
      _draw_centered_caption(
          draw,
          args.vector_caption,
          width + width // 2,
          caption_height // 2,
          font,
      )
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
  print(f"Output size: {width * 2}x{height + caption_height}")
  if caption_height:
    print(f"Captions: {args.camera_caption!r} | {args.vector_caption!r}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
