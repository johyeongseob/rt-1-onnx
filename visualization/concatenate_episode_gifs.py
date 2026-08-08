"""Concatenate combined episode GIFs along the time axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPOSITORY_DIR / "visualization_artifacts"
DEFAULT_RESULTS_DIR = REPOSITORY_DIR / "validation_artifacts"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Play combined episode GIFs consecutively in one GIF."
  )
  parser.add_argument("--start-episode", type=int, default=1)
  parser.add_argument("--end-episode", type=int, default=10)
  parser.add_argument(
      "--episodes",
      type=int,
      nargs="+",
      default=None,
      help=(
          "Exact episode indices to concatenate in the provided order. "
          "Overrides --start-episode and --end-episode."
      ),
  )
  parser.add_argument(
      "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR
  )
  parser.add_argument(
      "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=(
          DEFAULT_ARTIFACTS_DIR
          / "camera_and_world_vector_episodes_00001_00010.gif"
      ),
  )
  parser.add_argument(
      "--frame-step",
      type=int,
      default=1,
      help="Keep one frame per N decoded frames while preserving duration.",
  )
  parser.add_argument(
      "--scale",
      type=float,
      default=1.0,
      help="Output image scale relative to the input GIF dimensions.",
  )
  parser.add_argument(
      "--colors",
      type=int,
      default=256,
      help="Maximum colors per output GIF frame.",
  )
  parser.add_argument(
      "--optimize", action="store_true", help="Enable Pillow GIF optimization."
  )
  parser.add_argument(
      "--speed",
      type=float,
      default=1.0,
      help="Playback speed multiplier. For example, 3 produces a 3x GIF.",
  )
  parser.add_argument(
      "--caption-height",
      type=int,
      default=0,
      help=(
          "Add a top caption area of this many pixels and display each "
          "episode instruction. Zero disables captions."
      ),
  )
  parser.add_argument(
      "--caption-font-size", type=int, default=20
  )
  parser.add_argument(
      "--label-height",
      type=int,
      default=0,
      help="Add a bottom label area of this many pixels.",
  )
  parser.add_argument("--left-label", type=str, default="Camera")
  parser.add_argument("--right-label", type=str, default="ONNX inference")
  parser.add_argument("--label-font-size", type=int, default=18)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.start_episode < 0:
    raise ValueError("--start-episode must be zero or greater.")
  if args.end_episode < args.start_episode:
    raise ValueError("--end-episode must not precede --start-episode.")
  if args.frame_step < 1:
    raise ValueError("--frame-step must be at least 1.")
  if args.scale <= 0:
    raise ValueError("--scale must be greater than zero.")
  if not 2 <= args.colors <= 256:
    raise ValueError("--colors must be between 2 and 256.")
  if args.speed <= 0:
    raise ValueError("--speed must be greater than zero.")
  if args.caption_height < 0:
    raise ValueError("--caption-height must be zero or greater.")
  if args.caption_font_size < 1:
    raise ValueError("--caption-font-size must be at least 1.")
  if args.label_height < 0:
    raise ValueError("--label-height must be zero or greater.")
  if args.label_font_size < 1:
    raise ValueError("--label-font-size must be at least 1.")

  artifacts_dir = args.artifacts_dir.expanduser().resolve()
  results_dir = args.results_dir.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  episodes = (
      args.episodes
      if args.episodes is not None
      else list(range(args.start_episode, args.end_episode + 1))
  )
  if not episodes or any(episode < 0 for episode in episodes):
    raise ValueError("Episode indices must be zero or greater.")
  input_paths = [
      artifacts_dir
      / f"episode_{episode:05d}"
      / "camera_and_world_vector.gif"
      for episode in episodes
  ]
  missing = [path for path in input_paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"Combined episode GIFs were not found: {missing}")

  frames: list[Image.Image] = []
  durations: list[int] = []
  captions: list[str] = []
  expected_size: tuple[int, int] | None = None
  for episode, path in zip(episodes, input_paths):
    instruction = ""
    if args.caption_height:
      result_path = (
          results_dir / f"episode_{episode:05d}" / "episode" / "onnx.json"
      )
      if not result_path.is_file():
        raise FileNotFoundError(
            f"Episode result JSON was not found for the caption: {result_path}"
        )
      result = json.loads(result_path.read_text(encoding="utf-8"))
      instruction = str(result["instruction"])
    episode_frame_count = 0
    with Image.open(path) as image:
      fallback_duration = int(image.info.get("duration", 100))
      if expected_size is None:
        expected_size = image.size
      elif image.size != expected_size:
        raise ValueError(
            f"GIF size mismatch: expected {expected_size}, "
            f"received {image.size} from {path}."
        )
      for frame in ImageSequence.Iterator(image):
        frames.append(frame.convert("RGB"))
        durations.append(int(frame.info.get("duration", fallback_duration)))
        captions.append(instruction)
        episode_frame_count += 1
    print(f"Episode GIF: {path} ({episode_frame_count} decoded frames)")

  if not frames:
    raise ValueError("No GIF frames were decoded.")

  sampled_frames = []
  sampled_durations = []
  sampled_captions = []
  for start in range(0, len(frames), args.frame_step):
    sampled_frames.append(frames[start])
    source_duration = sum(durations[start:start + args.frame_step])
    sampled_durations.append(max(10, round(source_duration / args.speed)))
    sampled_captions.append(captions[start])
  frames = sampled_frames
  durations = sampled_durations
  captions = sampled_captions

  width = max(1, round(expected_size[0] * args.scale))
  height = max(1, round(expected_size[1] * args.scale))
  output_size = (width, height)
  rendered_frames = []
  for frame, caption in zip(frames, captions):
    frame = frame.resize(output_size, Image.Resampling.LANCZOS)
    if args.caption_height or args.label_height:
      canvas = Image.new(
          "RGB",
          (width, height + args.caption_height + args.label_height),
          (235, 235, 235),
      )
      canvas.paste(frame, (0, args.caption_height))
      draw = ImageDraw.Draw(canvas)
      if args.caption_height:
        font_size = args.caption_font_size
        while True:
          try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
          except OSError:
            font = ImageFont.load_default()
            break
          bounds = draw.textbbox((0, 0), caption, font=font)
          if bounds[2] - bounds[0] <= width - 20 or font_size <= 10:
            break
          font_size -= 1
        bounds = draw.textbbox((0, 0), caption, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (
                (width - text_width) // 2,
                (args.caption_height - text_height) // 2,
            ),
            caption,
            fill=(20, 20, 20),
            font=font,
        )
      if args.label_height:
        try:
          label_font = ImageFont.truetype(
              "DejaVuSans.ttf", args.label_font_size
          )
        except OSError:
          label_font = ImageFont.load_default()
        label_y = args.caption_height + height
        for center_x, label in (
            (width // 4, args.left_label),
            (3 * width // 4, args.right_label),
        ):
          bounds = draw.textbbox((0, 0), label, font=label_font)
          text_width = bounds[2] - bounds[0]
          text_height = bounds[3] - bounds[1]
          draw.text(
              (
                  center_x - text_width // 2,
                  label_y + (args.label_height - text_height) // 2,
              ),
              label,
              fill=(20, 20, 20),
              font=label_font,
          )
      frame = canvas
    rendered_frames.append(
        frame.convert("P", palette=Image.ADAPTIVE, colors=args.colors)
    )
  frames = rendered_frames

  output_path.parent.mkdir(parents=True, exist_ok=True)
  frames[0].save(
      output_path,
      save_all=True,
      append_images=frames[1:],
      duration=durations,
      loop=0,
      disposal=2,
      optimize=args.optimize,
  )

  print(f"Episodes: {episodes}")
  print(f"Output frames: {len(frames)}")
  print(f"Frame step: {args.frame_step}")
  print(
      f"Frame size: {width}x"
      f"{height + args.caption_height + args.label_height}"
  )
  print(f"Colors: {args.colors}")
  print(f"Playback speed: {args.speed}x")
  print(f"Caption height: {args.caption_height}")
  print(f"Label height: {args.label_height}")
  print(f"Total duration: {sum(durations) / 1000.0:.2f} seconds")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
