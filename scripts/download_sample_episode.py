"""Download reproducible RT-1 dataset episodes for shared validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DEFAULT_DATASET_DIR = (
    "gs://gresearch/robotics/fractal20220817_data/0.1.0"
)
REPOSITORY_DIR = Path(__file__).resolve().parent.parent


def _to_json(value: Any) -> Any:
  """Convert TensorFlow and NumPy values into JSON-compatible values."""
  if isinstance(value, tf.Tensor):
    value = value.numpy()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, dict):
    return {key: _to_json(item) for key, item in value.items()}
  return value


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Read the public Fractal RT-1 dataset from Google Cloud Storage "
          "and save one or more episodes locally."
      )
  )
  parser.add_argument(
      "--dataset-dir",
      default=DEFAULT_DATASET_DIR,
      help="TFDS directory containing fractal20220817_data.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=REPOSITORY_DIR / "data" / "fractal_samples",
      help=(
          "Parent directory in which to create an episode_XXXXX directory."
      ),
  )
  parser.add_argument(
      "--episode-index",
      type=int,
      default=None,
      help=(
          "Exact zero-based episode index to save. By default, save the first "
          "episode marked successful."
      ),
  )
  parser.add_argument(
      "--start-index",
      type=int,
      default=None,
      help="First zero-based episode index in an inclusive range.",
  )
  parser.add_argument(
      "--end-index",
      type=int,
      default=None,
      help="Last zero-based episode index in an inclusive range.",
  )
  parser.add_argument(
      "--max-episodes",
      type=int,
      default=1000,
      help="Maximum number of episodes to inspect when searching for success.",
  )
  return parser.parse_args()


def _select_episodes(
    dataset: tf.data.Dataset,
    episode_index: int | None,
    start_index: int | None,
    end_index: int | None,
    max_episodes: int,
) -> list[tuple[int, dict[str, Any]]]:
  if episode_index is not None and episode_index < 0:
    raise ValueError("--episode-index must be zero or greater.")
  if (start_index is None) != (end_index is None):
    raise ValueError("--start-index and --end-index must be used together.")
  if episode_index is not None and start_index is not None:
    raise ValueError(
        "--episode-index cannot be combined with --start-index/--end-index."
    )
  if start_index is not None:
    if start_index < 0 or end_index < start_index:
      raise ValueError(
          "The range must satisfy 0 <= --start-index <= --end-index."
      )
    count = end_index - start_index + 1
    episodes = list(dataset.skip(start_index).take(count))
    if len(episodes) != count:
      raise RuntimeError(
          f"Only {len(episodes)} episodes were found in the requested range "
          f"{start_index}..{end_index}."
      )
    return [
        (start_index + offset, episode)
        for offset, episode in enumerate(episodes)
    ]

  if episode_index is not None:
    episodes = list(dataset.skip(episode_index).take(1))
    if not episodes:
      raise RuntimeError(f"Episode {episode_index} was not found.")
    return [(episode_index, episodes[0])]

  for index, episode in enumerate(dataset.take(max_episodes)):
    aspects = episode["aspects"]
    if (
        bool(aspects["has_aspects"].numpy())
        and bool(aspects["success"].numpy())
    ):
      return [(index, episode)]

  raise RuntimeError(
      f"No annotated successful episode was found within the first "
      f"{max_episodes} episodes."
  )


def _save_episode(
    episode: dict[str, Any],
    index: int,
    output_dir: Path,
    dataset_dir: str,
) -> None:
  episode_dir = output_dir / f"episode_{index:05d}"
  if episode_dir.exists():
    print(f"Skipping episode {index}: {episode_dir} already exists")
    return

  steps = list(episode["steps"])
  if not steps:
    raise RuntimeError(f"Episode {index} contains no steps.")

  frames_dir = episode_dir / "frames"
  frames_dir.mkdir(parents=True)

  first_observation = steps[0]["observation"]
  instruction = _to_json(
      first_observation["natural_language_instruction"]
  )
  language_embedding = first_observation[
      "natural_language_embedding"
  ].numpy()
  np.save(episode_dir / "language_embedding.npy", language_embedding)

  step_records = []
  for step_index, step in enumerate(steps):
    observation = step["observation"]
    frame_path = frames_dir / f"frame_{step_index:04d}.png"
    tf.io.write_file(
        str(frame_path), tf.io.encode_png(observation["image"])
    )
    step_records.append({
        "frame": str(frame_path.relative_to(episode_dir)),
        "action": _to_json(step["action"]),
        "reward": _to_json(step["reward"]),
        "is_first": _to_json(step["is_first"]),
        "is_last": _to_json(step["is_last"]),
        "is_terminal": _to_json(step["is_terminal"]),
    })

  metadata = {
      "source": dataset_dir,
      "split": "train",
      "episode_index": index,
      "instruction": instruction,
      "num_steps": len(steps),
      "image_shape": list(first_observation["image"].shape),
      "language_embedding_shape": list(language_embedding.shape),
      "aspects": _to_json(episode["aspects"]),
      "attributes": _to_json(episode["attributes"]),
  }

  with (episode_dir / "metadata.json").open(
      "w", encoding="utf-8"
  ) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)
  with (episode_dir / "steps.json").open(
      "w", encoding="utf-8"
  ) as file:
    json.dump(step_records, file, ensure_ascii=False, indent=2)

  print(f"Saved episode {index} to {episode_dir}")
  print(f"Instruction: {instruction}")
  print(f"Frames: {len(steps)}")
  if metadata["aspects"]["has_aspects"]:
    print(f"Success: {metadata['aspects']['success']}")
  else:
    print("Success: Unknown (not annotated)")


def main() -> None:
  args = _parse_args()
  print(f"Opening dataset: {args.dataset_dir}")
  builder = tfds.builder_from_directory(args.dataset_dir)
  dataset = builder.as_dataset(split="train", shuffle_files=False)

  episodes = _select_episodes(
      dataset,
      args.episode_index,
      args.start_index,
      args.end_index,
      args.max_episodes,
  )
  for index, episode in episodes:
    _save_episode(
        episode, index, args.output_dir, args.dataset_dir
    )


if __name__ == "__main__":
  main()
