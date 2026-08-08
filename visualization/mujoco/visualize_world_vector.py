"""Visualize an RT-1 episode's cumulative world-vector trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE = (
    REPOSITORY_DIR
    / "validation_artifacts"
    / "episode_00001"
    / "episode"
    / "onnx.json"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Animate cumulative RT-1 x/y/z action deltas in MuJoCo."
  )
  parser.add_argument(
      "--episode-json",
      type=Path,
      default=DEFAULT_EPISODE,
      help="Episode JSON produced by the TensorFlow or ONNX validator.",
  )
  parser.add_argument(
      "--fps",
      type=float,
      default=10.0,
      help="Animation frames per second.",
  )
  parser.add_argument(
      "--scale",
      type=float,
      default=0.25,
      help="Display scale applied to normalized world-vector deltas.",
  )
  parser.add_argument(
      "--once",
      action="store_true",
      help="Play once instead of looping until the viewer is closed.",
  )
  return parser.parse_args()


def _load_trajectory(path: Path, scale: float) -> tuple[dict, np.ndarray]:
  with path.open("r", encoding="utf-8") as file:
    episode = json.load(file)

  steps = episode.get("steps", [])
  if not steps:
    raise ValueError(f"Episode contains no steps: {path}")

  deltas = np.asarray(
      [step["actions"]["world_vector"] for step in steps], dtype=np.float64
  )
  if deltas.ndim != 2 or deltas.shape[1] != 3:
    raise ValueError(f"Expected world vectors with shape [T, 3], got {deltas.shape}")

  trajectory = np.cumsum(deltas, axis=0) * scale

  # Center the drawing horizontally and lift it above the floor. These offsets
  # affect only the display; the relative RT-1 trajectory remains unchanged.
  trajectory[:, :2] -= trajectory[:, :2].mean(axis=0)
  trajectory[:, 2] += 0.06 - min(0.0, float(trajectory[:, 2].min()))
  return episode, trajectory


def _build_model(trajectory: np.ndarray) -> mujoco.MjModel:
  xml = f"""
<mujoco model="rt1_world_vector">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="135" elevation="-25"/>
    <rgba haze="0.15 0.20 0.25 1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22"
             rgb2="0.24 0.26 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <mesh name="tapered_finger"
          vertex="0 -0.016 -0.020  0 0.016 -0.020  0 0.016 0.020  0 -0.016 0.020
                  0.087 -0.010 -0.007  0.087 0.010 -0.007
                  0.087 0.010 0.007  0.087 -0.010 0.007"
          face="0 3 2  0 2 1  4 5 6  4 6 7
                0 1 5  0 5 4  3 7 6  3 6 2
                0 4 7  0 7 3  1 2 6  1 6 5"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <body name="cursor" mocap="true" pos="0 0 0.1">
      <geom name="arm_link" type="capsule"
            fromto="-0.010 0 0 -0.105 0 0" size="0.030"
            rgba="0.45 0.47 0.50 1" contype="0" conaffinity="0"/>
    </body>
    <body name="upper_finger" mocap="true" pos="0 0 0.014">
      <geom type="mesh" mesh="tapered_finger"
            rgba="0.82 0.73 0.58 1" contype="0" conaffinity="0"/>
    </body>
    <body name="lower_finger" mocap="true" pos="0 0 -0.014">
      <geom type="mesh" mesh="tapered_finger"
            rgba="0.82 0.73 0.58 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
  return mujoco.MjModel.from_xml_string(xml)


def _play(model: mujoco.MjModel, trajectory: np.ndarray, fps: float, once: bool) -> None:
  if fps <= 0:
    raise ValueError("--fps must be greater than zero")

  data = mujoco.MjData(model)
  cursor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cursor")
  mocap_id = model.body_mocapid[cursor_body]
  frame_duration = 1.0 / fps

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      for index, position in enumerate(trajectory):
        if not viewer.is_running():
          return

        start = time.monotonic()
        data.mocap_pos[mocap_id] = position
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(max(0.0, frame_duration - (time.monotonic() - start)))

      if once:
        while viewer.is_running():
          viewer.sync()
          time.sleep(0.05)
        return



def main() -> None:
  args = _parse_args()
  episode_path = args.episode_json.expanduser().resolve()
  episode, trajectory = _load_trajectory(episode_path, args.scale)
  model = _build_model(trajectory)

  print(f"Episode: {episode.get('episode_index', 'unknown')}")
  print(f"Instruction: {episode.get('instruction', '')}")
  print(f"Points: {len(trajectory)}")
  print(f"Display scale: {args.scale}")
  print("Red sphere: current gripper position")
  _play(model, trajectory, args.fps, args.once)


if __name__ == "__main__":
  main()
