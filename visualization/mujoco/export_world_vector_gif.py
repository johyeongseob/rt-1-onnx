"""Export an interpolated RT-1 world-vector trajectory as a MuJoCo GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from visualize_world_vector import (
    DEFAULT_EPISODE,
    REPOSITORY_DIR,
    _build_model,
    _load_trajectory,
)


DEFAULT_OUTPUT = REPOSITORY_DIR / "visualization_artifacts" / "episode_00001" / "world_vector.gif"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Render an RT-1 x/y/z action trajectory to an animated GIF."
  )
  parser.add_argument("--episode-json", type=Path, default=DEFAULT_EPISODE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument(
      "--fps", type=int, default=12, help="GIF playback frame rate."
  )
  parser.add_argument(
      "--action-hz",
      type=float,
      default=3.0,
      help="Rate of the original RT-1 action points.",
  )
  parser.add_argument("--scale", type=float, default=0.25)
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=256)
  parser.add_argument(
      "--zoom",
      type=float,
      default=1.5,
      help="Camera zoom multiplier; values greater than one zoom in.",
  )
  return parser.parse_args()


def _camera_for(trajectory: np.ndarray, zoom: float) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  center = (trajectory.min(axis=0) + trajectory.max(axis=0)) / 2.0
  span = np.ptp(trajectory, axis=0)
  camera.lookat[:] = center
  camera.distance = max(0.35, float(span.max()) * 2.4) / zoom
  camera.azimuth = -45.0
  camera.elevation = -25.0
  return camera


def _interpolated_positions(
    trajectory: np.ndarray, frames_per_action: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  positions = []
  action_indices = []
  fractions = []

  for index, target in enumerate(trajectory):
    start = trajectory[index - 1] if index > 0 else target
    for fraction in np.linspace(0.0, 1.0, frames_per_action, endpoint=False):
      positions.append(start + (target - start) * fraction)
      action_indices.append(index)
      fractions.append(fraction)

  positions.append(trajectory[-1])
  action_indices.append(len(trajectory) - 1)
  fractions.append(1.0)
  return (
      np.asarray(positions),
      np.asarray(action_indices),
      np.asarray(fractions),
  )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = left
  rw, rx, ry, rz = right
  return np.asarray([
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
  ])


def _euler_quaternion(rotation_delta: np.ndarray) -> np.ndarray:
  roll, pitch, yaw = rotation_delta
  cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
  cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
  cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
  return np.asarray([
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
  ])


def _action_states(
    episode: dict,
) -> tuple[np.ndarray, np.ndarray]:
  orientations = []
  orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
  gripper_values = []

  for step in episode["steps"]:
    actions = step["actions"]
    delta = _euler_quaternion(
        np.asarray(actions["rotation_delta"], dtype=np.float64)
    )
    orientation = _quaternion_multiply(orientation, delta)
    orientation /= np.linalg.norm(orientation)
    orientations.append(orientation.copy())
    gripper_values.append(float(actions["gripper_closedness_action"][0]))
  return np.asarray(orientations), np.asarray(gripper_values)


def _interpolate_quaternion(
    start: np.ndarray, target: np.ndarray, fraction: float
) -> np.ndarray:
  if np.dot(start, target) < 0:
    target = -target
  result = (1.0 - fraction) * start + fraction * target
  return result / np.linalg.norm(result)


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
  pure = np.asarray([0.0, *vector])
  conjugate = quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
  return _quaternion_multiply(
      _quaternion_multiply(quaternion, pure), conjugate
  )[1:]


def _set_gripper_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    quaternion: np.ndarray,
    closedness: float,
) -> None:
  body_names = ("cursor", "upper_finger", "lower_finger")
  mocap_ids = []
  for name in body_names:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_ids.append(model.body_mocapid[body_id])

  normalized_closedness = np.clip((closedness + 1.0) / 2.0, 0.0, 1.0)
  open_angle = np.deg2rad(35.0)
  closed_angle = np.deg2rad(3.0)
  finger_angle = (
      open_angle
      + normalized_closedness * (closed_angle - open_angle)
  )
  upper_hinge = _rotate_vector(
      quaternion, np.asarray([0.0, 0.0, 0.014])
  )
  lower_hinge = _rotate_vector(
      quaternion, np.asarray([0.0, 0.0, -0.014])
  )
  upper_rotation = np.asarray([
      np.cos(finger_angle / 2.0),
      0.0,
      -np.sin(finger_angle / 2.0),
      0.0,
  ])
  lower_rotation = np.asarray([
      np.cos(finger_angle / 2.0),
      0.0,
      np.sin(finger_angle / 2.0),
      0.0,
  ])
  upper_quaternion = _quaternion_multiply(quaternion, upper_rotation)
  lower_quaternion = _quaternion_multiply(quaternion, lower_rotation)

  data.mocap_pos[mocap_ids[0]] = position
  data.mocap_pos[mocap_ids[1]] = position + upper_hinge
  data.mocap_pos[mocap_ids[2]] = position + lower_hinge
  data.mocap_quat[mocap_ids[0]] = quaternion
  data.mocap_quat[mocap_ids[1]] = upper_quaternion
  data.mocap_quat[mocap_ids[2]] = lower_quaternion


def main() -> None:
  args = _parse_args()
  if args.fps <= 0 or args.action_hz <= 0:
    raise ValueError("--fps and --action-hz must be greater than zero")
  if args.width <= 0 or args.height <= 0:
    raise ValueError("--width and --height must be greater than zero")
  if args.zoom <= 0:
    raise ValueError("--zoom must be greater than zero")

  episode_path = args.episode_json.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  episode, trajectory = _load_trajectory(episode_path, args.scale)
  model = _build_model(trajectory)
  data = mujoco.MjData(model)

  frames_per_action = max(1, round(args.fps / args.action_hz))
  positions, action_indices, fractions = _interpolated_positions(
      trajectory, frames_per_action
  )
  orientations, gripper_values = _action_states(episode)
  camera = _camera_for(trajectory, args.zoom)
  frames: list[Image.Image] = []

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  try:
    for position, action_index, fraction in zip(
        positions, action_indices, fractions
    ):
      previous_index = max(0, action_index - 1)
      quaternion = _interpolate_quaternion(
          orientations[previous_index], orientations[action_index], fraction
      )
      closedness = (
          (1.0 - fraction) * gripper_values[previous_index]
          + fraction * gripper_values[action_index]
      )
      _set_gripper_pose(
          model, data, position, quaternion, closedness
      )
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=camera)
      rgb = renderer.render()
      frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE))
  finally:
    renderer.close()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  duration_ms = round(1000 / args.fps)
  frames[0].save(
      output_path,
      save_all=True,
      append_images=frames[1:],
      duration=duration_ms,
      loop=0,
      disposal=2,
  )

  print(f"Episode: {episode.get('episode_index', 'unknown')}")
  print(f"Instruction: {episode.get('instruction', '')}")
  print(f"Action points: {len(trajectory)}")
  print("Visualized controls: x, y, z, roll, pitch, yaw, gripper open/close")
  print(f"Rendered frames: {len(frames)}")
  print(f"Output: {output_path}")


if __name__ == "__main__":
  main()
