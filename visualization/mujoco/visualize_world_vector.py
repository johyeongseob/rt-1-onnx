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
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="135" elevation="-25"/>
    <rgba haze="0.15 0.20 0.25 1"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22"
             rgb2="0.24 0.26 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="robot_blue" rgba="0.12 0.32 0.55 1"
              specular="0.55" shininess="0.65"/>
    <material name="joint_dark" rgba="0.055 0.065 0.075 1"
              specular="0.35" shininess="0.45"/>
    <material name="metal" rgba="0.62 0.67 0.72 1"
              specular="0.8" shininess="0.9"/>
    <material name="gripper" rgba="0.28 0.31 0.34 1"
              specular="0.65" shininess="0.75"/>
    <material name="rubber" rgba="0.025 0.03 0.035 1"
              specular="0.05" shininess="0.1"/>
    <material name="status_green" rgba="0.15 0.9 0.38 1"
              emission="0.35"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.5 1.4" dir="-0.2 0.3 -1"
           diffuse="0.85 0.85 0.85" castshadow="true"/>
    <light pos="-0.5 0.4 0.8" dir="0.5 -0.3 -0.5"
           diffuse="0.35 0.40 0.48"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <body name="cursor" mocap="true" pos="0 0 0.1">
      <!-- A compact industrial-robot silhouette attached to the RT-1
           end-effector pose. It is schematic, not an EDR kinematic model. -->
      <geom name="forearm" type="capsule"
            fromto="-0.060 0 0 -0.205 0 0" size="0.031"
            material="robot_blue" contype="0" conaffinity="0"/>
      <geom name="elbow_joint" type="cylinder" pos="-0.205 0 0"
            size="0.043 0.034" euler="90 0 0" material="joint_dark"
            contype="0" conaffinity="0"/>
      <geom name="elbow_hub" type="cylinder" pos="-0.205 0 0"
            size="0.035 0.038" euler="90 0 0" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="upper_arm" type="capsule"
            fromto="-0.218 0 -0.018 -0.335 0 -0.095" size="0.038"
            material="robot_blue" contype="0" conaffinity="0"/>
      <geom name="shoulder_joint" type="cylinder" pos="-0.335 0 -0.095"
            size="0.052 0.043" euler="90 0 0" material="joint_dark"
            contype="0" conaffinity="0"/>
      <geom name="shoulder_hub" type="cylinder" pos="-0.335 0 -0.095"
            size="0.043 0.047" euler="90 0 0" material="robot_blue"
            contype="0" conaffinity="0"/>
      <geom name="shoulder_mount" type="box" pos="-0.370 0 -0.130"
            size="0.040 0.052 0.030" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="wrist_body" type="cylinder" pos="-0.040 0 0"
            size="0.039 0.032" euler="0 90 0" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="wrist_ring" type="cylinder" pos="-0.004 0 0"
            size="0.043 0.008" euler="0 90 0" material="joint_dark"
            contype="0" conaffinity="0"/>
      <geom name="tool_flange" type="cylinder" pos="0.010 0 0"
            size="0.034 0.009" euler="0 90 0" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="gripper_palm" type="box" pos="0.026 0 0"
            size="0.018 0.027 0.031" material="gripper"
            contype="0" conaffinity="0"/>
      <geom name="status_light" type="sphere" pos="0.030 -0.028 0"
            size="0.004" material="status_green"
            contype="0" conaffinity="0"/>
    </body>
    <body name="upper_finger" mocap="true" pos="0 0 0.014">
      <geom name="upper_knuckle" type="cylinder" pos="0 0 0"
            size="0.014 0.017" euler="90 0 0" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="upper_jaw" type="capsule"
            fromto="0.004 0 0 0.071 0 0" size="0.010"
            material="gripper" contype="0" conaffinity="0"/>
      <geom name="upper_tip" type="box" pos="0.074 0 -0.006"
            size="0.014 0.012 0.008" material="gripper"
            contype="0" conaffinity="0"/>
      <geom name="upper_pad" type="box" pos="0.075 0 -0.015"
            size="0.013 0.011 0.003" material="rubber"
            contype="0" conaffinity="0"/>
    </body>
    <body name="lower_finger" mocap="true" pos="0 0 -0.014">
      <geom name="lower_knuckle" type="cylinder" pos="0 0 0"
            size="0.014 0.017" euler="90 0 0" material="metal"
            contype="0" conaffinity="0"/>
      <geom name="lower_jaw" type="capsule"
            fromto="0.004 0 0 0.071 0 0" size="0.010"
            material="gripper" contype="0" conaffinity="0"/>
      <geom name="lower_tip" type="box" pos="0.074 0 0.006"
            size="0.014 0.012 0.008" material="gripper"
            contype="0" conaffinity="0"/>
      <geom name="lower_pad" type="box" pos="0.075 0 0.015"
            size="0.013 0.011 0.003" material="rubber"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
  return mujoco.MjModel.from_xml_string(xml)


def _play(model: mujoco.MjModel, trajectory: np.ndarray, fps: float, once: bool) -> None:
  if fps <= 0:
    raise ValueError("--fps must be greater than zero")

  data = mujoco.MjData(model)
  mocap_ids = {}
  for name in ("cursor", "upper_finger", "lower_finger"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_ids[name] = model.body_mocapid[body_id]
  frame_duration = 1.0 / fps

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      for index, position in enumerate(trajectory):
        if not viewer.is_running():
          return

        start = time.monotonic()
        data.mocap_pos[mocap_ids["cursor"]] = position
        data.mocap_pos[mocap_ids["upper_finger"]] = (
            position + np.asarray([0.0, 0.0, 0.014])
        )
        data.mocap_pos[mocap_ids["lower_finger"]] = (
            position + np.asarray([0.0, 0.0, -0.014])
        )
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
  print("Schematic industrial gripper: current end-effector position")
  _play(model, trajectory, args.fps, args.once)


if __name__ == "__main__":
  main()
