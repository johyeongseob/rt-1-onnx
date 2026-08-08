"""NumPy decoding for RT-1 action tokens."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


VOCAB_SIZE = 256
ACTION_LOGIT_START = 102
ACTION_TOKENS = 11


def extract_action_tokens(
    logits: npt.NDArray[np.float32],
    action_time: int = 5,
) -> npt.NDArray[np.int64]:
  """Select and argmax the current timestep's eleven action logits."""
  if logits.ndim != 3 or logits.shape[1:] != (114, VOCAB_SIZE):
    raise ValueError(
        f"logits must have shape [B, 114, 256]; received {logits.shape}."
    )
  if action_time < 0 or action_time >= 6:
    raise ValueError(f"action_time must be in [0, 5]; received {action_time}.")
  action_logit_start = 7 + action_time * 19
  action_logits = logits[:, action_logit_start:action_logit_start + ACTION_TOKENS, :]
  return np.argmax(action_logits, axis=-1)


def _decode_float(
    tokens: npt.NDArray[np.int64], minimum: float, maximum: float
) -> npt.NDArray[np.float32]:
  values = tokens.astype(np.float32) / np.float32(VOCAB_SIZE - 1)
  return values * np.float32(maximum - minimum) + np.float32(minimum)


def decode_action_tokens(
    tokens: npt.NDArray[np.int64],
) -> dict[str, npt.NDArray]:
  """Convert eleven RT-1 vocabulary tokens into robot action values."""
  if tokens.ndim != 2 or tokens.shape[1] != ACTION_TOKENS:
    raise ValueError(
        f"tokens must have shape [B, 11]; received {tokens.shape}."
    )

  terminate = tokens[:, 0]
  terminate = np.where(terminate < 3, terminate, 0)
  return {
      "terminate_episode": np.eye(3, dtype=np.int32)[terminate],
      "world_vector": _decode_float(tokens[:, 1:4], -1.0, 1.0),
      "rotation_delta": _decode_float(
          tokens[:, 4:7], -np.pi / 2.0, np.pi / 2.0
      ),
      "gripper_closedness_action": _decode_float(
          tokens[:, 7:8], -1.0, 1.0
      ),
      "base_displacement_vector": _decode_float(tokens[:, 8:10], -1.0, 1.0),
      "base_displacement_vertical_rotation": _decode_float(
          tokens[:, 10:11], -np.pi, np.pi
      ),
  }


__all__ = ["decode_action_tokens", "extract_action_tokens"]
