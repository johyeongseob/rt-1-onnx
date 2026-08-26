"""End-to-end RT-1 inference assembled from the validated ONNX modules."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from onnxruntime_extensions import get_library_path


ONNX_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ONNX_DIR / "film_efficientnet"))

from action_decoder import decode_action_tokens, extract_action_tokens  # pylint: disable=g-import-not-at-top
from preprossors import convert_dtype_and_crop_images, resize_images  # pylint: disable=g-import-not-at-top


TIME_STEPS = 6
IMAGE_TOKENS = 8
ACTION_TOKENS = 11
EMBEDDING_DIM = 512
TOKENS_PER_STEP = IMAGE_TOKENS + ACTION_TOKENS
SEQUENCE_LENGTH = TIME_STEPS * TOKENS_PER_STEP
DEFAULT_USE_MODEL = (
    ONNX_DIR.parent
    / "models"
    / "universal_sentence_encoder_large_onnx"
    / "5"
    / "model.onnx"
)


def _action_index(position: int) -> int:
  if position % TOKENS_PER_STEP < IMAGE_TOKENS:
    return -1
  return position // TOKENS_PER_STEP


def _attention_mask() -> npt.NDArray[np.float32]:
  mask = np.tril(
      np.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=np.float32)
  )
  for query in range(SEQUENCE_LENGTH):
    query_action = _action_index(query)
    if query_action == -1:
      continue
    for key in range(SEQUENCE_LENGTH):
      key_action = _action_index(key)
      if key_action == -1:
        continue
      if key_action < query_action:
        mask[query, key] = 0.0
      if key_action == query_action and key <= query:
        mask[query, key] = 0.0
  return mask


class RT1ONNXPipeline:
  """Connect the USE encoder and three RT-1 policy ONNX models."""

  def __init__(
      self,
      film_model: Path | str,
      token_learner_model: Path | str,
      transformer_model: Path | str,
      use_model: Path | str = DEFAULT_USE_MODEL,
  ) -> None:
    model_paths = {
        "USE Large /5": Path(use_model).expanduser().resolve(),
        "FiLM-EfficientNet": Path(film_model).expanduser().resolve(),
        "TokenLearner": Path(token_learner_model).expanduser().resolve(),
        "Transformer": Path(transformer_model).expanduser().resolve(),
    }
    for name, path in model_paths.items():
      if not path.is_file():
        raise FileNotFoundError(f"{name} ONNX model was not found: {path}")

    providers = ["CPUExecutionProvider"]
    use_options = ort.SessionOptions()
    use_options.register_custom_ops_library(get_library_path())
    self._use = ort.InferenceSession(
        str(model_paths["USE Large /5"]),
        sess_options=use_options,
        providers=providers,
    )
    self._film = ort.InferenceSession(
        str(model_paths["FiLM-EfficientNet"]), providers=providers
    )
    self._token_learner = ort.InferenceSession(
        str(model_paths["TokenLearner"]), providers=providers
    )
    self._transformer = ort.InferenceSession(
        str(model_paths["Transformer"]), providers=providers
    )
    self._attention_mask = _attention_mask()

  def encode_instructions(
      self, instructions: str | list[str] | tuple[str, ...]
  ) -> npt.NDArray[np.float32]:
    """Encode instruction strings with the ONNX USE Large /5 model."""
    if isinstance(instructions, str):
      instruction_batch = [instructions]
    elif isinstance(instructions, (list, tuple)) and instructions \
        and all(isinstance(value, str) for value in instructions):
      instruction_batch = list(instructions)
    else:
      raise TypeError(
          "instructions must be a string or a non-empty list of strings."
      )
    embeddings = self._use.run(
        ["outputs"], {"inputs": instruction_batch}
    )[0]
    if embeddings.shape != (len(instruction_batch), EMBEDDING_DIM):
      raise ValueError(
          "USE output must have shape [B, 512]; "
          f"received {embeddings.shape}."
      )
    return np.ascontiguousarray(embeddings, dtype=np.float32)

  def predict_instruction(
      self,
      images: npt.NDArray[np.uint8],
      instructions: str | list[str] | tuple[str, ...],
  ) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray]]:
    """Run RT-1 from six-frame histories and natural-language instructions."""
    embeddings = self.encode_instructions(instructions)
    if embeddings.shape[0] != images.shape[0]:
      raise ValueError(
          "The instruction batch size must match the image batch size; "
          f"received {embeddings.shape[0]} and {images.shape[0]}."
      )
    return self.predict(images, embeddings)

  def predict_episode_instruction(
      self,
      images: npt.NDArray[np.uint8],
      instructions: str | list[str] | tuple[str, ...],
  ) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray]]:
    """Run an episode from image sequences and natural-language instructions."""
    embeddings = self.encode_instructions(instructions)
    if embeddings.shape[0] != images.shape[0]:
      raise ValueError(
          "The instruction batch size must match the image batch size; "
          f"received {embeddings.shape[0]} and {images.shape[0]}."
      )
    return self.predict_episode(images, embeddings)

  def predict(
      self,
      images: npt.NDArray[np.uint8],
      language_embedding: npt.NDArray[np.float32],
  ) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray]]:
    """Run RT-1 for a six-frame image history and language embedding.

    Args:
      images: uint8 array with shape ``[B, 6, H, W, 3]``.
      language_embedding: float32 array with shape ``[B, 512]``.

    Returns:
      A pair containing ``[B, 11]`` action tokens and decoded actions.
    """
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
      raise TypeError("images must be a NumPy uint8 array.")
    if images.ndim != 5 or images.shape[1] != TIME_STEPS \
        or images.shape[-1] != 3:
      raise ValueError(
          "images must have shape [B, 6, H, W, 3]; "
          f"received {images.shape}."
      )
    if not isinstance(language_embedding, np.ndarray) \
        or language_embedding.dtype != np.float32:
      raise TypeError("language_embedding must be a NumPy float32 array.")
    if language_embedding.shape != (images.shape[0], EMBEDDING_DIM):
      raise ValueError(
          "language_embedding must have shape [B, 512]; "
          f"received {language_embedding.shape}."
      )

    image_tokens = self._encode_images(images, language_embedding)
    return self._predict_image_history(image_tokens)

  def predict_episode(
      self,
      images: npt.NDArray[np.uint8],
      language_embedding: npt.NDArray[np.float32],
  ) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray]]:
    """Run every frame using the official six-timestep rolling history."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
      raise TypeError("images must be a NumPy uint8 array.")
    if images.ndim != 5 or images.shape[1] < 1 or images.shape[-1] != 3:
      raise ValueError(
          "images must have shape [B, T, H, W, 3] with T >= 1; "
          f"received {images.shape}."
      )
    if not isinstance(language_embedding, np.ndarray) \
        or language_embedding.dtype != np.float32:
      raise TypeError("language_embedding must be a NumPy float32 array.")
    if language_embedding.shape != (images.shape[0], EMBEDDING_DIM):
      raise ValueError(
          "language_embedding must have shape [B, 512]; "
          f"received {language_embedding.shape}."
      )

    encoded = self._encode_images(images, language_embedding)
    batch_size, time_steps = encoded.shape[:2]
    history = np.zeros(
        (batch_size, TIME_STEPS, IMAGE_TOKENS, EMBEDDING_DIM),
        dtype=np.float32,
    )
    token_steps = []
    action_steps: dict[str, list[npt.NDArray]] = {}
    for time_index in range(time_steps):
      if time_index < TIME_STEPS:
        history[:, time_index] = encoded[:, time_index]
      else:
        history = np.roll(history, -1, axis=1)
        history[:, -1] = encoded[:, time_index]
      tokens, actions = self._predict_image_history(
          history, action_time=min(time_index, TIME_STEPS - 1)
      )
      token_steps.append(tokens)
      for key, value in actions.items():
        action_steps.setdefault(key, []).append(value)

    all_tokens = np.stack(token_steps, axis=1)
    all_actions = {
        key: np.stack(values, axis=1) for key, values in action_steps.items()
    }
    return all_tokens, all_actions

  def _encode_images(
      self,
      images: npt.NDArray[np.uint8],
      language_embedding: npt.NDArray[np.float32],
  ) -> npt.NDArray[np.float32]:
    """Encode an arbitrary number of timesteps into eight image tokens."""
    batch_size, time_steps = images.shape[:2]
    flat_images = np.ascontiguousarray(
        images.reshape(-1, *images.shape[2:])
    )
    flat_images = resize_images(
        convert_dtype_and_crop_images(flat_images), (300, 300)
    )
    contexts = np.repeat(
        language_embedding[:, np.newaxis, :], time_steps, axis=1
    ).reshape(batch_size * time_steps, EMBEDDING_DIM)
    contexts = np.ascontiguousarray(contexts, dtype=np.float32)

    features = self._film.run(
        None, {"image": flat_images, "context": contexts}
    )[0]
    image_tokens = self._token_learner.run(
        None, {"features": features}
    )[0]
    return np.ascontiguousarray(
        image_tokens.reshape(
            batch_size, time_steps, IMAGE_TOKENS, EMBEDDING_DIM
        ),
        dtype=np.float32,
    )

  def _predict_image_history(
      self,
      image_tokens: npt.NDArray[np.float32],
      action_time: int = TIME_STEPS - 1,
  ) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray]]:
    """Predict one action from exactly six timesteps of image tokens."""
    batch_size = image_tokens.shape[0]
    if image_tokens.shape[1:] != (
        TIME_STEPS, IMAGE_TOKENS, EMBEDDING_DIM
    ):
      raise ValueError(
          "image_tokens must have shape [B, 6, 8, 512]; "
          f"received {image_tokens.shape}."
      )
    action_slots = np.zeros(
        (batch_size, TIME_STEPS, ACTION_TOKENS, EMBEDDING_DIM),
        dtype=np.float32,
    )
    sequence = np.concatenate([image_tokens, action_slots], axis=2)
    sequence = np.ascontiguousarray(
        sequence.reshape(batch_size, SEQUENCE_LENGTH, EMBEDDING_DIM),
        dtype=np.float32,
    )
    logits = self._transformer.run(
        None,
        {"sequence": sequence, "attention_mask": self._attention_mask},
    )[0]
    logits = np.asarray(logits, dtype=np.float32)
    action_tokens = extract_action_tokens(logits, action_time=action_time)
    return action_tokens, decode_action_tokens(action_tokens)


__all__ = ["RT1ONNXPipeline"]
