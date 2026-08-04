# Copyright 2022 Google LLC
# Modifications Copyright 2026 rt-1-lab contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only image preprocessing for RT-1 ONNX."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


ImageBatch = npt.NDArray[np.uint8]
FloatImageBatch = npt.NDArray[np.float32]


def convert_dtype_and_crop_images(images: ImageBatch) -> FloatImageBatch:
  """Convert a uint8 BHWC image batch to float32 in the range [0, 1].

  Args:
    images: A NumPy uint8 array with shape ``[B, H, W, 3]``.

  Returns:
    A contiguous NumPy float32 array with the same shape.

  Raises:
    TypeError: If ``images`` is not a NumPy array with dtype ``uint8``.
    ValueError: If ``images`` is not a four-dimensional RGB image batch.
  """
  if not isinstance(images, np.ndarray):
    raise TypeError("images must be a NumPy array.")
  if images.dtype != np.uint8:
    raise TypeError(
        f"images must have dtype uint8; received {images.dtype}."
    )
  if images.ndim != 4 or images.shape[-1] != 3:
    raise ValueError(
        "images must have shape [B, H, W, 3]; "
        f"received {images.shape}."
    )

  scale = np.float32(1.0 / 255.0)
  return np.ascontiguousarray(images.astype(np.float32) * scale)


def resize_images(
    images: FloatImageBatch,
    size: tuple[int, int] = (300, 300),
) -> FloatImageBatch:
  """Resize a float32 BHWC image batch using bilinear interpolation.

  The coordinate transformation matches the half-pixel convention used by
  ``tf.image.resize`` with its default bilinear method.

  Args:
    images: A NumPy float32 array with shape ``[B, H, W, 3]``.
    size: Output ``(height, width)``.

  Returns:
    A contiguous NumPy float32 array with shape
    ``[B, size[0], size[1], 3]``.
  """
  if not isinstance(images, np.ndarray):
    raise TypeError("images must be a NumPy array.")
  if images.dtype != np.float32:
    raise TypeError(
        f"images must have dtype float32; received {images.dtype}."
    )
  if images.ndim != 4 or images.shape[-1] != 3:
    raise ValueError(
        "images must have shape [B, H, W, 3]; "
        f"received {images.shape}."
    )

  output_height, output_width = size
  if output_height <= 0 or output_width <= 0:
    raise ValueError(f"size values must be positive; received {size}.")

  input_height, input_width = images.shape[1:3]
  y_scale = np.float32(input_height / output_height)
  x_scale = np.float32(input_width / output_width)

  output_y = np.arange(output_height, dtype=np.float32)
  output_x = np.arange(output_width, dtype=np.float32)
  source_y = (output_y + np.float32(0.5)) * y_scale - np.float32(0.5)
  source_x = (output_x + np.float32(0.5)) * x_scale - np.float32(0.5)
  source_y = np.clip(source_y, 0.0, input_height - 1)
  source_x = np.clip(source_x, 0.0, input_width - 1)

  y0 = np.floor(source_y).astype(np.intp)
  x0 = np.floor(source_x).astype(np.intp)
  y1 = np.minimum(y0 + 1, input_height - 1)
  x1 = np.minimum(x0 + 1, input_width - 1)
  y_weight = (source_y - y0).astype(np.float32)[None, :, None, None]
  x_weight = (source_x - x0).astype(np.float32)[None, None, :, None]

  top_left = images[:, y0[:, None], x0[None, :], :]
  top_right = images[:, y0[:, None], x1[None, :], :]
  bottom_left = images[:, y1[:, None], x0[None, :], :]
  bottom_right = images[:, y1[:, None], x1[None, :], :]

  top = top_left + (top_right - top_left) * x_weight
  bottom = bottom_left + (bottom_right - bottom_left) * x_weight
  resized = top + (bottom - top) * y_weight
  return np.ascontiguousarray(resized, dtype=np.float32)


__all__ = ["convert_dtype_and_crop_images", "resize_images"]
