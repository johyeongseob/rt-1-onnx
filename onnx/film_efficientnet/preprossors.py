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


__all__ = ["convert_dtype_and_crop_images"]
