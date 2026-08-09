# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlunxin unfold_copy override.

unfold_copy(input, dimension, size, step) is the copy variant of
`input.unfold(dimension, size, step)`: it materializes the sliding-window
view into a fresh contiguous tensor. The view is exactly an as_strided view
(listed-window semantics), so we build the view with the same stride math as
Torch's `unfold` and then copy it into a contiguous output using the vendor
as_strided_copy machinery (block-DMA fast paths + generic strided Triton
kernels). This keeps correctness for arbitrary ndim / negative dim /
non-contiguous inputs / odd size-step combinations, which the generic
flag_gems implementation does not support (it only handles 2D-dim1 / 3D-dim1
/ 3D-dim2 contiguous inputs and silently mis-indexes non-contiguous ones).
"""

import logging

import torch

from flag_gems.runtime.backend._kunlunxin.ops.as_strided_copy import (
    _can_use_byte_triton,
    _can_use_triton,
    _launch_as_strided_copy,
    _launch_byte_as_strided_copy,
    _try_fast_copy,
)

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def _make_unfold_view(input, dimension, size, step):
    """Build the unfold view with Torch-compatible semantics/errors."""
    ndim = input.ndim
    if ndim == 0:
        dim_size = 1
        dimension = 0
    else:
        orig_dim = dimension
        if dimension < 0:
            dimension += ndim
        if dimension < 0 or dimension >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {orig_dim})"
            )
        dim_size = input.shape[dimension]

    if size > dim_size:
        raise RuntimeError(
            f"maximum size for tensor at dimension {dimension} is {dim_size} "
            f"but size is {size}"
        )

    n_windows = (dim_size - size) // step + 1

    if ndim == 0:
        new_shape = [n_windows, size]
        new_strides = [step, 1]
    else:
        new_shape = list(input.shape)
        new_shape[dimension] = n_windows
        new_shape.append(size)
        old_strides = list(input.stride())
        dim_stride = old_strides[dimension]
        new_strides = list(old_strides)
        new_strides[dimension] = step * dim_stride
        new_strides.append(dim_stride)

    return input.as_strided(new_shape, new_strides, input.storage_offset())


def unfold_copy(input, dimension, size, step):
    logger.debug("GEMS_KUNLUNXIN UNFOLD_COPY")
    if step <= 0:
        raise RuntimeError(f"step is {step} but must be > 0")

    view = _make_unfold_view(input, dimension, int(size), int(step))

    out_shape = tuple(view.shape)
    contiguous_stride = torch.empty(out_shape, device="meta").stride()
    out = torch.empty_strided(
        out_shape, contiguous_stride, dtype=input.dtype, device=input.device
    )

    if view.numel() == 0:
        return out

    if _try_fast_copy(view, out):
        return out
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    raise NotImplementedError(
        "Kunlunxin unfold_copy does not support this stride layout."
    )