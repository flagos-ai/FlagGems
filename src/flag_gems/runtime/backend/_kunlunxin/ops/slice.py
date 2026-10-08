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

import logging

import torch

logger = logging.getLogger(__name__)


def slice(
    input_tensor: torch.Tensor, dim: int, start, end, step: int = 1
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SLICE")

    if step == 0:
        raise RuntimeError("slice step cannot be 0")

    ndim = input_tensor.ndim
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = dim % ndim  # normalize negative dim
    dim_size = input_tensor.size(dim)

    # start/end may arrive as 0-dim integral tensors (from narrow-style
    # callsites); normalize to plain ints for the arithmetic below.
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    if step > 0:
        if start is None:
            start = 0
        elif start < 0:
            start = dim_size + start
        start = max(0, min(start, dim_size))

        if end is None:
            end = dim_size
        elif end < 0:
            end = dim_size + end
        end = max(0, min(end, dim_size))

        length = max(0, (end - start + step - 1) // step)

        if step == 1:
            # ``torch.narrow`` is a registered view-based impl (zero-copy via
            # ``as_strided``); equivalent to ``input_tensor[.., start:end, ..]``.
            return torch.narrow(input_tensor, dim, start, length)
    else:
        # Negative step: element positions start, start+step, ... > end.
        if start is None:
            start = dim_size - 1
        elif start < 0:
            start = dim_size + start
        start = max(-1, min(start, dim_size - 1))

        if end is None:
            end = -1
        elif end < 0:
            end = dim_size + end
        end = max(-1, min(end, dim_size - 1))

        length = max(0, (start - end + (-step) - 1) // (-step))

    # General (strided) zero-copy view.
    size = list(input_tensor.shape)
    size[dim] = length
    stride = list(input_tensor.stride())
    stride[dim] = input_tensor.stride(dim) * step
    storage_offset = input_tensor.storage_offset() + start * input_tensor.stride(dim)

    return torch.as_strided(input_tensor, size, stride, storage_offset)
