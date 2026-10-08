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
#
import logging

import torch

logger = logging.getLogger(__name__)


def narrow(inp, dim, start, length):
    """Narrow a tensor along a dimension (kunlunxin / XPU, zero-copy view).

    ``torch.narrow`` is a view operation: the returned tensor shares storage
    with the input. The zero-copy ``torch.as_strided`` view (adjusting only the
    size along ``dim`` and the storage offset) is used instead of a copying
    kernel; on XPU this avoids the generic ``gems narrow`` python-impl
    dispatch and the native ``torch.narrow`` fallback inside ``unsafe_chunk``.
    """
    logger.debug("GEMS_KUNLUNXIN NARROW")
    assert (
        dim >= -inp.ndim and dim < inp.ndim
    ), f"Invalid dim: {dim} for tensor with {inp.ndim} dimensions"
    dim = dim % inp.ndim

    # `start` may be a 0-dim integral tensor.
    if isinstance(start, torch.Tensor):
        start = int(start.item())

    # Handle negative start.
    if start < 0:
        start = start + inp.size(dim)

    assert length >= 0 and start + length <= inp.size(
        dim
    ), f"Invalid narrow range: start={start}, length={length}, dim_size={inp.size(dim)}"

    size = list(inp.shape)
    size[dim] = length
    stride = list(inp.stride())
    storage_offset = inp.storage_offset() + start * inp.stride(dim)

    return torch.as_strided(inp, size, stride, storage_offset)
